"""
Step 6 — Stratified K-Fold Validation Training Script.

Implements 5-fold stratified cross-validation to ensure:
- No data leakage between train and val folds.
- Each fold preserves the class distribution.
- The best fold model (by Macro F1) is saved as the final submission model.
"""

import sys
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from train_pipeline import (
    build_model,
    CSVImageDataset,
    get_train_transforms,
    get_val_transforms,
    FocalLoss,
    EarlyStopping,
    mixup_data,
    mixup_criterion
)


# ─── Config ────────────────────────────────────────────────────────────────────
CSV_PATH  = Path("Training+Testing_data_label.csv")
ROOT_DIR  = Path("training_preprocessed")
MODEL_OUT = Path("best_model_kfold.pth")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

N_SPLITS    = 5
BATCH_SIZE  = 16
NUM_EPOCHS  = 50
IMAGE_SIZE  = 224
NUM_CLASSES = 5
NUM_WORKERS = 0  # set to 0 for Windows compatibility


def build_weighted_sampler(labels, indices):
    """Build a WeightedRandomSampler from a subset of labels."""
    subset_labels = np.array([labels[i] for i in indices])
    class_counts = np.bincount(subset_labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / np.where(class_counts == 0, 1, class_counts)
    sample_weights = class_weights[subset_labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def run_kfold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load full dataset with training transforms (we'll override per split)
    full_dataset = CSVImageDataset(
        csv_path=CSV_PATH,
        root_dir=ROOT_DIR,
        transform=None,  # transforms applied per-split below
    )

    all_labels = np.array([item[1] for item in full_dataset.items])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_results = []
    best_f1  = -1.0
    best_state = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{N_SPLITS}")
        print(f"{'='*60}")

        # Build per-split datasets with correct transforms
        train_dataset = CSVImageDataset(
            csv_path=CSV_PATH, root_dir=ROOT_DIR,
            transform=get_train_transforms(IMAGE_SIZE)
        )
        val_dataset = CSVImageDataset(
            csv_path=CSV_PATH, root_dir=ROOT_DIR,
            transform=get_val_transforms(IMAGE_SIZE)
        )

        train_subset = Subset(train_dataset, train_idx)
        val_subset   = Subset(val_dataset,   val_idx)

        # WeightedRandomSampler for training fold
        sampler = build_weighted_sampler(all_labels, train_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        # Fresh model per fold
        model = build_model(
            model_name="convnext_small",
            num_classes=NUM_CLASSES,
            pretrained=True,
        ).to(device)

        # Class weights for FocalLoss from training fold
        train_labels = all_labels[train_idx]
        counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
        alpha_weights = 1.0 / np.where(counts == 0, 1, counts)
        alpha_weights = alpha_weights / alpha_weights.sum() * NUM_CLASSES
        alpha_tensor = torch.FloatTensor(alpha_weights).to(device)
        criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0, smoothing=0.1)

        # Differential LR (Phase 1: freeze backbone, Phase 2: unfreeze)
        head_params = list(model.head.parameters())
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith('head')]

        import torch.optim as optim
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 2e-5},  # Lower backbone LR to prevent catastrophic forgetting
            {'params': head_params,     'lr': 1e-3},
        ], weight_decay=1e-2)  # Increased weight decay to combat overfitting

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        early_stopping = EarlyStopping(
            patience=7, mode='max',
            path=str(REPORT_DIR / f"fold_{fold+1}_best.pth")
        )

        # Phase 1: freeze backbone
        print("Phase 1: Training classifier head only...")
        for p in backbone_params:
            p.requires_grad = False

        for epoch in range(NUM_EPOCHS):
            # Phase 2: unfreeze backbone after epoch 5
            if epoch == 5:
                print("Phase 2: Full fine-tuning (backbone unfrozen)...")
                for p in backbone_params:
                    p.requires_grad = True

            # ── TRAIN ──────────────────────────────────────────────────────
            model.train()
            running_loss, all_preds_t, all_labels_t = 0.0, [], []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                # Apply MixUp regularization to prevent overfitting
                if np.random.random() > 0.5 and len(inputs) > 1:
                    inputs_m, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.4, device=device)
                    outputs = model(inputs_m)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds_t.extend(preds.cpu().numpy())
                all_labels_t.extend(labels.cpu().numpy())

            train_loss = running_loss / len(train_loader)
            train_f1   = f1_score(all_labels_t, all_preds_t, average='macro', zero_division=0)
            train_acc  = 100 * (np.array(all_labels_t) == np.array(all_preds_t)).mean()

            # ── VALIDATE ───────────────────────────────────────────────────
            model.eval()
            val_loss, all_preds_v, all_labels_v = 0.0, [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    preds = outputs.argmax(dim=1)
                    all_preds_v.extend(preds.cpu().numpy())
                    all_labels_v.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)
            val_f1    = f1_score(all_labels_v, all_preds_v, average='macro', zero_division=0)
            val_acc   = 100 * (np.array(all_labels_v) == np.array(all_preds_v)).mean()

            print(
                f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} "
                f"| Train Loss {train_loss:.4f}  Acc {train_acc:.1f}%  F1 {train_f1:.4f} "
                f"| Val Loss {val_loss:.4f}  Acc {val_acc:.1f}%  F1 {val_f1:.4f}"
            )

            scheduler.step()
            early_stopping(val_f1, model)
            if early_stopping.early_stop:
                print("  Early stopping triggered.")
                break

        # ── FOLD EVALUATION ────────────────────────────────────────────────
        best_fold_state = torch.load(str(REPORT_DIR / f"fold_{fold+1}_best.pth"), map_location=device)
        model.load_state_dict(best_fold_state)
        model.eval()

        fold_preds, fold_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                fold_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                fold_labels.extend(labels.cpu().numpy())

        fold_f1  = f1_score(fold_labels, fold_preds, average='macro', zero_division=0)
        fold_acc = 100 * (np.array(fold_labels) == np.array(fold_preds)).mean()
        fold_results.append({'fold': fold+1, 'val_f1': fold_f1, 'val_acc': fold_acc})

        print(f"\n  Fold {fold+1} Best → Val Acc: {fold_acc:.2f}%  Macro F1: {fold_f1:.4f}")

        # Per-class report
        print(classification_report(fold_labels, fold_preds,
              target_names=['No DR','Mild','Moderate','Severe','PDR'], zero_division=0))

        # Confusion matrix plot
        cm = confusion_matrix(fold_labels, fold_preds)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No DR','Mild','Moderate','Severe','PDR'],
                    yticklabels=['No DR','Mild','Moderate','Severe','PDR'])
        plt.title(f'Fold {fold+1} Confusion Matrix  (F1={fold_f1:.4f})')
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(REPORT_DIR / f"fold_{fold+1}_confusion.png")
        plt.close()

        # Track best fold
        if fold_f1 > best_f1:
            best_f1 = fold_f1
            best_state = copy.deepcopy(best_fold_state)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  K-FOLD RESULTS SUMMARY")
    print("="*60)
    results_df = pd.DataFrame(fold_results)
    print(results_df.to_string(index=False))
    print(f"\n  Mean Val Macro F1 : {results_df['val_f1'].mean():.4f} ± {results_df['val_f1'].std():.4f}")
    print(f"  Best Fold F1      : {results_df['val_f1'].max():.4f} (Fold {results_df.loc[results_df['val_f1'].idxmax(), 'fold']})")

    # Save the best model
    torch.save(best_state, MODEL_OUT)
    print(f"\n  Best model saved → {MODEL_OUT}")
    results_df.to_csv(REPORT_DIR / "kfold_results.csv", index=False)


if __name__ == "__main__":
    run_kfold()
