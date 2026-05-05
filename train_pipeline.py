from pathlib import Path
from typing import Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
import torch.nn.functional as F
import timm


def get_train_transforms(image_size: int = 224):
    """Build the training augmentation pipeline using albumentations."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224):
    """Build the validation preprocessing pipeline using albumentations."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ])


def build_imagenet_transform(image_size: int = 224):
    """Build the standard ImageNet preprocessing transform (legacy support)."""
    from torchvision import transforms
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


def build_model(model_name: str = "convnext_small", num_classes: int = 5, pretrained: bool = True):
    """Build a model for image classification using timm."""
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required to build the model. "
            "Install it with `pip install timm` in the active environment."
        ) from exc

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


class CSVImageDataset:
    """Dataset wrapper to load images and labels from a CSV file."""

    def __init__(
        self,
        csv_path: Path,
        root_dir: Path,
        transform=None,
        image_column: str = "image",
        label_column: str = "label",
    ):
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(csv_path)

        # Normalize CSV headers and string values in case the file contains quoted names.
        df.columns = df.columns.str.strip().str.strip('"').str.strip("'")

        # Create case-insensitive column mapping
        columns_lower = {col.lower(): col for col in df.columns}

        if image_column.lower() not in columns_lower or label_column.lower() not in columns_lower:
            raise ValueError(
                f"CSV file must contain '{image_column}' and '{label_column}' columns. "
                f"Found columns: {list(df.columns)}"
            )

        # Get the actual column names (with correct casing)
        actual_image_column = columns_lower[image_column.lower()]
        actual_label_column = columns_lower[label_column.lower()]

        image_series = df[actual_image_column].astype(str).str.strip().str.strip('"').str.strip("'")
        label_series = df[actual_label_column]

        items = []
        missing = []
        for image_name, label in zip(image_series.tolist(), label_series.tolist()):
            image_path = self.root_dir / str(image_name)
            if image_path.exists():
                items.append((image_name, label))
            else:
                missing.append(image_name)

        if missing:
            import warnings
            warnings.warn(
                f"{len(missing)} image paths from CSV were not found in {self.root_dir} and will be skipped. "
                f"Missing examples: {missing[:10]}{'...' if len(missing) > 10 else ''}",
                UserWarning,
            )

        if not items:
            raise ValueError(
                f"No valid image files found in CSV at {csv_path} under {self.root_dir}."
            )

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        from PIL import Image

        image_name, label = self.items[idx]
        image_path = self.root_dir / str(image_name)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            # Albumentations expects numpy arrays
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = augmented["image"]

        return image, int(label)


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, smoothing: float = 0.1, reduction: str = 'mean'):
        """
        Focal Loss with Label Smoothing.
        Args:
            alpha: Class weights (tensor).
            gamma: Focusing parameter.
            smoothing: Label smoothing factor.
            reduction: 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        
        # Apply label smoothing
        log_probs = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.smoothing)
        
        # Standard Cross Entropy with soft labels
        ce_loss = torch.sum(-true_dist * log_probs, dim=-1)
        
        # Focal factor calculation
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            alpha_weights = self.alpha[targets]
            focal_loss = focal_loss * alpha_weights
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def build_dataloader(
    csv_path: Path,
    root_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    is_training: bool = True,
    use_weighted_sampler: bool = True,
):
    """Build a PyTorch DataLoader with optional WeightedRandomSampler."""
    transform = get_train_transforms(image_size=image_size) if is_training else get_val_transforms(image_size=image_size)
    dataset = CSVImageDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        transform=transform,
    )

    sampler = None
    shuffle = is_training

    if is_training and use_weighted_sampler:
        labels = [item[1] for item in dataset.items]
        class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        shuffle = False

    pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=7, min_delta=0, mode='max', path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.path = path

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif (self.mode == 'max' and score < self.best_score + self.min_delta) or \
             (self.mode == 'min' and score > self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


def train_model(
    model,
    dataloader,
    val_dataloader=None,
    num_epochs: int = 50,
    device: Optional[str] = None,
):
    """Train the ResNet50 model on the dataset."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    
    # Calculate class weights for Alpha in Focal Loss
    labels = [item[1] for item in dataloader.dataset.items]
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weights = 1. / class_sample_count
    weights = weights / weights.sum() * len(weights) # normalize
    alpha = torch.FloatTensor(weights).to(device)
    
    criterion = FocalLoss(alpha=alpha, gamma=2.0, smoothing=0.1)
    
    # Differential Learning Rates
    head_params = list(model.head.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith('head')]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stopping = EarlyStopping(patience=7, mode='max')

    # Phase 1: Freeze backbone
    print("Phase 1: Training classifier head only (backbone frozen)...")
    for p in backbone_params:
        p.requires_grad = False

    for epoch in range(num_epochs):
        # Phase 2: Unfreeze backbone after 5 epochs
        if epoch == 5:
            print("Phase 2: Full fine-tuning (backbone unfrozen)...")
            for p in backbone_params:
                p.requires_grad = True

        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply MixUp with probability 0.5 (optional hyperparameter)
            if np.random.random() > 0.5 and len(inputs) > 1:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.4, device=device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                loss.backward()
                optimizer.step()
                
                # For metrics, track predictions
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * (np.array(all_labels) == np.array(all_preds)).mean()
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{num_epochs} [Train] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Macro F1: {epoch_f1:.4f}")
        
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            val_labels = []
            val_preds = []
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_labels.extend(labels.cpu().numpy())
                    val_preds.extend(predicted.cpu().numpy())
            
            val_epoch_loss = val_loss / len(val_dataloader)
            val_epoch_acc = 100 * (np.array(val_labels) == np.array(val_preds)).mean()
            val_epoch_f1 = f1_score(val_labels, val_preds, average='macro')
            
            print(f"          [Val]   Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.2f}%, Macro F1: {val_epoch_f1:.4f}")
            
            scheduler.step()
            early_stopping(val_epoch_f1, model)
            
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        else:
            scheduler.step()

    return model


def prepare_baseline(
    csv_path: Optional[Path] = None,
    root_dir: Optional[Path] = None,
    num_classes: int = 1000,
    batch_size: int = 32,
    image_size: int = 224,
):
    """Prepare the ResNet50 baseline model and data loader without training."""
    if csv_path is None:
        csv_path = Path(__file__).parent / "Training+Testing_data_label.csv"
    if root_dir is None:
        root_dir = Path(__file__).parent / "Training+Testing_data"

    model = build_model(model_name="convnext_small", num_classes=num_classes, pretrained=True)
    dataloader = build_dataloader(
        csv_path=csv_path,
        root_dir=root_dir,
        batch_size=batch_size,
        image_size=image_size,
        is_training=True,
        use_weighted_sampler=True,
    )

    return model, dataloader


if __name__ == "__main__":
    print("Preparing ResNet50 baseline model and dataloader...")
    model, dataloader = prepare_baseline()
    print("Starting training...")
    trained_model = train_model(model, dataloader, num_epochs=5)
    print("Training completed. Model is ready for evaluation or further use.")

