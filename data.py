import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def build_train_transforms(img_size: int = 28) -> T.Compose:
    """
    ASL-appropriate augmentation pipeline.

    Design decisions grounded in domain knowledge:
      - Horizontal flip  ✓  Signs are valid when mirrored (left/right-hand)
      - Vertical flip    ✗  Signers are never upside down
      - Rotation ±15°   ✓  Realistic hand-position variance
      - Zoom/crop       ✓  Partial occlusion is common in real capture
      - Brightness/contrast ✓  Real-world lighting is inconsistent
    """
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomResizedCrop(size=img_size, scale=(0.85, 1.0)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])


def build_val_transforms(img_size: int = 28) -> T.Compose:
    """Deterministic transforms — no randomness for validation or test."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_data_loaders(
    train_dir: str,
    valid_dir: str,
    img_size: int = 28,
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Build and return (train_loader, valid_loader) from directory paths."""
    train_ds = ImageFolder(train_dir, transform=build_train_transforms(img_size))
    valid_ds = ImageFolder(valid_dir, transform=build_val_transforms(img_size))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, valid_loader
