"""
dataset.py — HAM10000 Dataset loading and preprocessing
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class HAM10000Dataset(Dataset):
    """Custom PyTorch Dataset for the HAM10000 skin lesion dataset."""

    def __init__(self, metadata: pd.DataFrame, img_dirs: list, transform=None):
        self.metadata = metadata.reset_index(drop=True)
        self.img_dirs = img_dirs
        self.transform = transform
        self.classes = sorted(metadata['dx'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = row['image_id'] + '.jpg'
        full_path = None
        for d in self.img_dirs:
            candidate = os.path.join(d, img_name)
            if os.path.exists(candidate):
                full_path = candidate
                break
        if full_path is None:
            raise FileNotFoundError(f"Image {img_name} not found in provided directories.")

        image = Image.open(full_path).convert('RGB')
        label = self.class_to_idx[row['dx']]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    """Standard preprocessing pipeline matching ResNet-18's expected input."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_class_weights(metadata: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency class weights to counteract severe class imbalance
    (e.g. 'nv' at 67% of the dataset vs 'df' at under 2%)."""
    class_counts = metadata['dx'].value_counts().sort_index()
    weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float)
    return weights / weights.sum()


def build_dataloaders(metadata: pd.DataFrame, img_dirs: list,
                       batch_size: int = 32, val_split: float = 0.2, seed: int = 42):
    """Builds train/val DataLoaders from the full HAM10000 dataset."""
    transform = get_transforms()
    full_dataset = HAM10000Dataset(metadata, img_dirs, transform=transform)

    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, full_dataset
