"""
train.py — Multi-epoch training loop with checkpointing.

IMPORTANT: earlier development only ran ONE epoch (64.05% val accuracy).
This script trains for multiple epochs and saves the BEST checkpoint by
validation accuracy, since more training will very likely improve results
meaningfully before this goes on a portfolio.
"""
import logging
import torch
import torch.nn as nn
import pandas as pd

from dataset import build_dataloaders, get_class_weights
from model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S", force=True)
logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch_num, total_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 40 == 0:
            logger.info(f"Epoch {epoch_num}/{total_epochs} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    logger.info(f"Epoch {epoch_num}/{total_epochs} complete | Avg Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def main(metadata_path: str, img_dirs: list, num_epochs: int = 10,
         lr: float = 0.001, checkpoint_path: str = "best_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    metadata = pd.read_csv(metadata_path)
    train_loader, val_loader, full_dataset = build_dataloaders(metadata, img_dirs)
    num_classes = len(full_dataset.classes)
    logger.info(f"Dataset ready: {len(full_dataset)} images, {num_classes} classes")

    model = build_model(num_classes, device)
    weights = get_class_weights(metadata).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        val_acc = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch}/{num_epochs} | Val Accuracy: {val_acc:.2f}%")
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_accuracy": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"New best model saved (val acc: {val_acc:.2f}%)")

    logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    return history, best_val_acc


if __name__ == "__main__":
    # Adjust these paths to match your Kaggle/Colab environment
    METADATA_PATH = "HAM10000_metadata.csv"
    IMG_DIRS = ["ham10000_images_part_1", "ham10000_images_part_2"]
    main(METADATA_PATH, IMG_DIRS, num_epochs=10)
