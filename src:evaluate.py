"""
evaluate.py — Per-class evaluation (accuracy alone is misleading on this
imbalanced dataset, per our Day 8/11 findings).
"""
import torch
from sklearn.metrics import classification_report


def full_evaluation(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    return report
