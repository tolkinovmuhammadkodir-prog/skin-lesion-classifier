"""
gradcam.py — Grad-CAM explainability: shows which image regions most
influenced the model's prediction for a given class.
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F


def grad_cam(model, image_tensor, target_layer, class_idx):
    """
    Args:
        image_tensor: preprocessed image, shape [1, 3, 224, 224], with
            requires_grad_(True) already set.
        target_layer: the last convolutional layer (e.g. model.layer4[-1]).
        class_idx: which class's heatmap to generate.
    Returns:
        A [7, 7]-ish normalized heatmap as a numpy array (0-1 range).
    """
    activations, gradients = [], []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    model.eval()
    output = model(image_tensor)
    score = output[0, class_idx]
    model.zero_grad()
    score.backward()

    h1.remove()
    h2.remove()

    grads = gradients[0][0]
    acts = activations[0][0]

    weights = grads.mean(dim=(1, 2))
    cam = torch.zeros(acts.shape[1:], device=acts.device)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = F.relu(cam)
    cam = cam / cam.max()
    return cam.detach().cpu().numpy()


def make_overlay(image_tensor, cam, alpha: float = 0.4):
    """Upscales the Grad-CAM heatmap and blends it over the original image."""
    img = image_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * 0.5) + 0.5
    img = np.clip(img, 0, 1)

    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = heatmap * alpha + img * (1 - alpha)
    return (overlay * 255).astype(np.uint8)
