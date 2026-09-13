"""
app.py — Gradio web interface for the deployed classifier.
Run with: python app.py  (or in a Colab/Kaggle cell)
"""
import torch
import gradio as gr

from dataset import get_transforms
from model import build_model
from gradcam import grad_cam, make_overlay

# ---- Load trained model ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # sorted order from training

model = build_model(num_classes=len(CLASS_NAMES), device=device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

transform = get_transforms()


def predict(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    image_tensor.requires_grad_(True)

    output = model(image_tensor)
    _, pred_idx = torch.max(output, 1)
    pred_class = CLASS_NAMES[pred_idx.item()]

    cam = grad_cam(model, image_tensor, model.layer4[-1], class_idx=pred_idx.item())
    overlay = make_overlay(image_tensor, cam)

    return pred_class, overlay


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Prediction"), gr.Image(label="Grad-CAM Explanation")],
    title="Explainable Skin Lesion Classifier",
    description="Upload a dermoscopic image. The model predicts one of 7 diagnostic "
                 "categories from HAM10000, and Grad-CAM highlights which region of "
                 "the image most influenced that decision."
)

if __name__ == "__main__":
    demo.launch(debug=True)
