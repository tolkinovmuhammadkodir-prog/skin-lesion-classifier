# Explainable Skin Lesion Classifier

A deep learning classifier for dermoscopic skin lesion images (HAM10000 dataset),
built with transfer learning and Grad-CAM explainability, deployed as a live web app.

**[Live Demo](#)** ← replace with your Hugging Face Spaces link after permanent deployment

## Problem

Skin lesion classification is a 7-class diagnostic task with severe real-world class
imbalance — the majority class (`nv`) makes up ~67% of the HAM10000 dataset, while
the rarest class (`df`) is under 2%. A naive model can achieve misleadingly high
accuracy by effectively ignoring rare, clinically important classes entirely.

## Approach

- **Transfer learning**: ResNet-18 pretrained on ImageNet, backbone frozen, only the
  final layer retrained — reusing general visual features instead of training from
  scratch on a relatively small medical dataset.
- **Weighted loss**: inverse-frequency class weighting in Cross-Entropy loss to
  counteract the imbalance (rare classes weighted up to ~58x higher than the
  majority class), rather than letting the model default to ignoring them.
- **Explainability**: Grad-CAM generates a heatmap for every prediction, showing
  which region of the image most influenced the model's decision — critical for
  trust in a medical context, where a clinician needs to verify model reasoning,
  not just accept a raw label.
- **Deployment**: wrapped in a Gradio interface for live, interactive inference.

## Results

<!-- UPDATE THIS SECTION after running train.py for multiple epochs -->
| Metric | Value |
|---|---|
| Overall validation accuracy | *(update after multi-epoch training)* |
| Training epochs | *(update)* |

**Per-class precision/recall (initial 1-epoch baseline, to be updated):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| nv (majority) | 0.93 | 0.71 | 0.80 |
| mel (melanoma) | 0.27 | 0.54 | 0.36 |
| vasc (rare) | 0.11 | 0.65 | 0.19 |

**Key finding:** weighted loss significantly improved recall on rare and clinically
dangerous classes (e.g. `vasc` recall of 65%) compared to what an unweighted model
would likely achieve — at some cost to precision. For a medical screening context,
this is a defensible tradeoff: missing a real case (false negative) is costlier than
a false alarm (false positive).

## Project Structure

```
skin-lesion-classifier/
├── src/
│   ├── dataset.py      # Custom Dataset, class weighting, DataLoader setup
│   ├── model.py         # ResNet-18 transfer learning setup
│   ├── train.py         # Multi-epoch training loop with checkpointing
│   ├── evaluate.py       # Per-class precision/recall/F1 evaluation
│   └── gradcam.py        # Grad-CAM explainability implementation
├── app.py                # Gradio web app entry point
├── requirements.txt
└── README.md
```

## Running It

```bash
pip install -r requirements.txt

# Train (adjust paths in train.py to your dataset location)
python src/train.py

# Launch the web app (requires a trained best_model.pth in the working directory)
python app.py
```

## Tech Stack

PyTorch · torchvision · Gradio · OpenCV · scikit-learn · HAM10000 dataset

## Limitations & Future Work

- Currently frozen-backbone transfer learning only; fine-tuning deeper layers
  could further improve rare-class performance.
- Precision on rare classes (e.g. `akiec`, `vasc`) remains low — worth exploring
  F2-score optimization (weighting recall higher than precision) or targeted
  data augmentation for underrepresented classes.
- Not validated for clinical use — a research/portfolio project, not a medical device.
