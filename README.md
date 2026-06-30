# Sustainable SRGAN Driven Hybrid Vision Transformer Framework for Moringa Oleifera Leaf Disease Classification

This repository provides the minimal reproducible implementation of a disease-classification pipeline for **Moringa oleifera** leaf images. The pipeline follows the computational design described in the manuscript:

1. **Offline SRGAN enhancement** for improving low-resolution leaf images and preserving fine disease symptoms.
2. **MobileVNet-style lightweight convolutional feature extraction** for local texture, vein, lesion, and color-pattern representation.
3. **Pyramid Vision Transformer classification** for global disease-context modeling through spatial-reduction self-attention.

The implementation is intentionally compact so that reviewers can reproduce the proposed workflow without unnecessary project files.

---

## Repository Structure

```text
moringa-srgan-pvt-mobilevnet/
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
├── src/
│   ├── dataset.py
│   ├── preprocess.py
│   ├── srgan.py
│   ├── mobilevnet_pvt.py
│   ├── train_srgan.py
│   ├── generate_sr_images.py
│   ├── train_classifier.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
└── results/
    └── metrics_summary.csv
```

---

## Dataset Arrangement

Place the Moringa leaf disease dataset outside version control using the following format:

```text
dataset/
├── train/
│   ├── healthy/
│   ├── rust/
│   ├── leaf_spot/
│   └── bacterial_blight/
├── val/
│   ├── healthy/
│   ├── rust/
│   ├── leaf_spot/
│   └── bacterial_blight/
└── test/
    ├── healthy/
    ├── rust/
    ├── leaf_spot/
    └── bacterial_blight/
```

The same class names must be used in `config.yaml`. Additional classes can be added by updating the `classes` list and matching the folder names.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

---

## Step 1: Optional Preprocessing

This step resizes images, applies mild denoising, and performs contrast normalization before SRGAN or classifier training.

```bash
python src/preprocess.py --input dataset --output dataset_preprocessed --config config.yaml
```

After preprocessing, update `config.yaml` paths if `dataset_preprocessed` is used.

---

## Step 2: Train SRGAN for Offline Enhancement

```bash
python src/train_srgan.py --config config.yaml
```

The generator checkpoint is saved to:

```text
checkpoints/srgan/generator_best.pt
```

SRGAN is used as an offline image-enhancement stage. It is not required during real-time classifier inference unless explicit enhancement is requested.

---

## Step 3: Generate Super-Resolved Dataset

```bash
python src/generate_sr_images.py \
  --config config.yaml \
  --checkpoint checkpoints/srgan/generator_best.pt
```

The enhanced images are saved in:

```text
dataset_sr/
├── train/
├── val/
└── test/
```

---

## Step 4: Train MobileVNet-PVT Classifier

Train on the SRGAN-enhanced dataset:

```bash
python src/train_classifier.py --config config.yaml --use-sr
```

Train directly on the original dataset:

```bash
python src/train_classifier.py --config config.yaml
```

The best classifier checkpoint is saved to:

```text
checkpoints/classifier/best_mobilevnet_pvt.pt
```

---

## Step 5: Evaluate the Model

```bash
python src/evaluate.py \
  --config config.yaml \
  --checkpoint checkpoints/classifier/best_mobilevnet_pvt.pt \
  --use-sr
```

The script produces:

```text
results/metrics_summary.csv
results/classification_report.txt
results/confusion_matrix.png
```

Reported metrics include accuracy, precision, recall, F1-score, confusion matrix, parameter count, and average inference time per image.

---

## Step 6: Single Image Prediction

```bash
python src/predict.py \
  --config config.yaml \
  --checkpoint checkpoints/classifier/best_mobilevnet_pvt.pt \
  --image path/to/leaf_image.jpg
```

Output example:

```text
Predicted class: leaf_spot
Confidence: 0.9821
```

---

## Methodological Notes

The proposed implementation separates training-time enhancement and deployment-time classification:

- SRGAN improves image clarity, disease spot visibility, and high-frequency lesion texture during dataset preparation.
- MobileVNet blocks extract compact local features using depthwise separable convolution.
- Pyramid Vision Transformer blocks capture global disease-context relations using spatial-reduction attention.
- The classifier alone is used during standard inference, reducing runtime complexity.

---

## Code Availability

The repository contains the source code required to reproduce the computational workflow. For editorial deposition, archive the final GitHub release in a DOI-assigning repository such as Zenodo and add the resulting repository DOI in the manuscript Code Availability section.
