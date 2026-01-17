# Vision Transformer for CMS HCAL Run Classification (ML4DQM)

This project implements a Vision Transformer (ViT)–based classifier to
distinguish between two CMS HCAL detector runs using DigiOccupancy maps.
The work was completed as part of the ML4Sci GSoC 2025 evaluation task
for ML-based Data Quality Monitoring (ML4DQM).

---

## Problem Overview

The CMS Hadronic Calorimeter (HCAL) records hit multiplicities across
detector cells (ieta × iphi) for each lumi-section (LS).
Given two synthetic HCAL datasets corresponding to different runs,
the goal is to classify each LS image by its originating run.

---

## Dataset

- Shape: `(LS, 64, 72)`
- Each LS represents a detector occupancy snapshot
- Highly sparse with strong geometric structure

Datasets are provided by ML4Sci and are not included in this repository.

---

## Methodology

- Each lumi-section is treated as a 2D image
- Images are normalized and resized to 224×224
- Single-channel data is expanded to 3 channels
- Model: Vision Transformer (ViT-Tiny, patch size 16)
- Loss: Cross-entropy
- Optimizer: Adam

---

## Robustness Check (Bonus)

A temporal robustness test was performed:
- Training on early lumi-sections (LS 0–799)
- Testing on later lumi-sections (LS 800–999)

This evaluates time stability and generalization across detector operation.

---

## Results

| Metric | Value |
|------|------|
Accuracy | ~1.00 |
AUC | ~1.00 |

The model demonstrates strong separability and temporal robustness.

---

## Usage

```bash
# Train
python src/train.py

# Evaluate
python src/eval.py
```

## Tools
- Tools & Libraries
- PyTorch
- timm
- NumPy
- scikit-learn
- matplotlib
