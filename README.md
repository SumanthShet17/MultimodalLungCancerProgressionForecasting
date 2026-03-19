#  Lung Cancer Detection & Progression Scoring — Multi-Task Project

A **multi-task learning** project that combines **deep learning** (image-based) and **classical machine learning** (clinical-data-based) approaches for lung cancer detection, classification, and progression scoring, with a **late-fusion** strategy for final prediction.

---

##  Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Models](#models)
  - [1. Image Model — Multi-Task EfficientNet-B2](#1-image-model--multi-task-efficientnet-b2)
  - [2. Clinical Model — Logistic Regression](#2-clinical-model--logistic-regression)
  - [3. Late Fusion](#3-late-fusion)
- [Results](#results)
- [Setup & Usage](#setup--usage)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)

---

## Overview

Lung cancer is one of the leading causes of cancer-related mortality worldwide. Early and accurate detection is critical. This project tackles the problem using two complementary pipelines:

| Pipeline | Input | Task |
|---|---|---|
| **Deep Learning (Image)** | Chest CT scan images | Cancer type classification + progression score regression |
| **Machine Learning (Clinical)** | Tabular clinical/symptom data | Binary cancer presence prediction |
| **Late Fusion** | Combined outputs | Unified cancer risk score |

---

## Project Architecture

```
┌───────────────────────────────────────────────────────────┐
│                     Input Data                            │
│  ┌─────────────────┐        ┌──────────────────────────┐  │
│  │  CT Scan Images  │        │  Clinical / Symptom Data │  │
│  └────────┬────────┘        └────────────┬─────────────┘  │
│           │                              │                │
│           ▼                              ▼                │
│  ┌─────────────────┐        ┌──────────────────────────┐  │
│  │ EfficientNet-B2  │        │  Logistic Regression     │  │
│  │ (Multi-Task)     │        │  (scikit-learn)          │  │
│  │  • Classification│        │  • Binary classification │  │
│  │  • Regression    │        └────────────┬─────────────┘  │
│  └────────┬────────┘                     │                │
│           │                              │                │
│           ▼                              ▼                │
│  ┌──────────────────────────────────────────────────────┐ │
│  │                  Late Fusion                         │ │
│  │  Fused Score = α × P_clinical + (1-α) × P_image     │ │
│  └──────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

---

## Dataset

### Image Data
- **Source**: CT scan images of the chest (from a Kaggle archive).
- **Classes** (4):
  | Class | Train Samples |
  |---|---|
  | Adenocarcinoma | 195 |
  | Large Cell Carcinoma | 115 |
  | Normal | 148 |
  | Squamous Cell Carcinoma | 155 |
- Images are grayscale, resized to **456 × 456** pixels.

### Clinical Data
- A tabular CSV (`lungcanc.csv`) containing symptoms and demographic features:
  - `GENDER`, `AGE`, `SMOKING`, `YELLOW_FINGERS`, `ANXIETY`, `PEER_PRESSURE`, `CHRONIC DISEASE`, `FATIGUE`, `ALLERGY`, `WHEEZING`, `ALCOHOL CONSUMING`, `COUGHING`, `SHORTNESS OF BREATH`, `SWALLOWING DIFFICULTY`, `CHEST PAIN`.
- **Target**: `LUNG_CANCER` (YES / NO).

---

## Models

### 1. Image Model — Multi-Task EfficientNet-B2

A **transfer-learning** based multi-task model built on top of `tf_efficientnet_b2_ns` (from the `timm` library).

**Architecture:**
- **Backbone**: EfficientNet-B2 (pretrained, feature extractor)
- **Classification Head**: `Linear → ReLU → Dropout(0.3) → Linear(num_classes)`
- **Regression Head**: `Linear → ReLU → Dropout(0.3) → Linear(1) → Sigmoid`

**Training Details:**
| Hyperparameter | Value |
|---|---|
| Image Size | 456 × 456 |
| Batch Size | 8 |
| Optimizer | AdamW (lr=3e-4, wd=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=12) |
| Epochs | 15 |
| Classification Loss | CrossEntropyLoss |
| Regression Loss | MSELoss |
| Regression Weight (λ) | 0.4 |

**Data Augmentation (Train):**
- Grayscale → 3 channels
- Random Resized Crop
- Random Horizontal Flip
- Random Rotation (±10°)
- ImageNet Normalization

**Progression Score Mapping:**
| Class | Score |
|---|---|
| Normal | 0.0 |
| Adenocarcinoma | 0.5 |
| Squamous Cell Carcinoma | 0.75 |
| Large Cell Carcinoma | 1.0 |

### 2. Clinical Model — Logistic Regression

A standard **Logistic Regression** classifier (L2 penalty, LBFGS solver) trained on the clinical dataset.

- Features are **standardized** using `StandardScaler`.
- Train/Test split: **75/25** (stratified).

### 3. Late Fusion

Combines outputs from both models using a weighted average:

```
Fused Score = α × P_clinical + (1 − α) × P_image
```

Where `α = 0.6` by default.

**Stage Interpretation:**
| Fused Score | Interpretation |
|---|---|
| < 0.3 | Healthy |
| 0.3 – 0.6 | Early Stage |
| > 0.6 | Advanced Stage |

---

## Results

### Image Model (EfficientNet-B2)
| Metric | Best Value |
|---|---|
| **Test Accuracy** | **94.29%** |
| **R² Score** | **0.9170** |
| Train Accuracy (final epoch) | 99.67% |

### Clinical Model (Logistic Regression)
| Metric | Value |
|---|---|
| **Accuracy** | **87.16%** |

---

## Setup & Usage

### Prerequisites

- Python 3.8+
- Google Colab (recommended, with T4 GPU) or a local machine with CUDA support.

### Installation

```bash
pip install timm torch torchvision seaborn scikit-learn tqdm matplotlib pandas pillow
```

### Running the Notebook

1. Upload `MLDL_Project.ipynb` to Google Colab (or open locally in Jupyter).
2. Upload the image dataset archive and the `lungcanc.csv` clinical data file.
3. Run all cells sequentially.

The notebook will:
  - Extract and preprocess the CT scan dataset.
  - Train the multi-task EfficientNet-B2 model.
  - Plot training curves (Loss, Accuracy, R²).
  - Train the Logistic Regression model on clinical data.
  - Perform late fusion inference on a random test sample.

---

## Technologies Used

| Category | Tools / Libraries |
|---|---|
| Deep Learning | PyTorch, timm (EfficientNet-B2) |
| Machine Learning | scikit-learn (Logistic Regression) |
| Data Handling | Pandas, NumPy, PIL |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab (T4 GPU) |
| Language | Python 3 |

---

## Project Structure

```
.
├── MLDL_Project.ipynb   # Main notebook with all code and experiments
├── lungcanc.csv         # Clinical symptom dataset
├── README.md            # This file
└── Data/                # CT scan image dataset (after extraction)
    ├── train/
    │   ├── adenocarcinoma/
    │   ├── large.cell.carcinoma/
    │   ├── normal/
    │   └── squamous.cell.carcinoma/
    └── test/
        ├── adenocarcinoma/
        ├── large.cell.carcinoma/
        ├── normal/
        └── squamous.cell.carcinoma/
```

---

## 📄 License

This project is for academic/educational purposes.


