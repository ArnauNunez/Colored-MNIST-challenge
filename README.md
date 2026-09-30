# Colored MNIST Challenge: Robust CNN & Generalization Analysis

## Overview
This repository contains an end-to-end deep learning project for the **Colored MNIST Challenge**. It implements a custom Convolutional Neural Network (CNN) in **PyTorch** designed to classify hand-written digits under varying levels of environmental noise and data corruption.

---

## Architecture & Technical Stack
* **Framework:** PyTorch, torchvision, Torchinfo
* **Model Architecture:** Custom Deep CNN featuring:
  * 7 Convolutional blocks with alternating strides and filter sizes ($32 \to 64 \to 128$ channels).
  * **Batch Normalization** after convolutional layers to stabilize internal covariate shift.
  * **Dropout ($p = 0.4$)** integrated for active regularization and prevention of overfitting.
* **Preprocessing Pipeline:** Automated ingestion, resizing, grayscale conversion, contrast/brightness adjustments, and strict threshold binarization (`> 0.1`).
* **Security & Best Practices:** Strict usage of `.state_dict()` serialization (`weights_only=True`) to prevent unpickling vulnerabilities.

---

## Critical Insights & Generalization Gap Analysis
While the model achieves exceptional performance (>99%) on clean or baseline data (`Easy` set), performance drops significantly on the `Hard` dataset. This case study demonstrates critical ML competencies:
1. **Feature Loss via Hard Binarization:** The strict thresholding effective on clean images strips away structural details of low-intensity digits embedded in complex backgrounds.
2. **Lack of Spatial Invariance:** The model was trained on rigidly aligned objects, making its feature maps highly sensitive to geometric shifts, affine transformations, or rotations present in adversarial sets.

### Proposed Improvements (Next Steps)
* **On-the-fly Data Augmentation:** Implement `transforms.RandomRotation` and `transforms.RandomAffine` inside the training pipeline to enforce spatial invariance.
* **Dynamic Normalization:** Remove rigid binarization with adaptive histogram equalization (CLAHE) or standard tensor normalization to preserve weak structural patterns.

---

## Repository Structure
```text
├── Colored_MNIST_Challenge.ipynb  # Complete Jupyter Notebook (Code + Markdown + Outputs)
├── model_weights.pth              # Serialized model weights (state_dict)
├── predictions.npy                # Exported test predictions for evaluation tracking
└── README.md                      # Project documentation
```

---

### 1. Requirements & Dependencies
Ensure Python 3.9+ is installed. Run the following command to install required libraries:
```bash
pip install torch torchvision torchinfo matplotlib scikit-learn tqdm gdown
