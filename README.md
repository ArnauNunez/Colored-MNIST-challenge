# Colored MNIST Project

This repository contains a PyTorch pipeline for loading, preprocessing, and training models on a **Colored MNIST dataset**.

## Features
- Custom `Dataset` class for Colored MNIST
- Preprocessing: resize, grayscale, contrast, brightness, binarization
- Training/validation split with `DataLoader`
- Example ML workflow with digit recognition

## Requirements
- Python 3.9+
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- tqdm

Install dependencies:
```bash
pip install torch torchvision torchinfo matplotlib scikit-learn tqdm
