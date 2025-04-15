# MLP Mixer Implementation

This repository provides a PyTorch implementation of the **MLP-Mixer** architecture, as proposed in the paper [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601). Unlike traditional convolutional or transformer-based models, the MLP-Mixer leverages only multi-layer perceptrons (MLPs) for image classification tasks—achieving competitive results.

---

## 📁 Datasets

This implementation has been tested on the following datasets:

- CINIC-10
- Food-101
- CIFAR-10
- CIFAR-100
- SVHN
- STL-10

Each dataset has a dedicated Jupyter notebook for training and evaluation:

- `mlp_cinic10.ipynb`
- `mlp_food101.ipynb`
- `mlp_mixer_stl10.ipynb`
- `mlp_cifar10.ipynb`
- `mlp_cifar100.ipynb`
- `mlp_svhn.ipynb`

---

## 🧠 Model Overview

The **MLP-Mixer** model processes an input image by:

1. Splitting it into non-overlapping patches.
2. Linearly embedding each patch.
3. Applying a series of MLP-based "mixer" blocks that alternate between:
   - **Token-Mixing MLPs** (operate across spatial positions).
   - **Channel-Mixing MLPs** (operate across feature channels).

This pure-MLP approach removes the need for convolutional or self-attention layers.

---

## 🚀 Getting Started

### ✅ Prerequisites

Make sure you have the following packages installed:

- Python ≥ 3.6
- PyTorch
- Torchvision
- NumPy
- Matplotlib

Install using pip:

```bash
pip install torch torchvision numpy matplotlib

