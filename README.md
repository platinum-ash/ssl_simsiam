# 🛰️ SimSiam for Satellite Imagery - Self-Supervised Learning Implementation

This repository contains a PyTorch implementation of **SimSiam** (https://arxiv.org/abs/2011.10566), self-supervised learning (SSL) method, applied to satellite imagery. The project includes:

- 🧠 Pretraining SimSiam on the **SSL4EO** dataset (RGB subset)
- 🔬 Linear evaluation on **CIFAR-10**, **CIFAR-100**, and **EuroSat**
- 📈 Metrics logging via TensorBoard
- 📓 Jupyter notebooks for training, evaluation, and visual analysis

The goal is to explore the effectiveness of SSL—particularly SimSiam—for representation learning in remote sensing, and assess how pretrained features transfer across domains.

---

## 📁 Project Structure

├── dataset.py # Data loading logic
├── model.py # SimSiam model architecture (encoder, projector, predictor)
├── notebooks/ # Jupyter notebooks for training and evaluation
│ ├── evaluate_cifar100.ipynb # Linear evaluation on CIFAR-100
│ ├── evaluate_cifar10.ipynb # Linear evaluation on CIFAR-10
│ ├── evaluate_eurosat.ipynb # Linear evaluation on EuroSat
│ ├── randominit_vs_sslencoder.ipynb # Comparison: pretrained vs randomly initialized model
│ └── train.ipynb # Main training notebook for SimSiam
└── utils.py # Utility functions for loading and creating classifiers
