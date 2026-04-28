# Machinary-Fault-Detection
"A novel Deep Learning framework (MSSAT V2) for Machinery Fault Diagnosis using the MAFault dataset. Features a Multi-Scale Attention Transformer fused with a Gated Spectral Branch to outperform traditional CNN-LSTM hybrids in vibration signal classification.

# MSSAT: Multi-Scale Spectral Attention Transformer for Machinery Fault Diagnosis

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-96.95%25-brightgreen.svg)

## 📌 Project Overview
This repository contains the implementation of **MSSAT V2** (Multi-Scale Spectral Attention Transformer), an advanced architecture designed to identify mechanical faults in industrial machinery. Using the **MAFault Dataset (15GB)**, we demonstrate how combining temporal attention at multiple scales with frequency-domain analysis yields superior results compared to standard hybrid models.

### Target Fault Classes:
* Normal Operation
* Horizontal Misalignment
* Vertical Misalignment
* Imbalance
* Underhang Bearing Fault
* Overhang Bearing Fault

---

## 🏗️ Architecture Design
The MSSAT V2 architecture consists of three primary components:
1. **Multi-Scale Temporal Branches:** Parallel Transformer Encoders processing the signal at window lengths of 128, 256, and 512 to capture both transient and periodic features.
2. **Gated Spectral Branch:** Utilizes **rFFT (Real Fast Fourier Transform)** with a learnable gating mechanism to isolate specific mechanical harmonics from background sensor noise.
3. **Cross-Attention Fusion:** A Multi-head Attention layer that dynamically fuses temporal and spectral features before final classification via **Global Attention Pooling**.

---

## 📊 Performance Comparison
We benchmarked MSSAT V2 against the industry-standard **CNN-LSTM Hybrid** baseline using the same preprocessing and training constraints.

| Metric | MSSAT V2 (Proposed) | CNN + LSTM (Baseline) |
| :--- | :---: | :---: |
| **Max Testing Accuracy** | **96.95%** | 81.01% |
| **Macro F1-Score** | **0.9191** | 0.7078 |
| **Feature Separation** | High (Clear Clusters) | Moderate (Overlap) |

---

## 📈 Visual Results
### t-SNE Feature Embedding
The t-SNE plot demonstrates the high discriminative power of the MSSAT V2. Note the clear isolation of the **Overhang** and **V-Misalign** clusters compared to traditional methods.

*(Insert your saved t-SNE image here: `![t-SNE Results](your_image_path.png)`)*

### Multiclass ROC Curve
MSSAT V2 achieves a significantly higher Area Under the Curve (AUC), proving robustness across different decision thresholds.

---

## 🚀 How to Run
1. **Dataset:** Download the [MAFault Dataset](https://www.kaggle.com/datasets/vuxuancu/mafaulda-full).
2. **Environment:**
   ```bash
   pip install torch torchvision torchaudio pandas numpy scikit-learn matplotlib seaborn
Execution:
Open the provided notebook and update the base_path to point to your local dataset directory.

🎓 Author
David
Student at VIT-AP University
Specializing in AI/ML
