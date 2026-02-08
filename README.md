# Fair and Explainable Deep Learning for Skin Lesion Classification

**Author:** Mohammadreza Golkar Khouzani  
**Affiliation:** M.Sc. student in Artificial Intelligence, Faculty of Computer Engineering, Islamic Azad University, Khomeinishahr, Iran  
**Date:** February 2026  

---

## Overview
This repository contains the code, models, and data preprocessing scripts for the research paper:

**"Fair and Explainable Deep Learning for Skin Lesion Classification Across Fitzpatrick Skin Types"**

The study develops an **EfficientNet-B0 based deep learning pipeline** for multi-class skin lesion classification, emphasizing **fairness** across different skin tones and **explainability** using Grad-CAM. The pipeline is evaluated on the **HAM10000** and **Diverse Dermatology Images (DDI)** datasets.

---

## Key Features

- Multi-class skin lesion classification (7 classes) with patient-level splitting
- Class imbalance handling using weighted sampling
- Transfer learning and fine-tuning of EfficientNet-B0
- Explainability via Grad-CAM visualizations
- Fairness-aware evaluation across Fitzpatrick skin types
- Trust score metric combining accuracy, equalized odds, and Grad-CAM localization quality

---

## Datasets

1. **HAM10000**: Dermoscopic images of 10,015 lesions with seven diagnostic classes, enriched with Fitzpatrick scale annotations via Fitzpatrick17k.  
2. **Diverse Dermatology Images (DDI)**: Clinical images (656) used for fairness-oriented debiasing experiments.

---

## Requirements

- Python 3.9+
- PyTorch
- NumPy, Pandas, Matplotlib, Seaborn
- CUDA-enabled GPU recommended for training
- Optional: Jupyter or Google Colab for interactive visualization

---

## Usage

1. Clone the repository:
```bash
git clone https://github.com/<username>/<repo>.git
Prepare datasets:

Place HAM10000 and DDI datasets in the /data folder

Ensure metadata CSV files are correctly mapped

Train the model:

bash
Copy code
python train_ham10000.py
python train_ddi_debias.py
Generate Grad-CAM visualizations:

bash
Copy code
python generate_gradcam.py --dataset ham10000 --output gradcam/
Evaluate fairness and trust score:

bash
Copy code
python evaluate_fairness.py --dataset ddi
Results
HAM10000 Test Accuracy: 0.41

Macro F1-score: 0.33

Macro AUC: 0.86

Trust score on DDI: ~0.30

Grad-CAM visualizations show that the model attends primarily to lesion regions, supporting interpretability.

Future Work
Integration of additional XAI techniques (e.g., SHAP)

Domain adaptation for clinical vs dermoscopic images

Expansion of training data for darker skin tones

Refinement of trust score with dermatologist feedback

PDF
The full paper is available as PDF: Download PDF

Citation
If you use this repository in your research, please cite:

graphql
Copy code
@article{golkar2026fair,
  title={Fair and Explainable Deep Learning for Skin Lesion Classification Across Fitzpatrick Skin Types},
  author={Golkar Khouzani, Mohammadreza},
  year={2026},
  note={M.Sc. thesis, Islamic Azad University, Khomeinishahr, Iran}
}
