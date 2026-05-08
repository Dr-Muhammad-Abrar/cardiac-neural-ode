# Hybrid Neural ODE for Multi-Modal Cardiac Classification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PhysioNet](https://img.shields.io/badge/Data-PhysioNet-green.svg)](https://physionet.org/)

> **Hybrid Neural ODE Architecture for Multi-Modal Cardiac Signal Classification and Clinical Risk Scoring Using Real PhysioNet Databases**
>
> Muhammad Faisal Abrar et.al

---

## Results at a Glance

| Metric | Value |
|---|---|
| Weighted F1-Score | **0.850** |
| Macro F1-Score | **0.848** |
| Macro ROC-AUC | **0.970** |
| MI / LBBB F1 | 0.922 |
| Cardiomyopathy / RBBB F1 | **0.957** |
| Ventricular / Dangerous F1 | 0.868 |
| Training epochs | 50 |
| Total parameters | 276,005 |

---

## Overview

This project presents a seven-component hybrid framework combining:

- **Van der Pol ODE** — physics prior for ECG waveform dynamics
- **FitzHugh-Nagumo ODE** — heart rate variability prior
- **Three-element Windkessel ODE** — arterial blood pressure prior
- **Latent Neural ODE** (torchdiffeq) — learns residual cardiac dynamics
- **BiLSTM Classifier** — bidirectional arrhythmia classification
- **CasADi / IPOPT** — patient-specific Windkessel parameter fitting
- **Grad-CAM + SHAP** — explainability attribution maps
- **Clinical Risk Scoring** — stroke, heart failure, arrhythmia risk %

---

## Pipeline

```
8 PhysioNet Databases (ECG + ABP + SpO2)
        |
Signal Preprocessing (bandpass filter > segment > normalise > stack [256x3])
        |
Physics ODE Priors (Van der Pol | FitzHugh-Nagumo | Windkessel)
        |
Latent Neural ODE Encoder (BiGRU > reparameterise z0 > ODE trajectory)
        |
BiLSTM Arrhythmia Classifier (5 classes)
        |                           |
CasADi/IPOPT                  Grad-CAM + SHAP
Windkessel fitting             Explainability maps
R, C, HR per patient
        |
Clinical Risk Scoring
Stroke % | Heart Failure % | Arrhythmia %
```

---

## Datasets (8 PhysioNet Databases)

> Data not included — download from physionet.org (free registration)

| Database | Signals | Beats/Segments | Classes |
|---|---|---|---|
| MIT-BIH Arrhythmia | ECG 360Hz | 109,804 | N, LBBB, RBBB, A, V |
| MIT-BIH Malignant Ventricular | ECG 250Hz | 42 | VT, VF, VFL |
| MIT-BIH Normal Sinus Rhythm | ECG 128Hz | 3,600 | Normal |
| MIT-BIH Supraventricular | ECG 128Hz | 184,224 | SVA, N, RBBB, LBBB |
| PTB Diagnostic ECG | 15-lead 1000Hz | 549 | MI, CM, BBB, Dysrhythmia |
| PTB-XL | 12-lead 500Hz | 5,000 | NORM, MI, RBBB, AFIB |
| BIDMC PPG+Respiration | ECG+ABP+SpO2 125Hz | 265 | Stable ICU |
| PhysioNet Challenge 2015 | ECG+ABP+SpO2 | 500 | ICU alarms |

**Total: 303,784 annotated signals across 3 modalities**

---

## Classification Report (Test set, n=1,500)

```
                          precision  recall  f1-score  support
         Normal / Stable     0.738   0.765     0.751      306
               MI / LBBB     0.894   0.952     0.922      310
   Cardiomyopathy / RBBB     0.979   0.935     0.957      306
Atrial / Supraventricular     0.755   0.734     0.745      282
  Ventricular / Dangerous     0.882   0.855     0.868      296

               accuracy                         0.850     1500
              macro avg     0.850   0.848     0.848     1500
           weighted avg     0.851   0.850     0.850     1500

Macro ROC-AUC: 0.9700
```

---

## Project Structure

```
cardiac-neural-ode/
|
|-- main.py               <- run this
|-- config.py             <- all paths and hyperparameters
|-- requirements.txt
|
|-- src/
|   |-- data_loader.py    <- all 8 PhysioNet dataset loaders
|   |-- dataset.py        <- unified CardiacDataset + DataLoaders
|   |-- model.py          <- Neural ODE + BiLSTM architecture
|   |-- train.py          <- training loop + evaluation
|   |-- ode_fitting.py    <- CasADi/IPOPT Windkessel fitting
|   |-- risk_scorer.py    <- clinical risk scoring
|   `-- explainability.py <- Grad-CAM + SHAP + ODE trajectories
|
|-- results/              <- saved plots (7 PNG files)
|-- data/physionet_cache/ <- place downloaded datasets here
`-- checkpoints/          <- saved model weights
```

---

## Setup and Usage

```bash
# Install
pip install -r requirements.txt

# Verify dataset paths
python main.py --verify-only

# Quick test (2 epochs, CPU)
python main.py --fast --epochs 2

# Full training (50 epochs, GPU recommended)
python main.py --epochs 50

# Load saved model and evaluate only
python main.py --no-train

# Skip explainability for faster run
python main.py --no-xai
```

---

## Dataset Folder Names

Place in `data/physionet_cache/` with exact names:

```
mit-bih-arrhythmia-database-1.0.0/
mit-bih-malignant-ventricular-ectopy-database-1.0.0/
mit-bih-normal-sinus-rhythm-database-1.0.0/
mit-bih-supraventricular-arrhythmia-database-1.0.0/
bidmc-ppg-and-respiration-dataset-1.0.0/
reducing-false-arrhythmia-alarms-in-the-icu-the-physionetcomputing-in-cardiology-challenge-2015-1.0.0/
ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
ptb-diagnostic-ecg-database-1.0.0/

## Citation

```bibtex
@article{abrar2025cardiac,
  title   = {Hybrid Neural ODE Architecture for Multi-Modal Cardiac Signal
             Classification and Clinical Risk Scoring Using Real PhysioNet Databases},
  author  = {Abrar, Muhammad Faisal et.al},
  journal = {MDPI Mathematics},
  year    = {2026},
  note    = {Submitted to Special Issue: New Trends in Advanced Statistical
             Techniques and AI, A Multidisciplinary Approach}
}
```
I have not Uploaded the dataset to data folder. the links are provided in the file and it is publicaly available datasets. 
---

## Authors



Dr. Muhammad Faisal Abrar  University of Ha'il, Saudi Arabia 

\* Corresponding authors: m.abrar@uoh.edu.sa

---

## License

MIT License. PhysioNet datasets are subject to their respective data use agreements.

## Acknowledgements

The authors gratefully acknowledge the PhysioNet resource and the creators of all eight databases for making their data freely available to the research community.
