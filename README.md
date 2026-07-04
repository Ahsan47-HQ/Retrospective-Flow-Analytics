# Cognitive Flow Project

Predicting puzzle correctness from behavioral and physiological time-series data, using deep learning evaluated with Leave-One-Subject-Out (LOSO) cross-validation.

## Overview

This project trains and compares four sequence models, LSTM, GRU, CNN-LSTM, and Attention-LSTM, to predict the binary target `Correct` from sliding windows of per-timestep features. Because the data comes from multiple participants, evaluation uses LOSO cross-validation: each participant is held out as a test set in turn, so reported metrics reflect how well a model generalizes to a person it has never seen during training.

## Project Structure

```
Cognitive_Flow_Project/
├── data/
│   └── raw/
│       └── AllCombined_MentalEffortFocus.csv
├── results/
│   ├── loso_metrics.csv
│   ├── loso_summary.csv
│   ├── summary.md
│   ├── summary.tex
│   ├── accuracy_boxplot.png
│   ├── precision_boxplot.png
│   ├── recall_boxplot.png
│   ├── f1_boxplot.png
│   └── roc_auc_boxplot.png
├── saved_models/
│   └── {model}_{participant}.keras
├── notebooks/
│   ├── data_exploration.ipynb
│   └── processed_data_visualization.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_all.py
│   ├── evaluate_all.py
│   └── models/
│       ├── lstm.py
│       ├── gru.py
│       ├── cnn_lstm.py
│       └── attention_lstm.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

Markdown export in `evaluate_all.py` requires the `tabulate` package:

```bash
pip install tabulate
```

3. Place the raw dataset at `data/raw/AllCombined_MentalEffortFocus.csv`.

## Pipeline

### 1. Preprocessing

`src/preprocessing.py` loads the raw CSV, drops missing rows, label encodes categorical columns other than `Person`, and builds sliding-window sequences grouped by participant (window size 30, stride 5). It returns:

- `X`, the windowed feature sequences
- `y`, the binary `Correct` labels
- `person_ids`, the participant associated with each window
- `feature_cols`, the feature column names used
- `encoders`, the fitted label encoders

`FocusLevel`, `Focus01`, and `Puzzle` are excluded from the features to prevent label leakage. There is no train/test split and no scaling done at this stage. Both happen inside each LOSO fold to prevent data leakage.

### 2. Training

Run from the `src/` directory:

```bash
python train_all.py
```

For every architecture and every held-out participant, the script:

- Splits the data by participant into train and test sets
- Splits a validation set from the training windows only, stratified on the label
- Fits a StandardScaler on the training fold only, then applies it to train, validation, and test
- Computes class weights on the training fold to handle class imbalance
- Trains with early stopping and learning rate reduction, both monitoring validation ROC-AUC
- Saves a checkpoint for each fold, reloads it, and evaluates on the held-out participant
- Records accuracy, precision, recall, F1, and ROC-AUC for that fold

All per-fold metrics are saved to `results/loso_metrics.csv`, and a mean plus or minus standard deviation summary is printed at the end. This script only trains and records metrics. It does not produce plots, markdown, or LaTeX output.

### 3. Evaluation summary

```bash
python evaluate_all.py
```

This script performs no training and loads no models. It reads `results/loso_metrics.csv` and produces:

- `results/loso_summary.csv`, mean and standard deviation per model, rounded to 4 decimals and sorted alphabetically
- `results/summary.md`, the same summary as a markdown table
- `results/summary.tex`, the same summary as a LaTeX table
- Boxplots of accuracy, precision, recall, F1, and ROC-AUC across folds, one PNG per metric, saved to `results/`

## Models

| Model | Architecture |
|---|---|
| LSTM | Two stacked LSTM layers with dropout and a dense head |
| GRU | Two stacked GRU layers with dropout and a dense head |
| CNN-LSTM | 1D convolution and pooling, followed by two stacked LSTM layers with dropout |
| Attention-LSTM | Two stacked LSTM layers, self-attention, global average pooling, and a dense head |

## Metrics

Metrics are reported per fold and summarized as mean plus or minus standard deviation across all held-out participants:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

All reported metrics are computed on each fold's held-out test participant, never on validation data. Validation metrics are used only for early stopping, learning rate reduction, and checkpoint selection during training.

## Current Results

Latest LOSO summary across 8 participants:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| LSTM | 0.669 ± 0.203 | 0.590 ± 0.373 | 0.539 ± 0.408 | 0.512 ± 0.366 | 0.790 ± 0.301 |
| GRU | 0.559 ± 0.254 | 0.609 ± 0.430 | 0.466 ± 0.365 | 0.439 ± 0.306 | 0.714 ± 0.343 |
| CNN-LSTM | 0.530 ± 0.224 | 0.491 ± 0.395 | 0.474 ± 0.364 | 0.438 ± 0.322 | 0.711 ± 0.266 |
| Attention-LSTM | 0.555 ± 0.241 | 0.523 ± 0.338 | 0.619 ± 0.365 | 0.513 ± 0.296 | 0.650 ± 0.235 |

LSTM currently performs best on average, but standard deviations are large across all models. This is driven mostly by a small number of participants, in particular one whose test fold is predicted as a single class by every model, which should be investigated separately from overall model comparison.

## Notes

- The dataset is not committed to version control.
- Model checkpoints are saved per fold as `{model}_{participant}.keras`, so results can be traced back to a specific held-out participant.
- Random seed for validation splitting is fixed at 42 for reproducibility.
- Class weights are computed per fold from the training split only.