# Cognitive Flow Project

Predicting mental effort and focus from behavioral data using deep learning, evaluated with Leave-One-Subject-Out (LOSO) cross-validation.

## Overview

This project trains and compares four sequence models (LSTM, GRU, CNN-LSTM, Attention-LSTM) to predict correctness/performance from windowed time-series features. Because the data comes from multiple participants, evaluation uses LOSO cross-validation: each participant is held out as a test set in turn, so reported metrics reflect how well a model generalizes to a person it has never seen.

## Project Structure

```
Cognitive_Flow_Project/
├── data/
│   └── raw/
│       └── AllCombined_MentalEffortFocus.csv
├── results/
│   ├── loso_metrics.csv
│   ├── loso_summary.csv
│   ├── accuracy_boxplot.png
│   ├── f1_boxplot.png
│   └── roc_auc_boxplot.png
├── saved_models/
│   ├── lstm_best.keras
│   ├── gru_best.keras
│   ├── cnn_lstm_best.keras
│   └── attention_best.keras
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

3. Place the raw dataset at `data/raw/AllCombined_MentalEffortFocus.csv`.

## Pipeline

### 1. Preprocessing

`src/preprocessing.py` loads the raw CSV, cleans it, encodes categorical columns, and builds sliding-window sequences per participant. It returns:

- `X`, the windowed feature sequences
- `y`, the labels
- `person_ids`, the participant associated with each sequence
- `feature_cols`, the list of feature column names
- `encoders`, the fitted label encoders

There is no train/test split and no scaling done at this stage. Both happen inside each LOSO fold to prevent data leakage.

### 2. Training

Run from the `src/` directory:

```bash
python train_all.py
```

This trains all four models using LOSO cross-validation. For each architecture and each held-out participant, the script:

- Splits the data by participant into train and test sets
- Splits a validation set from the training participants only
- Fits a StandardScaler on the training fold only, then applies it to train, validation, and test
- Computes class weights to handle class imbalance
- Trains with early stopping and checkpointing
- Evaluates on the held-out participant and records accuracy, precision, recall, F1, and ROC-AUC

Only the best checkpoint per architecture, based on validation ROC-AUC, is kept in `saved_models/`. Metrics for every fold are still saved in `results/loso_metrics.csv`.

### 3. Evaluation summary

```bash
python evaluate_all.py
```

This reads `results/loso_metrics.csv`, aggregates mean and standard deviation per model, saves `results/loso_summary.csv`, prints a formatted summary, and generates boxplots comparing models across folds.

## Models

| Model | Architecture |
|---|---|
| LSTM | Single LSTM layer with dropout and a dense head |
| GRU | Single GRU layer with dropout and a dense head |
| CNN-LSTM | 1D convolution and pooling followed by an LSTM layer |
| Attention-LSTM | LSTM with a self-attention layer and global average pooling |

## Metrics

Results are reported as mean plus or minus standard deviation across all LOSO folds:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

All reported metrics are computed on each fold's held-out test participant, never on validation data.

## Notes

- The dataset is not committed to version control.
- Only the best model checkpoint per architecture is kept to avoid saving one file per fold per model.
- Random seed for validation splitting is fixed at 42 for reproducibility.