# Cognitive Flow Prediction using Physiological Signals

A deep learning project exploring whether wearable physiological signals can predict cognitive task performance.

The system uses time-series data from physiological sensors and trains temporal models to classify whether a user solves a cognitive puzzle correctly.


## Overview

Physiological signals can reflect changes in cognitive state, attention, and workload.  
This project models these temporal patterns using recurrent and attention-based neural networks.

Input signals include:

- Accelerometer data (ACCx, ACCy, ACCz)
- Movement
- Electrodermal Activity (EDA)
- Temperature
- Heart Rate (HR)
- Demographic features

Target:

- Puzzle correctness prediction


## Models Implemented

- **LSTM**  
  Captures long-term temporal dependencies in physiological signals.

- **GRU**  
  A lightweight recurrent model using gating mechanisms with fewer parameters.

- **CNN-LSTM**  
  Uses Conv1D layers for local physiological patterns and LSTM for temporal trends.

- **Attention LSTM**  
  Learns which parts of a physiological sequence are most important for prediction.


## Project Structure

```
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── results/
│   ├── metrics.csv
│   └── confusion matrices
│
├── src/
│   ├── models/
│   ├── preprocessing.py
│   ├── train_all.py
│   └── evaluate_all.py
│
└── requirements.txt
```


## Installation

```bash
pip install -r requirements.txt
```


## Usage

### Preprocess Dataset

```bash
cd src
python preprocessing.py
```


### Train Models

```bash
python train_all.py
```


### Evaluate Models

```bash
python evaluate_all.py
```


## Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix


## Future Improvements

- Leave-One-Subject-Out (LOSO) validation
- Transformer-based temporal models
- Sensor-level attention mechanisms
- Deployment on wearable/edge devices


## Objective

To develop an interpretable AI system capable of understanding cognitive performance from physiological signals.