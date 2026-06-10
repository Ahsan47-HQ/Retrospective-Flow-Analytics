# Retrospective Flow Analysis (RFA) for Human Physiology

## Overview
This project explores whether physiological signals from a wrist-worn 
sensor can predict cognitive outcomes during problem-solving tasks. 
Specifically, we predict whether a participant successfully solved a 
puzzle using electrodermal activity (EDA), skin temperature, heart rate, 
and wrist movement — collected via the Empatica E4 wearable device.

## Research Question
Can physiological signals alone predict puzzle-solving success (Correct), 
and does adding self-reported focus level improve this prediction?

## Dataset
- 8 participants, 5 puzzles each (40 total puzzle attempts)
- 5,094 second-level physiological measurements at 1Hz
- Features: EDAz, TEMPz, HRz, Movement, DifficultyLevel, FocusLevel
- Target: Correct (binary — solved or not solved)
- Data not included in repository due to privacy restrictions

## Methodology
- Leave-One-Person-Out (LOPO) cross-validation
- 5 models: Majority Baseline, Logistic Regression, Decision Tree, 
  KNN, Random Forest
- Three feature conditions compared:
  - Physiological signals only
  - Physiological signals + DifficultyLevel
  - Physiological signals + DifficultyLevel + FocusLevel

## Key Findings
- Physiological signals + difficulty achieves AUC ~0.647 for predicting 
  correctness, meaningfully above chance (0.5)
- Adding self-reported FocusLevel hurts tree and distance-based models 
  (Random Forest AUC drops from 0.640 to 0.535)
- Skin temperature (TEMPz) is the most important physiological feature, 
  nearly matching DifficultyLevel in importance
- Cross-person generalization is the main limiting factor — consistent 
  with prior EduFit research

## Project Structure
RFA/
├── notebooks/
│   └── explorations.ipynb   # Main analysis notebook
├── data/                    # Not pushed — private research data
├── results/                 # Output plots and tables
├── src/                     # Source scripts
├── requirements.txt
└── .gitignore

## References
- Romine et al. (2020) — EduFit: Cognitive Load Tracker, Sensors
- Agarwal et al. (2021) — Sensor-Based Prediction of Mental Effort, Signals
- Romine et al. (2022) — Mental Effort Measurement Using EDA, Sensors

