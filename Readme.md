## Project Structure

```
rfa/
├── configs/
│   └── config.yaml      # Hyperparameters and data paths
├── src/
│   ├── data_loader.py   # Data preprocessing, sliding windows, LOSO splits
│   ├── model.py         # DeepFlow 1D CNN with metadata fusion
│   └── trainer.py       # Training loop and WandB integration
├── main.py              # Entry point to run LOSO cross-validation
├── requirements.txt     # Python dependencies
└── Readme.md            # You are here
```

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure WandB**:
   Make sure you are logged into Weights & Biases:
   ```bash
   wandb login
   ```

3. **Start Training**:
   Run the Leave-One-Subject-Out (LOSO) cross-validation:
   ```bash
   python main.py --config configs/config.yaml
   ```

## Model Architecture

The implementation follows the **DeepFlow** design:
- **CNN Backbone**: 4 blocks of Conv1D (32 filters) + BatchNorm + MaxPool1D + Dropout (0.1).
- **Fusion Head**: Concatenates CNN features from sensors with contextual metadata (Difficulty, Age, Gender).
- **Output**: 3-class classification (Focus Level 0, 1, 2) with Softmax.

## Configuration

You can modify `configs/config.yaml` to adjust:
- `window_size`: Temporal length of each sample (default 30 samples).
- `loso`: Toggle cross-validation mode.
- `epochs`: Number of training steps per fold.

## Overview

The primary goal of this analysis is to explore the relationship between physiological markers (such as heart rate, electrodermal activity, and skin temperature) and psychological states like focus and mental effort. By applying RFA, we can identify patterns that precede high-performance states or periods of intense focus.

## Data Sources

The project integrates diverse data types:
- **Wearable Sensors**: Multimodal physiological data including:
  - **Acceleration (ACC)**: 3-axis wrist movement (ACCx, ACCy, ACCz) and total magnitude (Movement).
  - **Electrodermal Activity (EDAz)**: Z-score of the individual's electrodermal activity.
  - **Skin Temperature (TEMPz)**: Z-score of the individual's skin temperature.
  - **Heart Rate (HRz)**: Z-score of the individual's heart rate.
- **Task Context**: 
  - **Puzzle**: Identifier for the specific task/puzzle completed.
  - **Difficulty level**: Ordinal variable of puzzle difficulty.
  - **Correct**: Success indicator for the task.
- **Subjective Metrics**: 
  - **Focus level**: Ordinal variable (0: not focused, 1: moderately focused, 2: highly focused).
  - **Focus01**: Binary indicator for high focus.
- **Demographics**: Contextual information including Person (ID), Age, Education, Gender, and Ethnicity.

## Key Objectives

1. **Correlation Analysis**: Identify significant physiological predictors of mental focus.
2. **Predictive Modeling**: Develop models to estimate focus levels retrospectively from wearable data.
3. **Flow State Characterization**: Understand the physiological signatures of "highly focused" individuals during varying levels of task difficulty.
4. **Behavioral Insight**: Linking success rates in complex problem-solving (puzzles) with physiological stability and mental effort.
