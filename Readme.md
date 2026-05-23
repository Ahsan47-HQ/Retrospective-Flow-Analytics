# Retrospective Flow Analysis (RFA) for Human Physiology

Retrospective Flow Analysis (RFA) is a framework focused on analyzing human physiological signals and subjective mental states to understand the dynamics of "flow" and cognitive performance. This project utilizes data captured from wearable sensors and behavioral tasks to retrospectively model focus levels, mental effort, and task performance.

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
