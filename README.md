# MCL

**Meta-Learning with Multi-Domain Feature Optimization for Robust Cross-Subject Cognitive Task Classification in Hybrid EEG–fNIRS**

This repository holds the implementation of the Meta-learning Cross-subject
Learner (MCL), a model for subject-independent classification of cognitive
tasks from hybrid EEG–fNIRS recordings. The problem it targets: a classifier
trained on one group of people usually drops sharply on a new person, because
neural and hemodynamic responses vary between individuals. MCL handles this
with meta-learning rather than a single fixed classifier.

The model has three parts:

- A dual-stream CNN–LSTM encoder that reads the EEG and fNIRS branches
  separately and fuses them, capturing spatial and temporal structure.
- A meta-learning loop, an inner loop that adapts to each sampled subject and
  an outer loop that learns a shared initialization, so the model generalizes
  to subjects it has not seen.
- Multi-domain feature optimization across the time, frequency, and
  time–frequency domains.

On the cognitive BCI tasks studied here, MCL reaches up to 96.6% cross-subject
accuracy and stays competitive with prior EEG–fNIRS methods while remaining
light enough (about 55k parameters) for real-time BCI use.

## Features

- **Meta-learning.** Adapts to a new subject from a few examples instead of
  assuming one decision boundary fits everyone.
- **EEG–fNIRS fusion.** Combines the two modalities in a dual-stream encoder so
  each contributes what it is good at: EEG's timing, fNIRS's localization.
- **Multi-domain features.** Time, frequency, and time–frequency descriptors,
  with FDS and TDT coming out as the most useful for these tasks.

## File structure

Core model and training:

- `ModelArchitecture.py` — the dual-stream CNN–LSTM architecture
- `ModelInitialization.py` — sets up task distributions and optimizers
- `TrainingLoop.py` — inner- and outer-loop meta-learning training
- `DataAugmentation.py` — augmentation pipeline for EEG–fNIRS features

Features, baselines, and analysis:

- `FeatureExtraction.py` — the six feature groups (TDT, TDS, TDM, FDS, FDP, TFD)
- `Baselines.py` — Conv / Dense / Recur baselines for the exploration stage
- `Metrics.py` — accuracy/F1, cross-subject evaluation, and paired significance tests
- `Complexity.py` — parameter count, FLOPs, and asymptotic time complexity
- `Visualization.py` — t-SNE, confusion-matrix, and per-subject plots from model outputs
- `Config.py` — shared constants (tasks, feature groups, shapes, hyperparameters)

## Citation

If you find this work useful, please cite our article:
