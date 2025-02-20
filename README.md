# MCL
MCL: Attention-Driven Meta-Learning for Robust Cross-Subject EEG-fNIRS Classification

This repository contains the implementation of Meta-CNN-LSTM (MCL), an attention-driven meta-learning model designed for robust cross-subject classification of EEG-fNIRS data. The MCL model integrates:
- Meta-learning optimization for adaptive subject generalization.
- CNN-LSTM hybrid architecture to capture spatial-temporal dependencies.
- Attention mechanisms to enhance feature importance.
- Event-Related Desynchronization (ERD/ERS)
- Feature optimization across time (TD), frequency (FD), and time-frequency (TFD) domains.

MCL achieves 96.6% cross-subject accuracy on cognitive BCI tasks, outperforming state-of-the-art methods while maintaining efficiency for real-time brain-computer interface (BCI) applications.

# Features
- Meta-Learning Framework: Adaptively optimizes feature representations for subject-independent classification.
- EEG-fNIRS Data Fusion: Integrates complementary neural signals to enhance classification robustness.
- Attention Mechanism: Selectively enhances critical EEG-fNIRS features, reducing irrelevant noise.
- Optimized Feature Selection: Identifies key EEG-fNIRS features (e.g., FDS, TDT) for improved model generalization.
  
# File Structure
- DataAugmentation.py --> Data augmentation pipeline for EEG-fNIRS features
- ModelInitialization.py --> Initializes task distributions and optimizers
- TrainingLoop.py --> Implements inner and outer loop training for meta-learning
- ModelArchitecture.py --> Defines the CNN-LSTM architecture with attention mechanism

# If you find this work useful, please cite our article:
