# Theta-rhythmic-network-synchronization-in-a-mouse-model-of-ROSopathy

## Overview

This repository contains the implementation of a master thesis project focused on classifying LFP signals to identify different brain states and type of mice. The project employs both state-of-the-art deep learning approaches using Patch Time Series Transformers (PatchTST) and traditional machine learning methods with Support Vector Machines (SVM).

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Data Format](#data-format)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)

## Features

- **Dual Classification Approaches**: 
  - Deep Learning: PatchTST with custom convolutional heads
  - Classical ML: SVM with Continuous Wavelet Transform features
- **Brain State Classification**: Identifies 7 different signal types (SP, SO, SR, SLM, HF, DG, LS)
- **Mice Type Classification**: Identifies 2 different signal types (TG, WT)
- **Robust Validation**: Aggregated predictions for long-duration signals
- **Parallel Processing**: Efficient feature extraction using joblib
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, and accuracy metrics

## Project Structure

```
├── reading_data.ipynb          # Main notebook for data processing and model training
├── Shufle.py                   # Data splitting utility (train/val/test)
├── LayerClassification/        # Brain layer state classification
│   ├── Checker.py             # Inference script for layer classification
│   ├── pca_model.joblib       # Saved PCA model
│   ├── scaler_model.joblib    # Saved StandardScaler
│   └── svm_model.joblib       # Saved SVM model
└── TypeClassification/         # Binary WT/TG classification
    ├── main.py                # Training script for type classification
    ├── Checker.py             # Inference script
    └── *.joblib               # Saved models
```

## Requirements

### Python Libraries
```
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
torch>=1.10.0
neo>=0.10.0
pywt>=1.2.0
joblib>=1.1.0
tqdm>=4.62.0
matplotlib>=3.4.0
seaborn>=0.11.0
tsai>=0.3.0
fastai>=2.5.0
pandas>=1.3.0
```

### Hardware Requirements
- GPU recommended for deep learning training (CUDA-capable)
- Minimum 16GB RAM for processing full LFP recordings
- ~50GB storage for raw LFP data and processed files

## Data Format

### Input LFP Files
- **Format**: Raw binary signal files
- **Channels**: 11 channels
- **Sampling Rate**: 1250 Hz
- **Data Type**: int16

## Usage

### 1. Data Preprocessing

Convert raw files to NumPy arrays:
```python
from reading_data import read_eeg_file, parse_recording_metadata

# Read metadata
recordings, classes = parse_recording_metadata('path/to/SelectionForML.txt')

# Process file
signal = read_eeg_file('path/to/recording.eeg', num_channels=11)
```

### 2. Training Deep Learning Model

```python
# Initialize model
model = PatchTSTClassifier(
    c_in=1,
    c_out=len(classes),
    seq_len=1024,
    n_layers=12,
    d_model=512,
    n_heads=16
)

# Train model (see reading_data.ipynb for full training loop)
```

### 3. Training SVM Model

```bash
python TypeClassification/main.py
```

### 4. Inference

For new recordings:
```bash
# For layer classification
python LayerClassification/Checker.py

# For type classification  
python TypeClassification/Checker.py
```

## Models

### PatchTST Architecture
- **Base Model**: Patch Time Series Transformer
- **Enhancements**:
  - 12 transformer layers
  - 512-dimensional embeddings
  - 16 attention heads
  - Custom convolutional classification head (6 layers)
- **Input**: 1024-sample sequences
- **Training**: Adam optimizer, learning rate 1e-5

### SVM with CWT Features
- **Feature Extraction**: Continuous Wavelet Transform (Morlet wavelet)
- **Scales**: 9-200 (optimized for LFP frequencies)
- **Window Size**: 20 seconds
- **Dimensionality Reduction**: PCA (100 components)
- **Classifier**: SVM with RBF kernel

## Results

### Performance Metrics
- **Deep Learning Model**: Achieves low accuracy on balanced dataset
- **SVM Model**: Achieves high accuracy (96%) 
- **Validation Strategy**: Aggregated predictions over full recordings