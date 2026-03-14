# AIS_ML: Ultrasonic Object Classification with FIUS

Distinguish and classify different objects using Frequency-modulated Interrupted Ultrasonic Sensing (FIUS) on Red Pitaya hardware.

## Project Overview

This project implements two CNN architectures for ultrasonic object classification:
- **Flat CNN**: Single model classifying 5 object types (wall, person, chair, backpack, plant)
- **Hierarchical CNN**: Two-stage classifier with acoustic grouping for improved performance

## Dataset

- **Source**: Ultrasonic measurements collected using Red Pitaya FIUS hardware
- **Classes**: 5 object types (bigtable merged into wall)
- **Format**: Binary ADC data with FFT preprocessing
- **Size**: 5000 segments across 95 recordings
- **Preprocessing**: Peak alignment, normalization, energy computation

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure data is in `./data/binary/` directory

## Reproducibility Guide

### Data Preprocessing

Convert raw ultrasonic data to processed format:
```bash
python src/convert_data.py
```
This creates normalized ADC and FFT measurements with peak alignment.

### Training

#### Flat CNN
```bash
# Train without augmentation
python src/train.py

# Train with augmentation (recommended)
python src/train.py --augment
```

#### Hierarchical CNN
```bash
python src/train_hierarchical.py
```

### Evaluation

#### Flat CNN
```bash
python src/evaluate.py
```

#### Hierarchical CNN
```bash
python src/evaluate_hierarchical.py
```

### Real-time Prediction

#### Flat CNN
```bash
python src/predict.py --input path/to/adc_data.npy
```

#### Hierarchical CNN
```bash
python src/predict_hierarchical.py --input path/to/adc_data.npy
```

## Model Architecture

### Flat CNN
- Dual-branch CNN (ADC + FFT inputs)
- 4 convolutional blocks per branch
- 256 features to classifier
- Dropout 0.35

### Hierarchical CNN
- Group classifier: 2 classes (soft/absorbing vs hard/reflective)
- Fine classifiers: 3-class and 2-class models
- Hard gating for decision routing

## Performance Results

### Flat CNN
- Accuracy: 35.44%
- Macro-F1: 0.3174
- Latency: ~2.5 ms/sample

### Hierarchical CNN
- Accuracy: 47.56%
- Macro-F1: 0.4794
- Latency: ~4.6 ms/sample

Both models meet <10ms latency requirement for real-time AIS applications.

## AIS Decision Logic

The system provides:
- Top-1 classification
- Top-2 alternatives with confidence
- Uncertainty flags for safety-critical decisions
- Hierarchical routing based on acoustic properties
