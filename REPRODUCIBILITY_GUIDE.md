# Reproducibility Guide: FIUS Ultrasonic Object Classification

## Project Overview

This project implements a Convolutional Neural Network (CNN) for classifying objects using ultrasonic sensor data from the FIUS (Fast Ultrasonic Imaging System) hardware. The system processes ADC waveforms and FFT data to classify objects into categories: wall, person, chair, backpack, plant, and bigtable. The CNN architecture uses dual-branch inputs (ADC and FFT) with adaptive pooling and fully connected layers for real-time AIS (Autonomous Indoor System) applications requiring <10ms inference latency.

## Required Environment

- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib, Scikit-learn
- CUDA-compatible GPU (recommended for training)
- Red Pitaya FIUS hardware (for data collection)

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, numpy, matplotlib; print('Environment ready')"
```

## How to Preprocess Raw Data

The project uses preprocessed binary data, but raw data preprocessing can be performed if needed:

```bash
# Convert raw binary data to processed format
python src/convert_data.py
```

This creates normalized ADC and FFT measurements with peak alignment from the raw data structure.

## How to Train the Flat CNN

### Basic Training
```bash
python src/train.py --epochs 100 --balanced_sampling
```

### Training with Recommended Parameters
```bash
python src/train.py --epochs 150 --batch_size 16 --learning_rate 0.00005 --augment --balanced_sampling
```

### Custom Training Options
- `--augment`: Enable data augmentation
- `--balanced_sampling`: Use balanced class sampling
- `--val_ratio 0.3`: Set validation split ratio
- `--seed 12345`: Set random seed for reproducibility

## How to Evaluate the Flat CNN

### Full Evaluation
```bash
python src/evaluate.py
```

### File-Level Evaluation
```bash
python src/evaluate_file_level.py
```

### Export Results
```bash
python src/export_result_tables.py
```

## How to Run Real-Time/File Prediction

### Single File Prediction
```bash
python src/predict.py --input data/raw/test/adc_1.txt
```

### Batch Prediction with Profiling
```bash
python src/predict.py --input data/binary/backpack/adc_measurements/adc_backpack_0.53m_.npy --profile_latency
```

### Pulse-Level Latency Profiling
```bash
python src/profile_pulse_latency.py
```

## How to Generate Plots/Tables

### Architecture Diagrams
```bash
python src/plot_architecture_diagrams.py
```

### Training Curves and Metrics
Generated automatically during training and saved to `results/` directory.

### Result Tables
```bash
python src/export_result_tables.py
```

## How to Run Hierarchical CNN (If Desired)

### Training
```bash
python src/train_hierarchical.py --epochs 100
```

### Evaluation
```bash
python src/evaluate_hierarchical.py
```

### Prediction
```bash
python src/predict_hierarchical.py --input path/to/data.npy
```

## Where Outputs Are Saved

- **Models**: `models/` directory (e.g., `fius_cnn_v1.pth`)
- **Results**: `results/` directory
  - Training history: `results/training_history.json`
  - Evaluation metrics: `results/file_level_results.json`, `results/file_level_results.csv`
  - Plots: `results/cnn_architecture_diagram.png`, `results/ais_pipeline_diagram.png`, etc.
- **Logs**: Console output and saved metrics in results files

## Common Troubleshooting Notes

### CUDA Issues
- Ensure CUDA is installed: `nvcc --version`
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce batch size if out of memory: `--batch_size 8`

### Data Loading Errors
- Verify data files exist: `find data/ -name "*.npy" | wc -l` (should be ~95 files)
- Check file integrity: `python -c "import numpy as np; print(np.load('path/to/file.npy').shape)"`

### Training Issues
- If loss doesn't decrease, try enabling augmentation: `--augment`
- For poor accuracy, increase epochs or check class balance

### Prediction Errors
- Ensure model file exists in `models/`
- Check input file format matches expected shape (4, 2048)

## Recommended Final Workflow for Report Reproduction

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess Data (if needed)**
   ```bash
   python src/convert_data.py
   ```

3. **Train Model**
   ```bash
   python src/train.py --epochs 150 --augment --balanced_sampling
   ```

4. **Evaluate Model**
   ```bash
   python src/evaluate.py
   python src/evaluate_file_level.py
   ```

5. **Generate Diagrams**
   ```bash
   python src/plot_architecture_diagrams.py
   ```

6. **Profile Latency**
   ```bash
   python src/profile_pulse_latency.py
   ```

7. **Export Results**
   ```bash
   python src/export_result_tables.py
   ```

8. **Verify AIS Requirements**
   - Check latency < 10ms in profiling output
   - Review accuracy metrics in results files

This workflow reproduces all results, figures, and performance metrics presented in the project report.