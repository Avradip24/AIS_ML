# Reproducibility Guide: FIUS Ultrasonic Object Classification

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (optional, but recommended for training)
- Red Pitaya FIUS hardware (for data collection)

## Environment Setup

### 1. Clone and Install

```bash
git clone <repository-url>
cd AIS_ML_Project/AIS_ML
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy, matplotlib; print('Dependencies OK')"
```

## Data Preparation

### 1. Raw Data Structure

Ensure your data follows this structure:
```
data/
├── binary/
│   ├── backpack/
│   │   └── adc_measurements/
│   │       ├── adc_backpack_0.53m_.npy
│   │       └── ...
│   ├── bigtable/
│   ├── chair/
│   ├── person/
│   ├── plant/
│   └── wall/
└── raw/
    ├── backpack/
    │   ├── adc_graphs/
    │   ├── adc_measurements/
    │   ├── fft_graphs/
    │   └── fft_measurements/
    └── ...
```

### 2. Preprocessing

Convert raw binary data to processed format:

```bash
# Process all binary files
python src/convert_data.py

# Expected output: 95 files processed
# Creates normalized ADC and FFT data with peak alignment
```

## Model Training

### Flat CNN Training

#### Basic Training
```bash
python src/train.py
```

#### Training with Augmentation (Recommended)
```bash
python src/train.py --augment
```

#### Custom Training Parameters
```bash
# Adjust batch size
python src/train.py --batch_size 32

# Change validation ratio
python src/train.py --val_ratio 0.3

# Custom seed for reproducibility
python src/train.py --seed 12345
```

### Hierarchical CNN Training

```bash
python src/train_hierarchical.py
```

## Model Evaluation

### Flat CNN Evaluation

```bash
# Evaluate all classes
python src/evaluate.py

# Evaluate specific classes
python src/evaluate.py --classes "wall,person,chair"

# Quick evaluation (fewer recordings)
python src/evaluate.py --quick
```

### Hierarchical CNN Evaluation

```bash
python src/evaluate_hierarchical.py
```

### Latency Measurement

```bash
python src/measure_latency.py
```

## Real-time Prediction

### Single Sample Prediction

#### Flat CNN
```bash
# Predict from ADC file
python src/predict.py --input data/binary/backpack/adc_measurements/adc_backpack_0.53m_.npy

# Predict with confidence threshold
python src/predict.py --input path/to/data.npy --threshold 0.8
```

#### Hierarchical CNN
```bash
python src/predict_hierarchical.py --input path/to/data.npy
```

### Batch Prediction

```bash
# Process multiple files
for file in data/binary/*/*.npy; do
    python src/predict.py --input "$file"
done
```

## Configuration

### Model Configuration

Edit `config.yaml` to modify:

```yaml
training:
  batch_size: 16
  epochs: 150
  learning_rate: 0.00005

dataset:
  input_size: 2048
  samples_per_file: 400
  classes: ["Wall", "Person", "Chair", "Backpack", "Plant", "BigTable"]
```

### Class Merging

The system automatically merges `BigTable` into `Wall` class for 5-class classification.

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
python src/train.py --batch_size 8
```

#### 2. Missing Model Files
```bash
# Check model directory
ls -la models/

# Retrain if models are missing
python src/train.py
python src/train_hierarchical.py
```

#### 3. Data Loading Errors
```bash
# Verify data structure
find data/ -name "*.npy" | wc -l

# Check file integrity
python -c "import numpy as np; print(np.load('path/to/file.npy').shape)"
```

#### 4. Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Performance Issues

#### Slow Training
- Use GPU if available
- Reduce batch size
- Use `--quick` for development

#### Poor Accuracy
- Enable augmentation: `--augment`
- Increase epochs in config
- Check class balance in dataset

## Expected Outputs

### Training Logs
```
Epoch 1/150: Train Loss: 1.824, Val Loss: 1.956, Val Acc: 25.2%
...
Epoch 150/150: Train Loss: 1.123, Val Loss: 2.023, Val Acc: 35.4%
```

### Evaluation Results
```
Validation Metrics:
Val Loss     : 2.0231
Val Accuracy : 35.44%
Val Macro-F1 : 0.3174
```

### Prediction Output
```
Prediction: backpack (confidence: 0.85)
Top-2: backpack (0.85), plant (0.12)
Status: confident
```

## Validation Checklist

- [ ] Dependencies installed
- [ ] Data files present (95 .npy files)
- [ ] Preprocessing completes without errors
- [ ] Training converges (loss decreases)
- [ ] Evaluation runs successfully
- [ ] Latency < 10ms
- [ ] Prediction works on sample data

## Performance Benchmarks

| Hardware | Training Time (150 epochs) | Inference Latency |
|----------|---------------------------|------------------|
| CPU i7-8700K | ~45 minutes | 2.5 ms |
| GPU RTX 3080 | ~8 minutes | 0.8 ms |

## Contact

For issues or questions, refer to the project documentation or contact the development team.