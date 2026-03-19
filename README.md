## FIUS — Deep Learning-Based Ultrasonic Object Classification
   **Combined AIS + ML Project — Winter Semester 2025/2026**  
      Frankfurt University of Applied Sciences (FRA-UAS)  

---

## Table of Contents

- [Project Context](#project-context)
- [The FIUS Sensor](#the-fius-sensor)
- [AIS Deliverables](#ais-deliverables)
- [ML Deliverables](#ml-deliverables)
- [System Architecture](#system-architecture)
- [Model Architectures](#model-architectures)
- [Results — ML](#results--ml)
- [Results — AIS Validation](#results--ais-validation)
- [Training Curves](#training-curves)
- [Confusion Matrices](#confusion-matrices)
- [Results Dashboard](#results-dashboard)
- [Codebase Structure](#codebase-structure)
- [How to Run](#how-to-run)
- [Key Engineering Decisions](#key-engineering-decisions)

---

## Project Context

This project combines two course requirements — **Autonomous Intelligent Systems (AIS)** and **Machine Learning (ML)** — into a single integrated system. The AIS component defines the real-time control loop, sensor integration, and latency constraints. The ML component provides the classification intelligence that fills the "Perceive" phase of that loop.

The task: classify indoor objects from raw ultrasonic echo waveforms captured by the FRA-UAS Intelligent Ultrasonic Sensor (FIUS), in real time, meeting a strict **< 10 ms per-pulse inference requirement**.

---

## The FIUS Sensor

The FIUS sensor is built around a **40 kHz ultrasonic transducer** interfaced with a **Red Pitaya STEMlab 125-14** board. It operates on the pulse-echo principle:

```
Sensor emits 40 kHz burst
         │
         ▼
  Sound travels through air at 343 m/s
         │
         ▼
  Hits object at distance x
         │
         ▼
  Echo returns to membrane
         │
         ▼
  Time-of-flight: x = (c × t) / 2
```

**Hardware specifications:**

| Parameter | Value |
|---|---|
| Transducer frequency | 40 kHz |
| Sampling rate (ADC) | 1.953 MHz |
| Buffer size | 2,048 samples per pulse |
| Pulses per recording | 50 |
| Communication | WiFi (UDP, static IP) |
| Threshold voltage | 2.5 V (I²C controlled) |
| Platform | Red Pitaya STEMlab 125-14 (ARM Cortex-A9) |
| Data acquisition GUI | UDP Client V0.23 (FRA-UAS) |

The Red Pitaya captures raw ADC waveforms (amplitude vs sample index) and optionally FFT magnitude spectra from the on-board signal processing pipeline. Both are stored as paired `.txt` files per recording session.

**Acoustic physics of the five classes:**

| Class | Surface type | Echo character | Distance range |
|---|---|---|---|
| Wall | Flat hard — specular | Sharp, high amplitude, early | 0.39–2.02 m |
| Person | Irregular, fabric — diffuse | Broader, medium amplitude, variable | 1.25–2.21 m |
| Chair | Structured hard — mixed | Sharp, medium, consistent | 1.25–1.91 m |
| Backpack | Soft absorber — diffuse | Weaker, longer tail | 0.53–1.41 m |
| Plant | Complex irregular — diffuse | Weakest, most complex shape | 0.68–1.16 m |

> **Critical observation from raw data analysis:** Person at 1.28 m and chair at 1.26 m produce echoes arriving at samples 8375 and 8456 respectively — only **81 samples (41 µs) apart**. The system cannot rely on time-of-flight alone for these classes; it must use amplitude and spectral shape.

---

## AIS Deliverables

The AIS component concerns the **Sense → Perceive → Decide loop**, system integration, real-time validation, and hardware interfacing.

### AIS-1: Hardware Integration and Wireless Data Acquisition

- FIUS sensor interfaced to Red Pitaya STEMlab 125-14 via I²C
- WiFi-based UDP data pipeline (static IP configuration, no tethering)
- UDP Client GUI (V0.23) configured: Buffer = 2048, Measurements = 50
- ADC signals captured as 16-bit raw amplitude arrays at 1.953 MHz
- FFT spectra captured in paired files (22 bins for near-field, 85 bins for far-field objects)

### AIS-2: The Sense–Perceive–Decide Loop

```
┌────────────────────────────────────────────────────────────────┐
│                     AIS CONTROL LOOP                           │
│                                                                │
│  SENSE          PERCEIVE              DECIDE                   │
│  ─────          ────────              ──────                   │
│  FIUS sensor  → CNN inference    →  Safety output              │
│  Red Pitaya     (< 1.29 ms/pulse)   Obstacle class             │
│  WiFi/UDP       4-channel tensor    Confidence score           │
│  50 pulses/file Dual-branch model   Consensus vote             │
│                 31,365 params        Navigation command        │
└────────────────────────────────────────────────────────────────┘
```

### AIS-3: Real-Time Latency Validation

The AIS constraint requires **< 10 ms per-pulse inference latency** for safety-critical navigation.

| Metric | Value | AIS Requirement | Status |
|---|---|---|---|
| Flat CNN — pure forward-pass / pulse | **1.29 ms** | < 10 ms | ✅ 7.7× headroom |
| Hierarchical CNN — forward-pass / pulse | **1.54 ms** | < 10 ms | ✅ 6.5× headroom |
| Random Forest — full inference | ~210 ms | < 10 ms | Does not meet |
| SVM / Logistic Regression | ~187 ms | < 10 ms | Does not meet |

> Latency measured: 6 files × 10 runs × 50 pulses = **3,000 timing measurements**. Warmup runs excluded. CPU only (no GPU acceleration).

**Latency breakdown (Flat CNN):**

| Phase | Time |
|---|---|
| File I/O + 8-row averaging | ~293 ms (one-time, not AIS-relevant) |
| Per-pulse preprocessing | ~0.01 ms |
| **Per-pulse model forward-pass** | **1.29 ms** ← AIS metric |
| Aggregation / consensus voting | ~0.11 ms |

The AIS latency requirement applies specifically to the per-pulse forward-pass, which is the bottleneck in a real-time streaming scenario. File I/O is a one-time cost independent of model complexity.

### AIS-4: Safety-Critical Decision Logic

The "Decide" phase implements three-way consensus voting to minimize false negatives on person detection:

1. **Mean probability vote** — average Softmax probabilities across all 50 pulses
2. **Majority vote** — most frequent argmax prediction across pulses
3. **Confidence-weighted vote** — votes weighted by per-pulse max confidence

If any two of these agree, the consensus label is used. This redundancy is specifically motivated by safety: a missed person detection has higher cost than a false alarm.

### AIS-5: System Validation Results

| Validation metric | Result |
|---|---|
| File-level accuracy (6 held-out files) | **100%** — 6/6 correct |
| AIS latency met (< 10 ms / pulse) | **✅ Yes** — 1.29 ms |
| Inference in real-time on CPU | **✅ Yes** — no GPU required |
| Consistent predictions (no flip between runs) | **✅ Yes** — deterministic at file level |

---

## ML Deliverables

The ML component concerns the dataset, models, training methodology, and classification performance.

### ML-1: Dataset Construction

**Collection protocol:**
- 95 paired ADC + FFT recording files across 5 object classes
- Each file: 50 pulses × 2,048 ADC samples + FFT spectrum
- Multiple orientations, distances, and sessions per class
- Bigtable merged into Wall class (acoustically indistinguishable)

**Final dataset:**

| Class | Segments | Files | Distance range | Sessions |
|---|---|---|---|---|
| Wall | 1,750 | 32 | 0.39–2.02 m | 5 orientations |
| Person | 800 | 16 | 1.25–2.21 m | 16 sessions |
| Chair | 800 | 16 | 0.86–1.91 m | 6 orientations |
| Backpack | 850 | 16 | 0.53–1.41 m | 6 orientations |
| Plant | 800 | 15 | 0.68–1.16 m | 5 conditions |
| **Total** | **5,000** | **95** | — | — |

**Train / validation split:** Recording-level stratified split (val_ratio=0.2, seed=42). All pulses from a single recording stay in the same split — no data leakage.

```
Recording counts after split:
  wall       train=26  val=6   (1,450 / 300 segments)
  person     train=13  val=3   (650 / 150 segments)
  chair      train=13  val=3   (650 / 150 segments)
  backpack   train=13  val=3   (700 / 150 segments)
  plant      train=12  val=3   (650 / 150 segments)
```

### ML-2: Feature Engineering — 4-Channel Tensor

Each training sample is a 4-channel tensor of shape **(4, 2048)**:

| Channel | Description | Formula |
|---|---|---|
| ADC_norm | Max-abs normalized waveform | signal / max\|signal\| |
| ADC_energy | Cumulative energy envelope | z-score(cumsum\|ADC_norm\|) |
| FFT_norm | Max-abs normalized spectrum | fft / max\|fft\| |
| FFT_energy | Cumulative spectral energy | z-score(cumsum\|FFT_norm\|) |

The energy channels capture **rate of energy accumulation** — hard specular reflectors (wall, chair) release energy rapidly in a sharp peak, while soft diffuse scatterers (person, plant) spread energy over a longer tail. This is the primary distinguishing feature between the two acoustic groups.

### ML-3: Data Augmentation

Applied only to training samples (val set uses clean data):

| Augmentation | Parameters | Purpose |
|---|---|---|
| Gaussian noise on ADC | σ = 0.02, prob = 0.7 | Simulate SNR variation |
| Gaussian noise on FFT | σ = 0.01, prob = 0.5 | Simulate spectral noise |
| Amplitude scaling | ×[0.85, 1.15], prob = 0.6 | Distance variation |
| Time shift | ±1% (±20 samples), prob = 0.5 | Minor position jitter |

> Time shift was capped at ±1% (±20 samples). A previous ±5% setting caused person/chair inter-class confusion because their echo peaks are only 81 samples apart.

### ML-4: Models Implemented

Three model families were implemented and benchmarked:

**A. Flat 1D-CNN (primary model)**  
Dual-branch architecture with SE attention — see [Model Architectures](https://github.com/Avradip24/AIS_ML/tree/main/models)

**B. Hierarchical CNN (soft-routing)**  
Three-model pipeline with group → fine classification

**C. Classical Baselines**  
Logistic Regression, Random Forest, SVM (RBF) — trained on 76 hand-crafted features per recording

### ML-5: Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam | Adaptive learning for sparse gradients |
| Learning rate | 5×10⁻⁵ | Stable for long runs without scheduler |
| Loss | CE + label smoothing (0.1) | Prevents overconfidence on ambiguous echoes |
| Class weighting | Inverse-frequency | wall 0.51×, others 1.06–1.14× |
| Balanced sampling | WeightedRandomSampler | Corrects 2.2× wall imbalance |
| Dropout | 0.4 (FC layers) | Reduces overfitting on small dataset |
| Gradient clipping | max_norm = 1.0 | Prevents FFT spike-induced explosions |
| Batch size | 8 | |
| Epochs | 100 | No early stopping |
| Best checkpoint | Epoch 97 | Saved by Val Macro-F1 |

### ML-6: Classification Results

Full results are detailed in [Results — ML](https://github.com/Avradip24/AIS_ML/tree/main/results) section.

| Model | Val Accuracy | Macro-F1 |
|---|---|---|
| **Flat CNN** | **82.1%** | **0.823** |
| Hierarchical CNN | 73.7% | 0.728 |
| Random Forest | 58.3% ± 2.8% | 0.522 |
| SVM (RBF) | 52.8% ± 2.8% | 0.505 |
| Logistic Regression | 55.6% ± 0.0% | 0.560 |

---

## System Architecture

<img width="2409" height="1359" alt="image" src="https://github.com/user-attachments/assets/3aaa9c67-8e57-4476-b056-d7c12a0aca9d" />

The complete Sense → Preprocess → Perceive → Decide pipeline. The FIUS sensor captures raw echo data via Red Pitaya and WiFi UDP. The preprocessing stage applies 8-row temporal averaging and 4-channel feature extraction. Two CNN branches process the tensor in parallel; consensus voting produces the final safety decision within the AIS latency budget.

---

## Model Architectures

### Flat CNN — UltrasonicCNN

<img width="2407" height="1209" alt="image" src="https://github.com/user-attachments/assets/a81ff64c-76e6-49f3-8219-d205e0fb8cf8" />


**Why dual-branch?** ADC and FFT carry complementary information. ADC reveals *when* the echo arrives (distance/timing). FFT reveals *what material* produced the echo (resonance signature). Separating the branches allows the model to learn domain-specific features before fusion.

**Why SE blocks?** The Squeeze-and-Excitation mechanism learns to re-weight the two channels (waveform vs energy envelope) per sample. For a wall echo, the waveform channel is most informative; for plant, the energy envelope provides the better signal.

**Why AdaptiveMaxPool instead of AvgPool?** Ultrasonic echo peaks are sharp, high-amplitude transients. Max pooling preserves these peaks. Average pooling would smear them.

```
Parameter count breakdown:
  ADC conv layers:     9,040
  FFT conv layers:     9,040
  SE blocks × 2:       2,256
  Classification head: 11,029
  ─────────────────────────
  Total:              31,365
```

### Hierarchical CNN — Soft Routing

```
                    Input (50, 4, 2048)
                           │
                    ┌──────▼──────┐
                    │   Group     │  2-class: Group0 vs Group1
                    │ Classifier  │  Accuracy: 80.1%
                    └──────┬──────┘
                  p(g0)    │    p(g1)
              ┌────────────┴────────────┐
              ▼                         ▼
      ┌───────────────┐        ┌────────────────┐
      │   Fine-0      │        │    Fine-1      │
      │ person        │        │  wall          │
      │ backpack      │        │  chair         │
      │ plant         │        │                │
      └───────┬───────┘        └────────┬───────┘
              └──────────┬─────────────┘
                         │ Soft fusion
                         ▼
              final = p(g0)·fine0 + p(g1)·fine1

  Group 0 (soft/absorbing):  person, backpack, plant
  Group 1 (hard/reflective): wall, chair
```

The soft fusion means even if the group classifier is uncertain (it is correct only 80.1% of the time), both fine classifiers contribute — the correct class can still win through the weighted combination.

---

## Results — ML

### Model Comparison

<img width="1785" height="792" alt="image" src="https://github.com/user-attachments/assets/b857ca95-3fad-48b0-a8da-fffec8097ea3" />

| Model | Val Accuracy | Macro-F1 | Params | Evaluation level |
|---|---|---|---|---|
| **Flat CNN** ★ | **82.1%** | **0.823** | 31,365 | Segment (1,000 val) |
| Hierarchical CNN | 73.7% | 0.728 | ~94,095 | Segment (900 val) |
| Random Forest | 58.3% ± 2.8% | 0.522 | — | Recording (18 val) |
| SVM (RBF) | 52.8% ± 2.8% | 0.505 | — | Recording (18 val) |
| Logistic Regression | 55.6% ± 0.0% | 0.560 | — | Recording (18 val) |

> Classical baselines operate at recording level (18 val recordings across 2 seeds). CNN models operate at segment level (1,000 val segments). Direct accuracy comparison should account for this difference in evaluation granularity.

### Per-Class Accuracy

<img width="1935" height="765" alt="image" src="https://github.com/user-attachments/assets/161ab3f9-b8a4-44e2-b991-1d2907dcd3ad" />

| Class | Flat CNN | Hierarchical CNN | Why the gap? |
|---|---|---|---|
| **Plant** | 93.6% | 93.6% | Unique diffuse low-energy echo — easy for both |
| **Chair** | 87.1% | 78.2% | Flat CNN benefits from full context; hier. routing helps but group boundary noisy |
| **Person** | 86.2% | 49.3% | Multi-distance variation hurts hierarchical routing |
| **Backpack** | 85.4% | 90.3% | Hierarchical group0 excels — soft absorber well separated |
| **Wall** | 70.6% | 66.1% | Hardest: spans 0.39–2.02 m, echo timing varies 14× |

### Baseline Comparison Detail

```
Classical baselines — recording-level evaluation (2 seeds: 42, 7)

Model                 Acc mean   Acc std    F1 mean    F1 std
──────────────────────────────────────────────────────────────
Random Forest         58.3%     ±2.8%      0.522     ±0.019
Logistic Regression   55.6%     ±0.0%      0.560     ±0.013
SVM (RBF)             52.8%     ±2.8%      0.505     ±0.014

Feature set: 76 features per recording
  - Peak position (echo timing proxy for distance)
  - Peak amplitude
  - Signal energy
  - Zero-crossing rate
  - Skewness and kurtosis
  - Energy ratio (first half vs second half)
  All computed across all 4 channels (ADC, FFT, + energy envelopes)
```

**Why CNN outperforms Random Forest here:**  
Random Forest is evaluated on only 18 val recordings (small, high-variance). CNN evaluates on 1,000 segments with stable metrics. Additionally, the CNN's convolutional layers automatically detect echo peak shape and timing relationships that fixed features cannot fully capture — particularly the subtle amplitude and spectral differences between person and chair at the same distance.

---

## Results — AIS Validation

### Latency Analysis

<img width="1783" height="684" alt="image" src="https://github.com/user-attachments/assets/efbef980-164a-40eb-9046-30c220fac2f8" />

```
Latency profiling methodology:
  - 6 test files × 10 runs × 50 pulses = 3,000 measurements
  - First 3 runs per file excluded (warmup / JIT compilation)
  - Reported: mean ± std of remaining runs
  - Platform: CPU only (Windows, Intel Core, no GPU)

Pure model forward-pass per pulse:
  Flat CNN:        1.29 ms ± 0.49 ms   → AIS requirement: < 10 ms ✅
  Hierarchical:    1.54 ms ± 0.41 ms   → AIS requirement: < 10 ms ✅
  Random Forest:   ~210 ms             → Does NOT meet requirement 
  SVM / LR:        ~187 ms             → Does NOT meet requirement 
```

### File-Level Live Prediction

All 6 held-out test files correctly classified. Results from `evaluate_file_level.py`:

| File | True Label | Flat CNN | Confidence | Margin | Hierarchical | Baseline RF |
|---|---|---|---|---|---|---|
| adc_1.txt | person | **person ✓** | 67.7% | 48.4% | person ✓ | person ✓ |
| adc_2.txt | plant | **plant ✓** | 43.2% | 15.6% | plant ✓* | plant ✓ |
| adc_3.txt | backpack | **backpack ✓** | 71.9% | 55.0% | backpack ✓ | backpack ✓ |
| adc_4.txt | wall | **wall ✓** | 62.6% | 34.3% | wall ✓ | wall ✓ |
| adc_5.txt | chair | **chair ✓** | 70.8% | 51.0% | chair ✓ | chair ✓ |
| adc_6.txt | wall | **wall ✓** | 44.6% | 15.4% | wall ✓ | wall ✓ |
| **Score** | | **6/6 (100%)** | | | **6/6 (100%)** | **6/6 (100%)** |

*adc_2 hierarchical prediction used confidence-weighted fallback on uncertain vote (plant vs person 38%/33%)

> **Note:** Confident predictions (margin > 30%) were adc_1, adc_3, adc_4, adc_5. The two wall-at-distance files (adc_2 = plant at ~1.16 m, adc_6 = wall at far distance) had lower margins — correctly flagged as CONFIDENT but with less decisive vote distribution.

---

## Training Curves

<img width="1936" height="688" alt="image" src="https://github.com/user-attachments/assets/b3dbe468-fe47-43d9-a74d-51301ea3f99d" />

**Flat CNN — 100 epoch run with balanced sampling:**

- Val loss decreased continuously: 1.618 → 0.800 (no plateau — model still improving at epoch 100)
- Val accuracy: 30.7% → 80.9% peak (epoch 91), settled at 82.1% on best checkpoint
- Best Macro-F1: **0.814** at epoch 97
- Training time: **409 seconds** (~7 min) on CPU
- No early stopping triggered — fixing the augmentation-on-validation bug allowed full 100-epoch training

The steady decline without plateau confirms that the normalization fix (matching train and prediction scales) was the critical factor — the model learned genuinely improving representations throughout.

---

## Confusion Matrices

<img width="2080" height="840" alt="image" src="https://github.com/user-attachments/assets/ac700183-d419-40e2-9ec6-c0a621c19fe1" />

### Flat CNN — 82.1% overall (1,000 val segments)

```
                wall   person   chair   backpack   plant   row acc
wall  (n=309)    218      47      13       22        9      70.6%
person(n=138)      1     119      12        6        0      86.2%
chair (n=147)      0      18     128        0        1      87.1%
backpk(n=151)      2      13       1      129        6      85.4%
plant (n=155)      0       3       4        3      145      93.6%
```

**Main confusions explained by physics:**
- **Wall → Person (47, 15.2%):** Wall at 1.5–2.0 m has echo timing similar to person at 1.2–2.2 m. Without absolute amplitude as a discriminator (removed by normalization), these overlap.
- **Person → Chair (12, 8.7%):** Same distance, 81-sample peak separation — only amplitude and FFT shape separate them.
- **Chair → Person (18, 12.2%):** Mirror error. Person has ~68% higher peak amplitude than chair; some low-energy chair pulses fall in the person region.
- **Backpack → Plant (6) and vice versa:** Both are diffuse low-energy scatterers at similar distances.

### Hierarchical CNN — 73.7% overall (900 val segments)

```
                wall   person   chair   backpack   plant   row acc
wall  (n=310)    205       5      30       43       27      66.1%
person(n=152)     16      75      26       27        8      49.3%
chair (n=142)      5      14     111       10        2      78.2%
backpk(n=155)      4       3       7      140        1      90.3%
plant (n=141)      0       4       2        3      132      93.6%
```

**Group classifier accuracy: 80.1%** — ~20% of samples are routed to the wrong fine classifier, causing cascaded errors. The soft-routing fusion mitigates but does not eliminate this.

---

## Results Dashboard

<img width="1975" height="1417" alt="image" src="https://github.com/user-attachments/assets/201c748b-07f7-4395-ba99-4ed398d774ca" />

---

## Codebase Structure

```
AIS_ML/
├── config.yaml                          # Paths, training hyperparameters
│
├── src/
│   │
│   ├── ── Data Pipeline ──
│   ├── convert_data.py                  # Raw .txt → .npy binary (run once)
│   ├── dataset.py                       # UltrasonicDataset, augmentation, merge map
│   ├── data_loader.py                   # process_file(): 8-row avg + 4-ch features
│   ├── verify_data.py                   # Dataset integrity checker
│   │
│   ├── ── Deep Learning ──
│   ├── model.py                         # UltrasonicCNN (dual-branch, SE attention)
│   ├── train.py                         # Flat CNN training
│   ├── train_hierarchical.py            # 3-model hierarchical training
│   ├── evaluate.py                      # Flat CNN evaluation
│   ├── evaluate_hierarchical.py         # Hierarchical evaluation (soft routing)
│   │
│   ├── ── Live Inference ──
│   ├── predict.py                       # Flat CNN: live file prediction + voting
│   ├── predict_hierarchical.py          # Hierarchical: soft-routing live prediction
│   ├── predict_baseline.py              # Random Forest: live prediction
│   │
│   ├── ── Baselines ──
│   ├── baseline_models.py               # LR, RF, SVM at recording level
│   ├── benchmark_classical.py           # Segment-level classical benchmark
│   │
│   ├── ── AIS Validation ──
│   ├── benchmark.py                     # CNN forward-pass latency (100 runs)
│   ├── benchmark_baseline.py            # Baseline latency (100 runs)
│   ├── profile_latency.py               # Full pipeline latency breakdown
│   ├── profile_pulse_latency.py         # Per-pulse AIS latency profiling
│   ├── measure_latency.py               # Flat vs hierarchical model latency
│   ├── evaluate_file_level.py           # File-level accuracy on test list
│   │
│   └── ── Reporting ──
│       ├── export_result_tables.py      # JSON/CSV/Markdown export
│       └── plot_architecture_diagrams.py
│
├── data/
│   ├── raw/                             # Raw ADC + FFT .txt files (by class)
│   │   └── test/                        # 6 held-out test files
│   └── binary/                          # Preprocessed .npy (post convert_data.py)
│
├── models/
│   ├── fius_cnn_v1.pth                  # Best flat CNN checkpoint (epoch 97)
│   └── fius_hierarchical_soft_cnn.pth   # Hierarchical checkpoint
│
├── results/
│   ├── training_history.json            # Per-epoch loss/accuracy/F1
│   ├── training_history_hierarchical.json
│   ├── file_level_results.json/csv      # 6 test file predictions
│   ├── hierarchical_eval_preset_a_soft.json
│   ├── latency_profile.json/csv
│   ├── pulse_latency_profile.json/csv
│   ├── final_result_tables.json/csv/md
│   └── *.png                            # Loss curve, accuracy curve, confusion matrix
│
├── figures/                             # Analysis figures for this README
│   ├── fig1_model_comparison.png
│   ├── fig2_per_class_accuracy.png
│   ├── fig3_confusion_matrices.png
│   ├── fig4_training_curves.png
│   ├── fig5_latency_predictions.png
│   ├── fig6_system_architecture.png
│   ├── fig7_cnn_architecture.png
│   └── fig8_results_dashboard.png
│
├── test_files.txt                       # Paths to 6 test files
└── ground_truth.csv                     # True labels for test files
```

---

## How to Run

### Full Pipeline (clean start)

```bash
# 1. Convert raw recordings to binary tensors
python src/convert_data.py

# 2. Train flat CNN (primary model)
python src/train.py --epochs 100 --balanced_sampling

# 3. Evaluate flat CNN on validation set
python src/evaluate.py

# 4. Train hierarchical CNN
python src/train_hierarchical.py --epochs 100 --balanced_sampling

# 5. Evaluate hierarchical CNN
python src/evaluate_hierarchical.py
```

### Classical Baselines

```bash
# Recording-level LR, RF, SVM (2 seeds)
python src/baseline_models.py --val_ratio 0.2 --seeds 42,7 --require_fft
```

### Live Prediction (single file)

```bash
# Flat CNN
python src/predict.py --input data/raw/test/adc_1.txt

# Hierarchical CNN
python src/predict_hierarchical.py --input data/raw/test/adc_1.txt

# Random Forest baseline
python src/predict_baseline.py --input data/raw/test/adc_1.txt
```

### AIS Validation

```bash
# File-level accuracy on all 6 test files (AIS loop validation)
python src/evaluate_file_level.py \
    --test_list test_files.txt \
    --ground_truth ground_truth.csv \
    --allow_fft_fallback

# AIS latency: per-pulse model forward-pass (the critical metric)
python src/measure_latency.py

# Full pulse-level latency breakdown (3,000 measurements)
python src/profile_pulse_latency.py \
    --test_files test_files.txt \
    --output_dir results \
    --allow_fft_fallback \
    --num_runs 10

# CNN inference latency (100-run benchmark per file)
python src/benchmark.py \
    --input data/raw/test/adc_1.txt \
    --runs 100 \
    --allow_fft_fallback
```

### Export Results

```bash
python src/export_result_tables.py --results_dir results --output_dir results
```

---

## Key Engineering Decisions

### 1. Bigtable → Wall semantic merge
Bigtable and wall are acoustically identical (both large flat hard surfaces producing specular echoes). Merging reduces the task from 6 to 5 classes, removes an artificial distinction, and increases wall training data from ~1,000 to 1,750 segments. Implemented via `MERGE_MAP = {"bigtable": "wall"}` in `dataset.py`.

### 2. 8-row temporal averaging
The FIUS sensor emits 8 sub-pulses per measurement cycle. `convert_data.py` averages these 8 rows into one training sample. The prediction pipeline (`data_loader.py`) must apply the same averaging — without it, prediction features had σ ≈ 4–80 (noise-dominated raw rows) vs the training distribution σ ≈ 66 (stable averaged rows). The model was essentially guessing on all live files until this was aligned.

### 3. Max-abs normalization on both sides
Training (`dataset.py`) and prediction (`data_loader.py`) both apply `signal / max|signal|` to normalize to [-1, 1]. Without this alignment the model exploited raw amplitude as a shortcut (wall echo amplitude ~1763 vs plant ~89 ADC units), yielding inflated validation accuracy that did not generalize to live data after normalization was added to the prediction pipeline.

### 4. Augmentation on training set only
Enabling `transform=True` on a single shared dataset instance would corrupt validation with random noise on every evaluation call — producing non-deterministic val metrics that triggered early stopping too early (epoch 31 instead of running all 100 epochs). The fix creates two separate `UltrasonicDataset` instances: augmented for training, clean for validation.

### 5. Balanced sampling for wall imbalance
Wall has 1,450 training segments vs ~650 for other classes (2.2× imbalance). `WeightedRandomSampler` ensures equal class contribution to gradient updates per epoch. Combined with inverse-frequency loss weights (wall: 0.51×), this prevents the model from exploiting wall's overrepresentation.

### 6. Time shift capped at ±1% not ±5%
Person echo peak (~8375 samples) and chair echo peak (~8456 samples) differ by only 81 samples. The original ±5% time shift (±102 samples) was larger than this gap — augmentation was manufacturing person↔chair confusion by moving person waveforms into chair's echo timing zone. Reducing to ±1% (±20 samples) keeps shift smaller than the inter-class separation.

### 7. Consensus voting for file-level decision
50 pulse-level predictions per file are aggregated via three independent voting mechanisms. This redundancy was the key factor in achieving 100% file-level accuracy: even files with low per-pulse confidence (adc_2: 43.2%, adc_6: 44.6%) were correctly classified because the majority of pulses agreed on the right answer.
