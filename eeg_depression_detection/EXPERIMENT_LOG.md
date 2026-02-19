# Experiment Log: EEG Depression Detection

## Experiment Information
- **Date Started:** 2026-01-31
- **Dataset:** Figshare MDD EEG (EDF format)
- **Condition:** Eyes Closed (EC)
- **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU
- **Framework:** PyTorch 2.5.1 + CUDA 12.1

---

## Dataset Statistics

### Raw Dataset
- **Total EDF files:** 181
- **EC (Eyes Closed) files:** 58
- **MDD patients:** 34 subjects
- **Healthy controls:** 30 subjects (some files have duplicates)

### Processed Dataset (3 subjects test)
- **Samples per subject:** ~150 epochs (4-second windows, 50% overlap)
- **WPD feature shape:** [19 channels, 576 features]
- **Scalogram shape:** [64 frequencies, 128 time bins]
- **Feature extraction time:** ~2 minutes per EDF file

---

## Environment Setup Log

### 1. Virtual Environment Creation
```bash
python3.12 -m venv venv
```

### 2. Dependencies Installed
```
PyTorch 2.5.1+cu121
torch-geometric 2.7.0
numpy 1.26.4
scipy 1.17.0
mne 1.11.0
pywavelets 1.9.0
scikit-learn 1.8.0
captum 0.8.0
```

### 3. CUDA Verification
```
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
```

---

## Training Runs

### Run 1: Full Dataset Training with LOSO CV
- **Start time:** 2026-01-31 16:36:08
- **Configuration:**
  - Batch size: 16
  - Gradient accumulation: 2 (effective batch = 32)
  - Learning rate: 1e-4
  - Max epochs: 100
  - Early stopping patience: 15
  - Mixed precision: True (FP16)
  - Cross-validation: LOSO (Leave-One-Subject-Out)
  - Number of subjects: ~50+ (EC condition)

- **Status:** RUNNING
  - Phase 1: Feature extraction (~2 hours for 58 EC files)
  - Phase 2: LOSO cross-validation (one fold per subject)

- **Output directory:** outputs/run_20260131_163608
- **Log file:** training_log.txt

---

## Model Architecture Summary

### Transformer Branch
- d_model: 128
- heads: 4
- layers: 4
- FFN dim: 512
- Input: Scalograms [64 × 128]

### GNN Branch
- Node features: 576 (WPD per electrode)
- Hidden dim: 128
- Heads: 4
- Layers: 3
- Graph: 19 nodes (10-20 montage)

### Fusion
- Cross-attention with gating
- Fusion dim: 128

### Total Parameters: ~2.5M (estimated)

---

## Results

### LOSO Cross-Validation Results
[To be filled after training]

| Fold | Subject | AUC-ROC | Accuracy | Sensitivity | Specificity |
|------|---------|---------|----------|-------------|-------------|
| 1    |         |         |          |             |             |
| 2    |         |         |          |             |             |
| ...  |         |         |          |             |             |

### Aggregated Results
| Metric | Mean | Std |
|--------|------|-----|
| AUC-ROC | | |
| Accuracy | | |
| Sensitivity | | |
| Specificity | | |
| F1-Score | | |

---

## Notes and Observations

1. **Dataset Format:** EDF files, not .mat as originally assumed
2. **Processing Time:** Feature extraction is slow (~2 min/file) but cached
3. **Memory Usage:** [To be monitored]
4. **GPU Utilization:** [To be monitored]

---

## File References

- Dataset: `/home/jabe/Workspace/pra/eeg_depression_detection/data/raw/figshare/`
- Cache: `/home/jabe/Workspace/pra/eeg_depression_detection/data/raw/figshare/cache/`
- Checkpoints: `/home/jabe/Workspace/pra/eeg_depression_detection/checkpoints/`
- Logs: `/home/jabe/Workspace/pra/eeg_depression_detection/logs/`
