# Enhanced Logging & Monitoring Guide

## Overview of Improvements

The training script (`train_best.py`) has been significantly upgraded with comprehensive logging, progress tracking, and real-time monitoring. This guide explains all the enhancements.

---

## 1. New Dependencies

Added `tqdm` for progress bars:
```bash
pip install tqdm
```

---

## 2. Logging Structure

### Phase-Based Organization

The training is now organized into **3 distinct phases** with clear headers:

```
================== PHASE 1: DATA LOADING ==================
[Progress bar: Loading EDF files]
- File counts
- Subject breakdown (MDD vs Healthy)
- Total epochs processed
- Memory footprint

================== PHASE 2: LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION ==================
[Step 1/5] Feature extraction
[Step 2/5] Feature augmentation  
[Step 3/5] LOSO CV Loop
  [Progress bar for each fold]
  - Per-model training times
  - Memory usage
  - Predictions per model

================== PHASE 3: METRICS & REPORTING ==================
- Sample-level metrics
- Subject-level metrics
- Per-model performance
- Misclassified subjects
- Training time summary
```

---

## 3. Real-Time Progress Monitoring

### 3.1 Progress Bars (tqdm)

**Example output while running:**

```
Loading EDF files: 64/64 [100%] ━━━━━━━━━━━━━━━━━━━━━━━━━ 45.2s
Feature extraction by subject: 64/64 [100%] ━━━━━━━━━━━━━ 342.5s
Feature augmentation: 64/64 [100%] ━━━━━━━━━━━━━━━━━━━━━━ 12.3s
LOSO CV Progress: 45/64 [70%] ━━━━━━━━━━━━━━━ 523.4s
```

**What each bar shows:**
- Current progress (completed/total)
- Percentage complete
- Visual progress bar
- Elapsed time + ETA

### 3.2 Per-Fold Logging

For each LOSO fold, you'll see:

```
Fold  5/64: MDD_S12    True=MDD Pred=MDD prob=0.847 (87s, ETA=45m) [✓] GPU=2340MB
```

**Fields:**
- `Fold 5/64`: Current fold number
- `MDD_S12`: Test subject ID
- `True=MDD`: Actual label
- `Pred=MDD`: Ensemble prediction
- `prob=0.847`: Subject-level probability
- `87s, ETA=45m`: Fold time & estimated remaining time
- `[✓]`: Correct classification
- `GPU=2340MB`: Current GPU memory usage

### 3.3 Model Training Details (Debug Level)

When running with verbose output, you'll see CNN epoch-by-epoch progress:

```
Epoch  1/30: Loss=0.6543 | ValLoss=0.6234 | LR=0.00100
Epoch  2/30: Loss=0.5234 | ValLoss=0.5012 | LR=0.00098
Epoch  7/30: Early stopping at epoch 7 (patience=7)
```

---

## 4. Detailed Logging Information

### 4.1 Data Loading Summary

```
────────────────────────────────────────────────────────
Data Loading Summary:
  Total subjects:        64 (34 MDD, 30 Healthy)
  Total epochs:          36,247
  Avg epochs/subject:    566
  Total data size:       3.45 GB
  Skipped files:         0
  Phase duration:        45.2s
────────────────────────────────────────────────────────
```

### 4.2 Feature Extraction Summary

```
✓ Feature extraction complete in 342.5s (36,247 epochs processed)
  Feature dimension: 912
  Average 105.7 epochs/sec
```

### 4.3 Per-Model Performance Breakdown

```
[Per-Model Performance]
  CNN        - Subject Accuracy: 0.7188 (46/64 subjects)
  XGB        - Subject Accuracy: 0.8125 (52/64 subjects)
  SVM        - Subject Accuracy: 0.7969 (51/64 subjects)
```

### 4.4 Sample vs Subject-Level Metrics

```
────────────────────────────────────────────────────────
SAMPLE-LEVEL METRICS (Each epoch)
────────────────────────────────────────────────────────
  Accuracy:    0.7845
  F1 Score:    0.7623
  AUC-ROC:     0.8342
  Precision:   0.7534
  Recall:      0.7712
  Sensitivity: 0.7712
  Specificity: 0.7934
  Confusion Matrix:
    Predicted:  Healthy  MDD
    Healthy:    14234    2156
    MDD:        3241     16616

────────────────────────────────────────────────────────
SUBJECT-LEVEL METRICS (Averaged across epochs)
────────────────────────────────────────────────────────
  Accuracy:    0.8438 (54/64 subjects correct)
  F1 Score:    0.8456
  AUC-ROC:     0.8975
  Precision:   0.8529
  Recall:      0.8571
  Sensitivity: 0.8571
  Specificity: 0.8333
  Confusion Matrix:
    Predicted:  Healthy  MDD
    Healthy:    25       5
    MDD:        5        29
```

### 4.5 Misclassified Subjects

```
  ⚠️  Misclassified subjects (10):
    MDD_S8:  True=MDD      → Pred=Healthy  (prob=0.412)
    H_S15:   True=Healthy  → Pred=MDD      (prob=0.587)
    MDD_S22: True=MDD      → Pred=Healthy  (prob=0.468)
    ...
```

Or if perfect:
```
  ✓ Perfect classification! All subjects correct.
```

### 4.6 Model Training Time Summary

```
────────────────────────────────────────────────────────
MODEL TRAINING TIME SUMMARY
────────────────────────────────────────────────────────
  CNN        - Avg: 45.3s/fold, Total: 2897s (64 folds)
  XGBOOST    - Avg: 12.1s/fold, Total: 774s (64 folds)
  SVM        - Avg: 8.7s/fold, Total: 557s (64 folds)
```

---

## 5. Log File Monitoring

### Live Monitoring While Training

```bash
# Watch logs in real-time
tail -f eeg_depression_detection/outputs_v3/training_v3.log

# Or follow with timestamps
tail -f eeg_depression_detection/outputs_v3/training_v3.log | while IFS= read -r line; do echo "[$(date '+%H:%M:%S')] $line"; done

# Search for specific patterns
grep "Fold" eeg_depression_detection/outputs_v3/training_v3.log | tail -20

# Count correct vs incorrect
grep "\[✓\]" eeg_depression_detection/outputs_v3/training_v3.log | wc -l  # Correct folds
grep "\[✗\]" eeg_depression_detection/outputs_v3/training_v3.log | wc -l  # Incorrect folds
```

### Log File Location

```
eeg_depression_detection/outputs_v3/training_v3.log
```

The log file contains:
- Everything printed to console
- Debug-level details (including epoch-by-epoch CNN training)
- Full timestamps
- Complete stack traces for any errors

---

## 6. GPU Memory Monitoring

The script logs GPU memory usage at the end of each fold:

```
Fold 5/64: ... GPU=2340MB
```

**What to expect:**
- Should stay between **2.0-2.5 GB** per fold
- If memory keeps increasing above 3.0 GB → potential memory leak
- If suddenly drops to near 0 → likely an OOM crash

**Monitor in parallel:**

```bash
# In another terminal:
nvidia-smi -l 1  # Update every 1 second
```

**Healthy pattern:**
```
GPU Mem: 2.3GB → 2.2GB → 2.4GB → 2.1GB → ...  (stable)
```

**Problem pattern:**
```
GPU Mem: 2.3GB → 2.8GB → 3.5GB → CRASH  (increasing)
```

---

## 7. Running the Enhanced Script

### Basic Run

```bash
cd eeg_depression_detection
python3 train_best.py
```

### Redirect Logs (Recommended)

```bash
python3 train_best.py 2>&1 | tee run_$(date +%Y%m%d_%H%M%S).log
```

This saves logs both to:
- `outputs_v3/training_v3.log` (internal log file)
- `run_20250307_153042.log` (external duplicate with timestamp)

### With Debugging

```bash
# Enable debug logs
export LOGLEVEL=DEBUG
python3 train_best.py
```

---

## 8. Key Improvements Over Previous Version

| Feature | Before | After |
|---------|--------|-------|
| **Progress tracking** | None | tqdm progress bars for all phases |
| **Per-fold logging** | Basic info | Detailed metrics + ETA + GPU memory |
| **Intermediate steps** | Hidden | Explicit Phase 1/2/3 structure |
| **Feature extraction** | Silent | Progress bar + speed stats |
| **CNN training** | No visibility | Epoch-by-epoch loss tracking |
| **Memory monitoring** | Manual nvidia-smi | Automatic GPU memory logging |
| **Final metrics** | Scattered | Organized by section with dividers |
| **Misclassified list** | Yes/No | Clear labeling + confidence scores |
| **Model timing** | Per-fold only | Per-fold + aggregate summary |
| **Timestamps** | Log start/end | Full timestamp on completion |
| **Error recovery** | Minimal | OOM handled with fallback predictions |

---

## 9. Expected Timeline

Based on RTX 5070 (12.4 GB VRAM):

```
Phase 1: Data Loading               ~45 seconds
Phase 2: Feature Extraction         ~5-7 minutes
Phase 2: Feature Augmentation       ~20 seconds
Phase 2: LOSO CV (64 folds)         ~8-10 hours
  - Per fold: ~7-8 minutes
  - CNN training: ~3-4 min/fold
  - XGBoost: ~1-2 min/fold
  - SVM: ~30-60 sec/fold
Phase 3: Metrics & Reporting        ~10-30 seconds

Total:                              ~8-10.5 hours
```

---

## 10. Troubleshooting with Logs

### Issue: "LOSO CV seems stuck"

Check logs:
```bash
grep "Fold" eeg_depression_detection/outputs_v3/training_v3.log | tail -5
```

If no new fold logs for > 15 mins, script is likely stuck in CNN training.

### Issue: "GPU memory keeps increasing"

Check:
```bash
grep "GPU=" eeg_depression_detection/outputs_v3/training_v3.log
```

Should see pattern like: `2123MB → 2234MB → 2145MB → 2312MB`

If pattern is: `2123MB → 2523MB → 3123MB` STOP and check for memory leaks.

### Issue: "OOM crashes after fold X"

Logs will show:
```
Fold 45/64: ... Error: out of memory
Attempting recovery...
```

Check GPU memory before the crash:
```bash
grep "GPU=" eeg_depression_detection/outputs_v3/training_v3.log | tail -20
```

---

## 11. Results File

After completion, check detailed results:

```bash
cat eeg_depression_detection/outputs_v3/results_best.json
```

Contains:
- Sample-level metrics (accuracy, F1, AUC, etc.)
- Subject-level metrics
- Per-model accuracies
- Per-subject probabilities with labels
- Timestamp of completion

---

## Summary

The enhanced script provides:
✓ **Real-time progress tracking** with progress bars  
✓ **Detailed per-fold metrics** with GPU memory  
✓ **Phase-based organization** for clarity  
✓ **Automatic error recovery** for OOM  
✓ **Comprehensive final report** with multiple metrics  
✓ **Complete audit trail** in log file  
✓ **Expected timeline** estimates with ETA  

**Just run it and watch the progress!**

```bash
python3 train_best.py
```
