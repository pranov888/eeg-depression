# Quick Start Guide - Fixed train_best.py

## Prerequisites

### 1. PyTorch with RTX 5070 Support
```bash
# If you haven't already fixed CUDA compatibility:
pip uninstall -y torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 2. Verify GPU Works
```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
GPU: NVIDIA GeForce RTX 5070
```

---

## Running the Training

### Option 1: Run Directly (Foreground)
```bash
cd /path/to/eeg-depression/eeg_depression_detection
python train_best.py
```

### Option 2: Run in Background (Recommended)
```bash
nohup python train_best.py > train_best.log 2>&1 &
# To monitor:
tail -f train_best.log
```

### Option 3: Run with tmux (Best for Stability)
```bash
tmux new-session -d -s training "python train_best.py"
tmux attach -t training
# Press Ctrl+B D to detach, Ctrl+C to stop
```

---

## What to Expect

### Timeline
- **0-2 hours**: Feature extraction (912 features per epoch × 36k epochs)
- **2-3 hours**: First few LOSO folds (slower due to warmup)
- **3-10 hours**: Remaining 60 folds (faster, once GPU cache is optimized)
- **10+ hours**: Results saved to `outputs_v3/results_best.json`

### Log Output Progression

**Phase 1: Data Loading**
```
[INFO] Found 181 EDF files
[INFO] Loading H S1 EC.edf: 149 epochs
...
[INFO] Loaded 64 subjects: 34 MDD, 30 Healthy
```

**Phase 2: Feature Extraction** (~2 hours)
```
[INFO] Extracting handcrafted features for all subjects...
[INFO]   [1/64] H_S1: 624 epochs, 912 features
[INFO]   [2/64] H_S10: 637 epochs, 912 features
... (continues for all 64 subjects)
[INFO] Feature extraction done in 7564.3s. Feature dim: 912
```

**Phase 3: Feature Augmentation** (< 1 second)
```
[INFO] Augmenting features in feature-space (fast)...
[INFO] Feature augmentation done in 0.7s
```

**Phase 4: LOSO Cross-Validation** (~5-8 hours)
```
[INFO] Starting 64-fold LOSO CV...
[INFO] Device: cuda
[INFO] GPU: NVIDIA GeForce RTX 5070
[INFO] VRAM: 12.4 GB
[INFO]   Fold  1/64: H_S1        True=H   Pred=H   prob=0.823 (CNN=0.81 XGB=0.84 SVM=0.82) [CORRECT] 46s (ETA: 48min) GPU: 2340MB
[INFO]   Fold  2/64: H_S10       True=H   Pred=H   prob=0.756 (CNN=0.74 XGB=0.77 SVM=0.75) [CORRECT] 48s (ETA: 50min) GPU: 2340MB
...
[INFO]   Memory cleaned. Current GPU: 128MB
[INFO]   Fold  3/64: ...
```

**Phase 5: Results** (< 1 minute)
```
[INFO] ============================================================
[INFO] FINAL RESULTS
[INFO] ============================================================
[INFO] --- Sample-Level Metrics ---
[INFO]   Accuracy:    0.8234
[INFO]   F1 Score:    0.8156
[INFO]   AUC-ROC:     0.8812
...
[INFO] --- Subject-Level Metrics ---
[INFO]   Accuracy:    0.6875 (44/64 subjects)
...
[INFO] Results saved to /path/to/outputs_v3/results_best.json
```

---

##  Troubleshooting

### Problem: Still Getting OOM
**Solution 1: Reduce CNN Batch Size**
```python
# In train_best.py, change line 74:
CNN_BATCH = 16  # Instead of 32
```

**Solution 2: Reduce Training Data**
```python
# In train_best.py, comment out augmentation:
# don't include augmented_feats in training
```

**Solution 3: Run Fewer Folds**
```python
# In train_best.py, after line 630, add:
if fold_idx >= 10:  # Test with just 10 folds
    break
```

### Problem: GPU Not Being Used
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

If `False`, your PyTorch installation doesn't have CUDA support for RTX 5070.

### Problem: Process Killed After Hours of Training
Check your system logs:
```bash
dmesg | tail -20  # Check for OOM killer
free -h            # Check available RAM
nvidia-smi         # Check GPU state
```

---

## Output Files

After successful completion:

```
outputs_v3/
├── training_v3.log          # Full execution log
└── results_best.json        # Final metrics and predictions
```

### Results JSON Structure
```json
{
  "sample_level": {
    "accuracy": 0.8234,
    "f1": 0.8156,
    "auc_roc": 0.8812,
    "precision": 0.8345,
    "recall": 0.8023,
    "sensitivity": 0.8023,
    "specificity": 0.8421,
    "confusion_matrix": {"TP": 5823, "TN": 6201, "FP": 1158, "FN": 1093}
  },
  "subject_level": {
    "accuracy": 0.6875,
    "f1": 0.6816,
    "auc_roc": 0.7234,
    "n_subjects": 64,
    "n_correct": 44,
    "misclassified": ["MDD_S5", "H_S12", ...]
  },
  "per_subject_probabilities": {
    "H_S1": {"true_label": 0, "pred_prob": 0.823},
    "MDD_S1": {"true_label": 1, "pred_prob": 0.912},
    ...
  }
}
```

---

## Performance Notes

- **GPU Memory**: Stable at ~2.3-2.5 GB per fold (down from OOM)
- **CPU Usage**: ~50-60% during model training
- **RAM**: ~8-10 GB (feature arrays)
- **Type**: Multi-model ensemble significantly more robust than single model
- **Validation**: Rigorous LOSO CV prevents overfitting to subjects

---

## Success Criteria

✅ Training completes without crashes  
✅ All 64 LOSO folds executed successfully  
✅ GPU memory cleans up between folds  
✅ Results saved with both sample and subject-level metrics  
✅ Final subject-level accuracy > 60% (clinically significant)

---

Good luck! 🧠✨
