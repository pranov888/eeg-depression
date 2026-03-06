# Training Script Enhancements Summary

## What Was Improved

### 1. **New Imports**
```python
from datetime import datetime  # For timestamps
from tqdm import tqdm          # For progress bars
```

---

## 2. **Enhanced Data Loading Phase**

### Before
```python
for fpath in edf_files:
    # ... basic loading ...
    log.info(f"  {fpath.name}: {len(epochs)} epochs")
```

### After
```python
for fpath in tqdm(edf_files, desc="Loading EDF files", unit="file"):
    # ... with progress bar ...
    tqdm.write(f"    {fpath.name}: {len(epochs)} epochs ({data.shape[1]/TARGET_SR:.1f}s raw data)")

# Enhanced summary with statistics
log.info(f"\n{'─'*60}")
log.info(f"Data Loading Summary:")
log.info(f"  Total subjects:        {len(all_subjects)} ({n_mdd} MDD, {n_h} Healthy)")
log.info(f"  Total epochs:          {total_epochs:,}")
log.info(f"  Avg epochs/subject:    {avg_epochs_per_subject:.0f}")
log.info(f"  Total data size:       {total_data_memory:.2f} GB")
log.info(f"  Skipped files:         {skipped}")
log.info(f"  Phase duration:        {phase_time:.1f}s")
```

**What you see:**
- Real-time progress bar showing files loaded
- Data size in GB
- Average epochs per subject
- Elapsed time for phase

---

## 3. **Improved Feature Extraction**

### Feature Extraction Progress
```python
# Added progress bars for batch extraction
def extract_features_batch(epochs: np.ndarray, desc: str = "Extracting features"):
    features = []
    for i in tqdm(range(len(epochs)), desc=desc, unit="epoch", leave=False):
        feat = extract_all_features(epochs[i])
        features.append(feat)
    return np.array(features, dtype=np.float32)
```

### Better Logging
```python
log.info("\n[Step 1/5] Extracting handcrafted features for all subjects...")

for idx, sid in enumerate(tqdm(subject_ids, desc="Feature extraction by subject")):
    feats = extract_features_batch(epochs, desc=f"  {sid}")
    log.debug(f"  [{idx+1}/{n_subjects}] {sid}: {feats.shape[0]} epochs, {feats.nbytes/1e6:.1f}MB")

log.info(f"✓ Feature extraction complete in {feat_time:.1f}s ({total_feature_epochs:,} epochs processed)")
log.info(f"  Average {total_feature_epochs/feat_time:.0f} epochs/sec")
```

**What you see:**
- Progress bar: which subject is being processed
- Nested progress bar: epoch count within each subject
- File size in MB
- Overall throughput (epochs/sec)

---

## 4. **Enhanced CNN Training Visibility**

### Added Fold Context
```python
def train_cnn(..., fold_idx=None):  # Added fold parameter
    fold_str = f" (Fold {fold_idx})" if fold_idx is not None else ""
    log.debug(f"  CNN Training{fold_str}: {len(train_epochs)} train epochs...")
```

### Epoch-by-Epoch Logging
```python
# Track losses
train_losses = []
val_losses = []

for epoch in range(CNN_EPOCHS):
    # ... training code ...
    
    # Log each epoch
    lr = optimizer.param_groups[0]["lr"]
    log.debug(f"    Epoch {epoch+1:2d}/{CNN_EPOCHS}: Loss={train_loss:.4f} | "
              f"ValLoss={val_loss:.4f} | LR={lr:.5f}")
    
    # Early stopping message
    if patience_counter >= CNN_PATIENCE:
        log.debug(f"    Early stopping at epoch {epoch+1} (patience={CNN_PATIENCE})")
        break

# Final summary
log.debug(f"  CNN Training Complete: final_train_loss={final_train_loss:.4f}, "
          f"final_val_loss={final_val_loss:.4f}, epochs={len(train_losses)}")
```

**What you see:**
- Each epoch's training and validation loss
- Current learning rate
- Early stopping message with epoch count

---

## 5. **LOSO Cross-Validation Phase Organization**

### Clear Phase Structure
```python
log.info("\n" + "=" * 60)
log.info("PHASE 2: LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION")
log.info("=" * 60)
```

### 5-Step Process Logging
```python
log.info(f"\n[Step 1/5] Extracting handcrafted features...")
log.info(f"\n[Step 2/5] Augmenting features...")
log.info(f"\n[Step 3/5] Running {n_subjects}-fold LOSO CV...")
# Results computed in steps 4 & 5
```

### Progress Bar for Folds
```python
for fold_idx, test_sid in enumerate(tqdm(subject_ids, desc="LOSO CV Progress", unit="fold")):
    # ... per-fold code ...
```

---

## 6. **Per-Fold Detailed Logging**

### Before
```
Fold  5/64: MDD_S12    True=MDD Pred=MDD prob=0.847 (CNN=0.812 XGB=0.845 SVM=0.883) [CORRECT] 87s (ETA: 45min) GPU: 2340MB
```

### After (Enhanced)
```
Fold  5/64: MDD_S12    True=MDD Pred=MDD prob=0.847 (87s, ETA=45m) [✓] GPU=2340MB

# With debug logging also shows:
Model probs - CNN=0.812, XGB=0.845, SVM=0.883
Post-cleanup GPU memory: 2234MB
```

**Improvements:**
- Cleaner main output (essential info only)
- Better formatting with ✓/✗ for correct/incorrect
- GPU memory always shown
- Additional details in debug logs

### Per-Model Training Times
```python
model_times = {"cnn": [], "xgb": [], "svm": []}

# During fold:
cnn_time = time.time() - cnn_t0
model_times["cnn"].append(cnn_time)
log.debug(f"    CNN: {cnn_time:.1f}s")
```

---

## 7. **LOSO CV Loop Summary**

### Completion Message
```python
log.info("=" * 60)
log.info(f"✓ LOSO CV Complete in {time.time() - total_t0:.1f}s")
log.info(f"  Avg fold time: {np.mean(fold_times):.1f}s")
log.info(f"  Total folds: {len(fold_times)}")
```

---

## 8. **Enhanced Metrics & Reporting Phase**

### Clear Phase Header
```python
log.info("\n" + "=" * 60)
log.info("PHASE 3: METRICS & REPORTING")
log.info("=" * 60)
```

### Organized Metrics Display

**Per-Model Performance:**
```python
log.info("\n[Per-Model Performance]")
for model_name, times in model_times.items():
    # Shows: CNN, XGBoost, SVM accuracy per subject
    log.info(f"  {model_name:10s} - Subject Accuracy: {model_subj_acc:.4f}")
```

**Sample-Level Metrics Section:**
```python
log.info("\n" + "─" * 60)
log.info("SAMPLE-LEVEL METRICS (Each epoch)")
log.info("─" * 60)
log.info(f"  Accuracy:    {sample_acc:.4f}")
log.info(f"  F1 Score:    {sample_f1:.4f}")
log.info(f"  AUC-ROC:     {sample_auc:.4f}")
# ... more metrics with formatted output
```

**Subject-Level Metrics Section:**
```python
log.info("\n" + "─" * 60)
log.info("SUBJECT-LEVEL METRICS (Averaged across epochs)")
log.info("─" * 60)
# Same metrics as sample-level but for subjects
```

**Misclassified Subjects:**
```python
if misclassified:
    log.info(f"\n  ⚠️  Misclassified subjects ({len(misclassified)}):")
    for sid in misclassified:
        log.info(f"    {sid}: True={true_l:8s} → Pred={pred_l:8s} (prob={pred_p:.3f})")
else:
    log.info(f"\n  ✓ Perfect classification! All subjects correct.")
```

**Model Timing Summary:**
```python
log.info("\n" + "─" * 60)
log.info("MODEL TRAINING TIME SUMMARY")
log.info("─" * 60)
for model_name, times in model_times.items():
    avg_time = np.mean(times)
    total_time = np.sum(times)
    log.info(f"  {model_name.upper():10s} - Avg: {avg_time:.1f}s/fold, Total: {total_time:.0f}s")
```

---

## 9. **Enhanced Results JSON**

### Added Metadata
```python
results = {
    "timestamp": datetime.now().isoformat(),  # ISO format timestamp
    # ... all previous fields ...
    "subject_level": {
        # ... metrics ...
        "misclassified_count": len(misclassified),
    }
}
```

### Better Organization
```json
{
  "timestamp": "2025-03-07T15:30:42.123456",
  "sample_level": {
    "accuracy": 0.7845,
    "n_epochs": 36247,
    ...
  },
  "subject_level": {
    "accuracy": 0.8438,
    "n_subjects": 64,
    "n_correct": 54,
    "misclassified_count": 10,
    ...
  }
}
```

---

## 10. **Main Function Improvements**

### Before
```python
def main():
    log.info("..." )
    # ... training ...
    elapsed = time.time() - t_start
    log.info(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    log.info("Done.")
```

### After
```python
def main():
    log.info("EEG DEPRESSION DETECTION — V3 ENSEMBLE TRAINING")
    log.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # ... training ...
    
    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("=" * 60)
    log.info(f"✓ Total runtime: {hours:.1f}h ({minutes:.1f}m / {elapsed:.0f}s)")
    log.info(f"✓ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"✓ Results: {OUTPUT_DIR / 'results_best.json'}")
    log.info(f"✓ Logs:    {OUTPUT_DIR / 'training_v3.log'}")
```

**Improvements:**
- Time shown in hours:minutes:seconds format
- Both start and end timestamps
- File paths for results and logs
- Visual markers (✓) for completion status

---

## 11. **Logging Configuration**

### Existing (Unchanged)
```python
logging.basicConfig(
    level=logging.INFO,  # Shows INFO and above (INFO, WARNING, ERROR)
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),        # Print to console
        logging.FileHandler(log_path, mode="w"),  # Write to file
    ],
)
```

### Debug Logs
Debug messages are written to file but not shown on console by default:

```python
log.debug(f"  CNN Training (Fold {fold_idx}): ...")  # File only
log.info(f"  Fold {fold_idx + 1}: ...")               # Console + File
```

To see debug logs in console:
```python
logging.getLogger().setLevel(logging.DEBUG)
```

---

## 12. **Key Statistics Tracked**

**Automatically calculated and logged:**
- Total epochs processed
- Epochs per second
- GPU memory usage (MB)
- Fold execution time
- Estimated time remaining (ETA)
- Per-subject accuracy
- Per-model accuracy
- Model training times (CNN, XGBoost, SVM)
- Misclassification rate
- Sensitivity, specificity, AUC-ROC per fold

---

## Summary of Enhancements

| Component | Enhancement |
|-----------|-------------|
| **Imports** | Added `datetime`, `tqdm` |
| **Data Loading** | Progress bars + summary statistics |
| **Feature Extraction** | Nested progress bars + throughput stats |
| **CNN Training** | Epoch-by-epoch loss tracking + early stopping messages |
| **LOSO CV** | 5-step structure + per-fold metrics + ETA |
| **Per-Fold Output** | Cleaner format with GPU memory + model times |
| **Metrics** | Organized by section with visual dividers |
| **Results JSON** | Added timestamp + misclassified count |
| **Completion** | Enhanced summary with file paths + timing |

All enhancements maintain backwards compatibility—existing functionality is preserved while adding visibility!
