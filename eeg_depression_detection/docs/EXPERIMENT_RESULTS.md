# Experiment Results Documentation

## Experiment 1: Transformer + GNN (Completed)

### Date: 2026-02-01
### Duration: ~10 hours (preprocessing + training)

---

## 1. Architecture Used

```
EEG Signals → Preprocessing → WPD+CWT → Transformer + GNN → Attention Fusion → Classification
```

### Components:
- **Transformer Branch:** Vision Transformer processing CWT scalograms (64×128)
- **GNN Branch:** Graph Attention Network processing WPD features (19 nodes × 576 features)
- **Fusion:** Cross-attention with adaptive gating
- **Classifier:** MLP (128→64→32→1)

### Model Parameters: 1,551,427

---

## 2. Dataset

| Metric | Value |
|--------|-------|
| Dataset | Figshare MDD EEG |
| Condition | Eyes Closed (EC) |
| Total Subjects | 58 |
| Total Samples | 8,620 |
| Class 0 (Healthy) | 4,247 (49.3%) |
| Class 1 (MDD) | 4,373 (50.7%) |

---

## 3. Training Configuration

| Parameter | Value |
|-----------|-------|
| Cross-validation | LOSO (58 folds) |
| Epochs per fold | 30 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Mixed precision | FP16 |
| Gradient accumulation | 2 |

---

## 4. Results

### 4.1 Final Metrics (Aggregated across all LOSO folds)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Sample-level Accuracy** | **87.27%** | Correct predictions per epoch |
| **Subject-level Accuracy** | **91.38%** | Correct diagnosis per patient |
| **AUC-ROC** | **0.9042** | Excellent discrimination |
| **F1-Score** | **0.8761** | Good precision-recall balance |
| **Sensitivity (Recall)** | **88.68%** | Depression detection rate |
| **Specificity** | **85.83%** | Healthy detection rate |
| **MCC** | **0.7456** | Strong correlation |

### 4.2 Confusion Matrix

```
                    Predicted
                    Healthy(0)  MDD(1)
Actual  Healthy(0)    3645       602
        MDD(1)         495      3878
```

- **True Negatives (TN):** 3,645 - Correctly identified healthy
- **False Positives (FP):** 602 - Healthy misclassified as MDD
- **False Negatives (FN):** 495 - MDD misclassified as healthy
- **True Positives (TP):** 3,878 - Correctly identified MDD

### 4.3 Clinical Interpretation

- **High Sensitivity (88.68%):** Good at detecting depression (few missed cases)
- **Good Specificity (85.83%):** Acceptable false alarm rate
- **Subject-level 91.38%:** 53/58 subjects correctly classified
- **5 subjects misclassified:** Borderline cases or atypical presentations

---

## 5. Comparison with Base Paper

| Metric | Khaleghi et al. (2025) | Our Results | Notes |
|--------|------------------------|-------------|-------|
| Accuracy | 98.0% | 87.27% | Ours is HONEST (LOSO) |
| Validation | 5-fold (leaky) | LOSO (no leakage) | Critical difference |
| F1-Score | 0.98 | 0.876 | |
| Architecture | CNN | Transformer+GNN | More sophisticated |
| Interpretability | SHAP only | IG + (LRP, TCAV pending) | Multi-level |

### Why Our Accuracy is "Lower" but Better:

1. **LOSO prevents data leakage:** Same subject never in train AND test
2. **Realistic clinical scenario:** Tests generalization to NEW patients
3. **Base paper's 98% is likely inflated:** K-fold allows same patient's epochs in both splits
4. **Our 91.38% subject-level accuracy is clinically meaningful**

---

## 6. Output Files

```
outputs/run_20260201_014750/
├── config.json          # Training configuration
├── results.json         # Detailed results and predictions
└── checkpoints/         # Model weights per fold
```

---

## 7. Next Steps: Add Bi-LSTM Branch

### Rationale:
Current architecture captures:
- **Transformer:** Time-frequency patterns (scalograms)
- **GNN:** Spatial electrode relationships (graph)

Missing:
- **Temporal dynamics:** Sequential patterns in raw EEG

### Expected Improvement:
- Bi-LSTM will capture temporal dependencies
- 3-way fusion may improve accuracy by 2-5%
- More comprehensive feature representation

---

## Experiment 2: Transformer + Bi-LSTM + GNN (PAUSED)

### Architecture:
```
                    ┌─────────────────┐
   Scalograms ────► │  Transformer    │────┐
                    └─────────────────┘    │
                                           │
                    ┌─────────────────┐    │    ┌─────────────┐
   Raw EEG ───────► │    Bi-LSTM      │────┼───►│  3-Way      │───► Classification
   (temporal)       └─────────────────┘    │    │  Fusion     │
                                           │    └─────────────┘
                    ┌─────────────────┐    │
   WPD features ──► │      GNN        │────┘
   (spatial graph)  └─────────────────┘
```

### Model Parameters: 2,526,812
| Component | Parameters |
|-----------|------------|
| Transformer | 818,560 |
| Bi-LSTM | 729,112 |
| GNN | 390,400 |
| Fusion | 578,179 |

### Status: PAUSED
- All code implemented and tested
- Checkpointing added to training script
- Data preprocessed and cached (1.3GB)
- Ready to resume when needed

### To Resume:
```bash
./venv/bin/python scripts/train_v2.py \
    --data_dir data/raw/figshare \
    --output_dir outputs_v2 \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --mixed_precision
```
