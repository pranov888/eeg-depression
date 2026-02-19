# EEG Depression Detection - Project Log

## Project Overview

**Objective:** Reproduce and improve upon Khaleghi et al. (2025) paper on EEG-based depression detection using deep learning with proper evaluation methodology.

**Key Contribution:** Honest LOSO (Leave-One-Subject-Out) cross-validation instead of leaky k-fold, plus multi-branch architecture combining Transformer + GNN (+ Bi-LSTM planned).

---

## Timeline

### Phase 1: Analysis & Setup (Completed)

**What we did:**
1. Analyzed the base paper "Interpretable deep learning for depression detection in neurological patients using EEG signals"
2. Identified critical flaw: Their 98% accuracy used 5-fold CV which causes data leakage (same patient's epochs in train AND test)
3. Designed improved architecture with proper LOSO evaluation

**Key insight:** Base paper's 98% accuracy is artificially inflated. With LOSO (no leakage), realistic accuracy is 85-92%.

---

### Phase 2: Implementation (Completed)

**Dataset:**
- Figshare MDD EEG Dataset
- 58 subjects (29 MDD, 29 Healthy)
- Eyes Closed (EC) condition
- 8,620 total samples (4-second epochs)

**Preprocessing Pipeline:**
```
Raw EEG → Bandpass Filter (0.5-45Hz) → Notch Filter (50Hz) →
         → Epoch Segmentation (4s, 50% overlap) → Feature Extraction
```

**Feature Extraction:**
1. **WPD (Wavelet Packet Decomposition):** 576 features per channel (19 channels × 576 = spatial features for GNN)
2. **CWT (Continuous Wavelet Transform):** 64×128 scalograms (time-frequency for Transformer)

**Architecture V1 (Completed):**
```
                    ┌─────────────────┐
   Scalograms ────► │  Transformer    │────┐
   (64×128)         │  (ViT-style)    │    │
                    └─────────────────┘    │    ┌─────────────┐
                                           ├───►│  Attention  │───► MLP ───► Output
                    ┌─────────────────┐    │    │   Fusion    │
   WPD features ──► │      GNN        │────┘    └─────────────┘
   (19×576)         │  (GAT layers)   │
                    └─────────────────┘
```

**Model Parameters:** 1,551,427

---

### Phase 3: Training V1 (Completed)

**Configuration:**
| Parameter | Value |
|-----------|-------|
| Cross-validation | LOSO (58 folds) |
| Epochs per fold | 30 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Mixed precision | FP16 |

**Training Time:** ~10 hours

**Results:**
| Metric | Value |
|--------|-------|
| **Sample-level Accuracy** | **87.27%** |
| **Subject-level Accuracy** | **91.38%** |
| **AUC-ROC** | **0.9042** |
| **F1-Score** | **0.8761** |
| **Sensitivity** | **88.68%** |
| **Specificity** | **85.83%** |
| **MCC** | **0.7456** |

**Confusion Matrix:**
```
                Predicted
                H(0)    MDD(1)
Actual H(0)     3645    602
       MDD(1)   495     3878
```

**Interpretation:**
- 53/58 subjects correctly classified
- High sensitivity (88.68%) - good at detecting depression
- Lower specificity (85.83%) - some false alarms
- Results are HONEST and clinically meaningful

---

### Phase 4: Bi-LSTM Extension (Paused)

**What was implemented:**
1. `models/branches/bilstm_encoder.py` - Bi-LSTM for temporal dynamics
2. `models/fusion/three_way_fusion.py` - 3-way attention fusion
3. `models/full_model_v2.py` - Complete V2 model
4. `scripts/train_v2.py` - Training script with checkpoint/resume support

**V2 Architecture:**
```
                    ┌─────────────────┐
   Scalograms ────► │  Transformer    │────┐
                    └─────────────────┘    │
                                           │
                    ┌─────────────────┐    │    ┌─────────────┐
   Raw EEG ───────► │    Bi-LSTM      │────┼───►│  3-Way      │───► Output
                    └─────────────────┘    │    │  Fusion     │
                                           │    └─────────────┘
                    ┌─────────────────┐    │
   WPD features ──► │      GNN        │────┘
                    └─────────────────┘
```

**V2 Model Parameters:** 2,526,812
- Transformer: 818,560
- Bi-LSTM: 729,112 (NEW)
- GNN: 390,400
- Fusion: 578,179

**Status:** Code ready, checkpointing added. Training paused.

**To resume V2 training:**
```bash
./venv/bin/python scripts/train_v2.py \
    --data_dir data/raw/figshare \
    --output_dir outputs_v2 \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --mixed_precision
```

---

## Project Structure

```
eeg_depression_detection/
├── data/
│   ├── datasets/
│   │   └── figshare_dataset.py      # Dataset loader with LOSO support
│   ├── preprocessing/
│   │   ├── filters.py               # Bandpass, notch filters
│   │   ├── wpd_extractor.py         # Wavelet Packet Decomposition
│   │   └── cwt_extractor.py         # Continuous Wavelet Transform
│   └── raw/figshare/
│       └── cache/
│           ├── figshare_EC_features.pkl      # V1 cache (661MB)
│           └── figshare_EC_features_v2.pkl   # V2 cache with raw EEG (1.3GB)
│
├── models/
│   ├── branches/
│   │   ├── transformer_encoder.py   # ViT-style encoder
│   │   ├── gnn_encoder.py           # Graph Attention Network
│   │   └── bilstm_encoder.py        # Bi-LSTM encoder (V2)
│   ├── fusion/
│   │   ├── attention_fusion.py      # 2-way fusion (V1)
│   │   └── three_way_fusion.py      # 3-way fusion (V2)
│   ├── full_model.py                # V1 model (Trans + GNN)
│   └── full_model_v2.py             # V2 model (Trans + LSTM + GNN)
│
├── scripts/
│   ├── train.py                     # V1 training script
│   └── train_v2.py                  # V2 training with checkpointing
│
├── outputs/
│   └── run_20260201_014750/
│       ├── config.json              # Training config
│       └── results.json             # V1 results (COMPLETED)
│
├── outputs_v2/                      # V2 outputs (PAUSED)
│
├── explainability/
│   └── integrated_gradients.py      # IG implementation
│
└── docs/
    ├── EXPERIMENT_RESULTS.md        # Detailed results
    └── PROJECT_LOG.md               # This file
```

---

## Comparison with Base Paper

| Aspect | Khaleghi et al. (2025) | Our Implementation |
|--------|------------------------|-------------------|
| **Accuracy** | 98.0% | 87.27% (sample) / 91.38% (subject) |
| **Validation** | 5-fold CV (LEAKY) | LOSO (NO LEAKAGE) |
| **Architecture** | CNN | Transformer + GNN |
| **Interpretability** | SHAP | Integrated Gradients (+ LRP, TCAV planned) |
| **Reproducibility** | Limited | Full code available |

**Why our "lower" accuracy is actually better:**
1. LOSO prevents data leakage - same subject never in train AND test
2. Tests generalization to completely NEW patients
3. Clinically meaningful - predicts real-world performance
4. Base paper's 98% would likely drop to ~85-90% with proper evaluation

---

### Phase 5: Explainability Implementation (Completed)

**What was implemented:**

1. **Integrated Gradients (IG)** - `explainability/integrated_gradients.py`
   - Feature attribution for scalograms and electrodes
   - Identifies which inputs drive predictions

2. **Layer-wise Relevance Propagation (LRP)** - `explainability/lrp.py`
   - Decomposes predictions back to inputs
   - Supports ε, γ, and αβ rules
   - Electrode and frequency band analysis

3. **TCAV (Concept Activation Vectors)** - `explainability/tcav.py`
   - Tests clinical concept influence
   - Implemented concepts:
     - Alpha asymmetry
     - Theta elevation
     - Beta suppression
     - Delta abnormality
     - Alpha reduction
     - Coherence reduction

4. **Unified Runner** - `explainability/run_explainability.py`
   - Runs all methods with single command
   - Outputs JSON results

**Documentation:** `docs/EXPLAINABILITY_METHODS.md`

**Explainability Results (from trained model):**

| Concept | MDD vs Healthy | Clinical Match |
|---------|----------------|----------------|
| Alpha Asymmetry | +0.12 | ✓ Right-dominant in MDD |
| Theta Elevation | +0.08 | ✓ Frontal theta ↑ in MDD |
| Delta Abnormality | +0.63 | ✓ Excessive delta in MDD |

**Key Finding:** Model uses clinically validated depression biomarkers!

---

## Project Status

### Completed
- [x] V1 Model (Transformer + GNN) - 91.38% subject accuracy
- [x] LOSO Cross-Validation - Honest evaluation
- [x] Explainability (IG, LRP, TCAV) - Clinically validated
- [x] Documentation - Full project log

### Ready to Run
- [ ] V2 Model (+ Bi-LSTM) - Code ready, checkpointing enabled

---

## Commands to Complete Project

### 1. Train V2 Model (Transformer + Bi-LSTM + GNN)
```bash
cd /home/jabe/Workspace/pra/eeg_depression_detection

# Full training (~10 hours, 58 LOSO folds)
./venv/bin/python scripts/train_v2.py \
    --data_dir data/raw/figshare \
    --output_dir outputs_v2 \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --mixed_precision

# If interrupted, resume with:
./venv/bin/python scripts/train_v2.py \
    --data_dir data/raw/figshare \
    --resume outputs_v2/run_XXXXXXXX_XXXXXX \
    --epochs 30 \
    --batch_size 16 \
    --mixed_precision
```

### 2. Run Explainability on V2 (after training)
```bash
./venv/bin/python explainability/run_explainability.py \
    --data_dir data/raw/figshare \
    --model_weights outputs_v2/run_XXXXXXXX/trained_model.pt \
    --methods ig lrp tcav
```

---

## Key Files for Results

- **V1 Results:** `outputs/run_20260201_014750/results.json`
- **V1 Config:** `outputs/run_20260201_014750/config.json`
- **Data Cache:** `data/raw/figshare/cache/figshare_EC_features.pkl`

---

## Commands Reference

**Run V1 Training (Completed):**
```bash
./venv/bin/python scripts/train.py \
    --data_dir data/raw/figshare \
    --output_dir outputs \
    --epochs 30 \
    --batch_size 16
```

**Run V2 Training (When Ready):**
```bash
./venv/bin/python scripts/train_v2.py \
    --data_dir data/raw/figshare \
    --output_dir outputs_v2 \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --mixed_precision
```

**Resume V2 from Checkpoint:**
```bash
./venv/bin/python scripts/train_v2.py \
    --data_dir data/raw/figshare \
    --resume outputs_v2/run_XXXXXXXX_XXXXXX \
    --epochs 30 \
    --batch_size 16 \
    --mixed_precision
```

---

## Notes

- V1 model achieves strong results with honest evaluation
- Bi-LSTM may add 2-5% improvement but requires ~10 more hours training
- Checkpointing now added to V2 script to prevent data loss on crashes
- All preprocessing cached - no need to reprocess raw EEG files
