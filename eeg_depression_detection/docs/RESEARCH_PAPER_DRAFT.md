# Advanced EEG-Based Depression Detection Using Transformer-GNN Fusion with Multi-Level Explainability

## A Comprehensive Research Documentation

**Project Status:** In Progress
**Last Updated:** 2026-01-31
**Training Status:** LOSO Cross-Validation Running

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Related Work - Base Paper Analysis](#3-related-work---base-paper-analysis)
4. [Proposed Methodology](#4-proposed-methodology)
5. [System Architecture](#5-system-architecture)
6. [Implementation Details](#6-implementation-details)
7. [Dataset](#7-dataset)
8. [Experimental Setup](#8-experimental-setup)
9. [Results](#9-results)
10. [Explainability Analysis](#10-explainability-analysis)
11. [Discussion](#11-discussion)
12. [Conclusion](#12-conclusion)
13. [References](#13-references)

---

## 1. Abstract

Depression affects over 280 million people worldwide, with neurological patients facing elevated risks due to medication side effects and disease burden. This research proposes an advanced interpretable deep learning framework for EEG-based depression detection that significantly extends existing approaches.

**Key Contributions:**
1. **Dual-Branch Architecture:** Combines a Vision Transformer for time-frequency analysis with a Graph Attention Network for spatial electrode relationships
2. **Advanced Feature Extraction:** Multi-wavelet Wavelet Packet Decomposition (WPD) using three wavelet families (db4, sym5, coif3) combined with Continuous Wavelet Transform (CWT) scalograms
3. **Attention-Based Fusion:** Cross-attention mechanism with adaptive gating to optimally combine complementary representations
4. **Multi-Level Explainability:** Integrated Gradients, Layer-wise Relevance Propagation (LRP), and Testing with Concept Activation Vectors (TCAV) for both feature-level and concept-level interpretability
5. **Rigorous Evaluation:** Leave-One-Subject-Out (LOSO) cross-validation preventing data leakage

**Target Performance:** Surpass baseline CNN-SHAP framework (98% accuracy*) with honest LOSO evaluation and enhanced interpretability.

*Note: Baseline's 98% likely inflated due to k-fold without subject-wise splitting.

---

## 2. Introduction

### 2.1 Problem Statement

Depression is a leading cause of disability worldwide, affecting approximately 280 million individuals globally. In neurological patient populations, depression prevalence is significantly elevated due to:

- **Neurobiological factors:** Direct impact of neurological disorders on mood-regulating brain circuits
- **Medication effects:** Antiepileptic drugs like levetiracetam cause depression in up to 22% of patients
- **Psychosocial burden:** Chronic illness management and reduced quality of life

Traditional depression diagnosis relies on subjective clinical interviews and self-report questionnaires, which suffer from:
- Inter-rater variability
- Patient recall bias
- Symptom overlap with neurological conditions
- Lack of objective biomarkers

### 2.2 EEG as an Objective Biomarker

Electroencephalography (EEG) offers a promising modality for objective depression detection:
- **Non-invasive:** No radiation or contrast agents
- **High temporal resolution:** Millisecond-level neural activity capture
- **Cost-effective:** Widely available in clinical settings
- **Established biomarkers:** Documented EEG alterations in depression:
  - Elevated theta power (4-8 Hz)
  - Reduced alpha power (8-13 Hz)
  - Increased delta/alpha ratio
  - Frontal alpha asymmetry
  - Reduced signal complexity (Hjorth parameters)

### 2.3 Limitations of Existing Approaches

Current deep learning approaches for EEG-based depression detection face several limitations:

| Limitation | Impact |
|------------|--------|
| Simple CNN architectures | Cannot capture long-range temporal dependencies |
| Flat feature vectors | Ignore spatial electrode relationships |
| Single wavelet family | Miss frequency-specific patterns |
| SHAP-only interpretability | Limited to feature-level explanations |
| K-fold cross-validation | Data leakage from same-subject samples in train/test |

### 2.4 Our Contributions

This research addresses these limitations through:

1. **Transformer Encoder:** Captures global time-frequency dependencies in CWT scalograms using self-attention
2. **Graph Neural Network:** Models inter-electrode connectivity based on both spatial proximity and functional correlation
3. **Multi-Wavelet WPD:** Extracts complementary features using db4 (sharp transients), sym5 (smooth oscillations), and coif3 (trend components)
4. **Attention Fusion:** Learns optimal combination of temporal and spatial representations
5. **Multi-Level XAI:** Provides feature-level (IG), layer-level (LRP), and concept-level (TCAV) explanations
6. **LOSO Validation:** Ensures no data leakage with subject-wise train/test splitting

---

## 3. Related Work - Base Paper Analysis

### 3.1 Base Paper Summary

**Title:** "Interpretable deep learning for depression detection in neurological patients using EEG signals"
**Authors:** Khaleghi et al.
**Published:** MethodsX, 2025
**DOI:** 10.1016/j.mex.2025.103736

#### 3.1.1 Methodology

The base paper implements:

```
EEG Features → PCA → t-SNE → CNN → SHAP → Depression Classification
```

**Dataset:**
- Kaggle Depression-Rest EEG dataset
- 232 patients (116 depressed, 116 healthy)
- Pre-extracted features (31 features × 6 brain regions)

**Feature Extraction:**
- Statistical: Mobility, Complexity, Mean, STD, Skewness, Kurtosis
- Frequency: Delta/Alpha ratio, FFT band powers
- Time-frequency: Wavelet entropy (single wavelet)

**Model Architecture:**
```
Input (21 PCA components)
    → Conv1D (64 filters, kernel=3) → BatchNorm → MaxPool
    → Conv1D (128 filters, kernel=3) → BatchNorm → MaxPool
    → Flatten → Dense (128) → Dropout (0.5)
    → Dense (1, sigmoid)
```

**Training:**
- 5-fold stratified cross-validation
- Adam optimizer (lr=0.0001)
- Binary cross-entropy loss
- Early stopping (patience=10)

**Results:**
- Accuracy: 98%
- Precision: 0.97-0.98
- Recall: 0.97-0.98
- F1-Score: 0.98

**Interpretability:**
- Kernel SHAP analysis
- Top features: Delta/Alpha ratio, FFT Theta Max Power

### 3.2 Critical Analysis of Base Paper

#### 3.2.1 Strengths
1. High reported accuracy (98%)
2. Integration of SHAP for interpretability
3. Identification of clinically meaningful biomarkers
4. Lightweight model suitable for deployment

#### 3.2.2 Limitations and Concerns

| Issue | Description | Impact |
|-------|-------------|--------|
| **Data Leakage Risk** | 5-fold CV without subject-wise splitting | Samples from same subject may appear in train and test, inflating accuracy |
| **Pre-extracted Features** | Uses pre-computed features from Kaggle | Limited control over feature quality and preprocessing |
| **Simple Architecture** | 2-layer CNN | Cannot capture complex temporal dependencies |
| **No Spatial Modeling** | Treats electrodes as flat features | Misses inter-electrode connectivity patterns |
| **Single Wavelet** | Only db4 wavelet | Misses frequency-specific patterns captured by other wavelets |
| **Limited XAI** | SHAP only | No concept-level or layer-wise explanations |
| **Moderate Confidence** | Prediction confidence 0.41-0.59 | Suggests potential overfitting or calibration issues |

#### 3.2.3 Reproducibility Concerns

The reported 98% accuracy with 5-fold CV is likely inflated because:
1. Same patient's epochs may appear in both train and test
2. Model learns patient-specific artifacts rather than depression patterns
3. Real-world performance on new patients would be significantly lower

**Evidence:** Their own confidence scores (0.41-0.59) suggest the model is not well-calibrated despite high accuracy.

### 3.3 Comparison: Base Paper vs. Our Approach

| Aspect | Base Paper (Khaleghi et al.) | Our Approach |
|--------|------------------------------|--------------|
| **Dataset** | Kaggle pre-extracted features | Figshare raw EDF files |
| **Subjects** | 232 patients | 64 subjects (34 MDD, 30 HC) |
| **Preprocessing** | Pre-done | Full pipeline (filtering, ICA, resampling) |
| **Feature Extraction** | 31 features, single wavelet | 576 features/channel, 3 wavelets + CWT |
| **Architecture** | 2-layer CNN | Transformer + GNN + Attention Fusion |
| **Parameters** | ~50K | ~1.5M |
| **Spatial Modeling** | None | Graph Attention Network |
| **Temporal Modeling** | Conv1D | Vision Transformer |
| **Cross-Validation** | 5-fold (leaky) | LOSO (subject-wise) |
| **Interpretability** | SHAP | IG + LRP + TCAV |
| **Clinical Concepts** | Feature importance only | Concept-level explanations |

---

## 4. Proposed Methodology

### 4.1 Overall Pipeline

```
Raw EEG (19 channels × 256 Hz × 5 min)
         │
         ▼
┌─────────────────────────────────────┐
│        PREPROCESSING                │
│  • Bandpass filter (1-45 Hz)        │
│  • Notch filter (50/60 Hz)          │
│  • ICA artifact removal             │
│  • Resample to 250 Hz               │
│  • Epoch segmentation (4s, 50% OL)  │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     FEATURE EXTRACTION              │
│  ┌─────────────┬─────────────┐      │
│  │     WPD     │     CWT     │      │
│  │ (3 wavelets)│ (scalograms)│      │
│  │ 19×576 feat │  64×128 img │      │
│  └─────────────┴─────────────┘      │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      DEEP LEARNING MODEL            │
│  ┌─────────────┬─────────────┐      │
│  │ Transformer │     GNN     │      │
│  │  (temporal) │  (spatial)  │      │
│  └──────┬──────┴──────┬──────┘      │
│         │             │             │
│         ▼             ▼             │
│  ┌─────────────────────────┐        │
│  │  Attention-Based Fusion │        │
│  │   (cross-attn + gating) │        │
│  └─────────────────────────┘        │
│                │                    │
│                ▼                    │
│  ┌─────────────────────────┐        │
│  │   Classification Head   │        │
│  │      (MLP + Sigmoid)    │        │
│  └─────────────────────────┘        │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      EXPLAINABILITY                 │
│  • Integrated Gradients (feature)   │
│  • LRP (layer-wise)                 │
│  • TCAV (concept-level)             │
└─────────────────────────────────────┘
         │
         ▼
    Depression Prediction
    + Clinical Interpretation
```

### 4.2 Preprocessing Pipeline

#### 4.2.1 Signal Filtering

```python
# Bandpass filter: 1-45 Hz (FIR, order 101)
# - Removes DC drift (<1 Hz)
# - Removes high-frequency noise (>45 Hz)
# - Preserves all relevant EEG bands

# Notch filter: 50 Hz and 60 Hz
# - Removes power line interference
# - IIR notch filter (Q=30)
```

#### 4.2.2 Artifact Removal

- **ICA (Independent Component Analysis):** Picard algorithm
- **Automatic component rejection:** Based on EOG/ECG correlation
- **Epoch rejection:** Amplitude threshold >100 μV

#### 4.2.3 Epoch Segmentation

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epoch length | 4 seconds | Captures multiple oscillation cycles |
| Overlap | 50% | Data augmentation, smooth transitions |
| Samples per epoch | 1000 (at 250 Hz) | Sufficient for wavelet analysis |
| Epochs per subject | ~150 | 5 min recording with overlap |

### 4.3 Feature Extraction

#### 4.3.1 Wavelet Packet Decomposition (WPD)

**Multi-Wavelet Approach:**

| Wavelet | Properties | Captures |
|---------|------------|----------|
| **db4** (Daubechies-4) | Compact support, 4 vanishing moments | Sharp transients, spikes |
| **sym5** (Symlet-5) | Near-symmetric, smooth | Smooth oscillations, alpha/beta |
| **coif3** (Coiflet-3) | Symmetric, good frequency localization | Trend components, slow waves |

**Decomposition Parameters:**
- Level: 5 (creates 32 terminal nodes)
- Frequency resolution: sr/(2^level) = 250/32 ≈ 7.8 Hz per band

**Features per Node (6 features):**
1. **Energy:** Total power in frequency band
2. **Shannon Entropy:** Signal complexity
3. **Log Energy Entropy:** Alternative complexity measure
4. **Mean:** DC component
5. **Standard Deviation:** Variability
6. **Skewness:** Distribution asymmetry

**Total WPD Features:**
- 3 wavelets × 32 nodes × 6 features = 576 features per channel
- 19 channels × 576 = 10,944 features per epoch

#### 4.3.2 Continuous Wavelet Transform (CWT)

**Scalogram Generation:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Mother wavelet | Complex Morlet (cmor1.5-1.0) | Good time-frequency localization |
| Frequency range | 1-45 Hz | All relevant EEG bands |
| Number of scales | 64 | Sufficient frequency resolution |
| Output size | 64×128 (freq×time) | Compatible with ViT patch size |

**Processing:**
1. Compute CWT for each channel
2. Take magnitude (complex wavelet)
3. Average across channels → single scalogram
4. Normalize (min-max) to [0, 1]

### 4.4 Deep Learning Architecture

#### 4.4.1 Transformer Encoder (Time-Frequency Branch)

**Purpose:** Capture long-range time-frequency dependencies in scalograms

**Architecture:**
```
Input: Scalogram (1, 64, 128)
    │
    ▼
Patch Embedding
    • Patch size: 8×16
    • Number of patches: 8×8 = 64
    • Embedding dim: 128
    │
    ▼
Positional Encoding (learnable)
    │
    ▼
Transformer Encoder (×4 layers)
    • Multi-head self-attention (4 heads)
    • Feed-forward network (512 dim)
    • Layer normalization
    • Dropout (0.1)
    │
    ▼
CLS Token → Global Representation (128-dim)
```

**Design Decisions:**
- **Patch-based:** Treats scalogram like an image (Vision Transformer style)
- **CLS token:** Aggregates global information
- **4 layers:** Balance between capacity and overfitting
- **128-dim:** Matches GNN output for fusion

#### 4.4.2 Graph Attention Network (Spatial Branch)

**Purpose:** Model inter-electrode relationships and brain connectivity

**Graph Construction:**
```
Nodes: 19 electrodes (10-20 montage)
Node features: 576-dim WPD features

Edges (Hybrid):
1. Spatial: k-nearest neighbors (k=6) based on 3D electrode positions
2. Functional: Pearson correlation > 0.5 between electrode signals
3. Combined: Union of spatial and functional edges
```

**Architecture:**
```
Input: Node features (19, 576) + Edge index
    │
    ▼
GAT Layer 1
    • Hidden dim: 128
    • Heads: 4
    • Dropout: 0.3
    │
    ▼
GAT Layer 2
    │
    ▼
GAT Layer 3
    │
    ▼
Global Mean Pooling → Graph Representation (128-dim)
```

**Design Decisions:**
- **Graph Attention:** Learns edge importance (which connections matter)
- **Hybrid edges:** Combines anatomical and functional connectivity
- **Mean pooling:** Permutation invariant aggregation

#### 4.4.3 Attention-Based Fusion

**Purpose:** Optimally combine Transformer and GNN representations

**Architecture:**
```
Transformer output (128-dim)  ──┐
                                ├──► Cross-Attention
GNN output (128-dim)  ──────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Cross-Attention Module             │
│  • Trans attends to GNN             │
│  • GNN attends to Trans             │
│  • 4 attention heads                │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Gating Mechanism                   │
│  • Learned gates (sigmoid)          │
│  • gate_t × trans + gate_g × gnn    │
│  • Adaptive weighting               │
└─────────────────────────────────────┘
    │
    ▼
Fused Representation (128-dim)
```

**Design Decisions:**
- **Cross-attention:** Allows information exchange between branches
- **Gating:** Adaptive weighting based on sample characteristics
- **Not just concatenation:** More expressive fusion

#### 4.4.4 Classification Head

```
Fused features (128-dim)
    │
    ▼
Linear (128 → 64) + BatchNorm + ReLU + Dropout(0.5)
    │
    ▼
Linear (64 → 32) + BatchNorm + ReLU + Dropout(0.3)
    │
    ▼
Linear (32 → 1) + Sigmoid
    │
    ▼
Depression Probability [0, 1]
```

### 4.5 Training Strategy

#### 4.5.1 Leave-One-Subject-Out (LOSO) Cross-Validation

**Rationale:** Prevents data leakage from same-subject epochs in train/test

**Procedure:**
```
For each subject s in {S1, S2, ..., S58}:
    Train set: All epochs from subjects ≠ s
    Test set: All epochs from subject s

    Train model → Collect predictions for s

Aggregate all predictions → Compute final metrics
```

**Advantages over K-Fold:**
1. No data leakage (subject-wise split)
2. Realistic clinical scenario (new patient)
3. Honest performance estimate
4. Tests generalization to unseen subjects

#### 4.5.2 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Decoupled weight decay, good for Transformers |
| Learning rate | 1e-4 | Conservative for stable training |
| Weight decay | 0.01 | Regularization |
| Batch size | 16 | Memory constraint |
| Gradient accumulation | 2 | Effective batch = 32 |
| Epochs per fold | 30 | Sufficient for convergence |
| Mixed precision | FP16 | 40% memory savings |
| Gradient clipping | 1.0 | Prevent exploding gradients |

#### 4.5.3 Loss Function

```python
# Binary Cross-Entropy with Logits
loss = BCEWithLogitsLoss(pos_weight=class_weight)

# Class weighting for imbalance
class_weight = n_negative / n_positive
```

---

## 5. System Architecture

### 5.1 Code Organization

```
eeg_depression_detection/
├── config/
│   ├── data_config.yaml      # Dataset parameters
│   ├── model_config.yaml     # Architecture hyperparameters
│   └── training_config.yaml  # Training settings
│
├── data/
│   ├── datasets/
│   │   └── figshare_dataset.py   # Dataset loader
│   ├── preprocessing/
│   │   └── filters.py            # Signal filtering
│   └── raw/figshare/             # Raw EDF files
│
├── features/
│   └── wavelet/
│       ├── wpd_extractor.py      # WPD feature extraction
│       └── cwt_extractor.py      # CWT scalogram generation
│
├── models/
│   ├── branches/
│   │   ├── transformer_encoder.py  # Vision Transformer
│   │   └── gnn_encoder.py          # Graph Attention Network
│   ├── fusion/
│   │   └── attention_fusion.py     # Cross-attention fusion
│   └── full_model.py               # Complete model
│
├── training/
│   └── trainer.py            # Training loop, LOSO CV
│
├── explainability/
│   └── integrated_gradients.py   # IG implementation
│
├── scripts/
│   └── train.py              # Main training script
│
└── docs/
    └── RESEARCH_PAPER_DRAFT.md   # This document
```

### 5.2 Model Summary

```
AdvancedEEGDepressionDetector(
  (transformer): EEGTransformerEncoder(
    (patch_embed): PatchEmbedding(...)
    (pos_embed): Parameter(64, 128)
    (transformer): TransformerEncoder(4 layers)
  )
  (gnn): EEGGraphAttentionNetwork(
    (convs): ModuleList(3 × GATConv)
    (norms): ModuleList(3 × LayerNorm)
  )
  (fusion): AttentionBasedFusion(
    (cross_attn_tg): MultiheadAttention(128, 4 heads)
    (cross_attn_gt): MultiheadAttention(128, 4 heads)
    (gate_t): Linear(256 → 1)
    (gate_g): Linear(256 → 1)
  )
  (classifier): ClassificationHead(
    (classifier): Sequential(Linear, BN, ReLU, Dropout, ...)
  )
)

Total parameters: 1,551,427
Trainable parameters: 1,551,427
```

---

## 6. Implementation Details

### 6.1 Dependencies

```
# Deep Learning
torch>=2.0
torch-geometric>=2.3
torchvision>=0.15

# EEG Processing
mne>=1.5
pywavelets>=1.4

# Scientific Computing
numpy>=1.24
scipy>=1.10
pandas>=2.0

# Machine Learning
scikit-learn>=1.3

# Explainability
captum>=0.6

# Visualization
matplotlib>=3.7
seaborn>=0.12
```

### 6.2 Hardware Requirements

- **GPU:** NVIDIA RTX 4070 (8GB VRAM)
- **RAM:** 16GB minimum
- **Storage:** 2GB for dataset + cache

### 6.3 Preprocessing Time

| Stage | Time | Notes |
|-------|------|-------|
| Load EDF | ~5s per file | MNE library |
| Filtering | ~10s per file | FIR + notch |
| WPD extraction | ~60s per file | 3 wavelets × 19 channels |
| CWT extraction | ~50s per file | 64 scales |
| **Total** | **~2 min per file** | Cached after first run |
| **58 files** | **~2 hours** | One-time preprocessing |

### 6.4 Training Time

| Stage | Time | Notes |
|-------|------|-------|
| Per epoch | ~2 min | Mixed precision |
| Per fold | ~30 epochs × 2 min = 1 hour | Early stopping |
| **58 folds** | **~8-10 hours** | Full LOSO |

---

## 7. Dataset

### 7.1 Figshare MDD EEG Dataset

**Source:** https://figshare.com/articles/dataset/EEG_Data_New/4244171

**Description:**
- Raw EEG recordings in EDF format
- 64 subjects: 34 MDD patients, 30 healthy controls
- 19-channel EEG (standard 10-20 montage)
- Sampling rate: 256 Hz
- Recording duration: ~5 minutes per condition
- Conditions: Eyes Closed (EC), Eyes Open (EO), Task

### 7.2 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total EDF files | 181 |
| EC condition files | 58 |
| Total subjects | 58 (using EC) |
| MDD patients | ~30 |
| Healthy controls | ~28 |
| Samples per subject | ~150 epochs |
| **Total samples** | **8,620** |
| Class 0 (Healthy) | 4,247 (49.3%) |
| Class 1 (MDD) | 4,373 (50.7%) |

### 7.3 Electrode Montage

```
Standard 10-20 System (19 electrodes):

        Fp1   Fp2
    F7  F3  Fz  F4  F8
    T3  C3  Cz  C4  T4
    T5  P3  Pz  P4  T6
        O1      O2
```

### 7.4 Preprocessing Applied

1. **Bandpass filter:** 1-45 Hz
2. **Notch filter:** 50 Hz, 60 Hz
3. **Resampling:** 256 Hz → 250 Hz
4. **Epoch segmentation:** 4 seconds, 50% overlap
5. **Artifact rejection:** Amplitude > 100 μV

---

## 8. Experimental Setup

### 8.1 Evaluation Protocol

**Cross-Validation:** Leave-One-Subject-Out (LOSO)
- 58 folds (one per subject)
- Train on 57 subjects, test on 1
- Aggregate predictions across all folds

### 8.2 Metrics

| Metric | Formula | Clinical Meaning |
|--------|---------|------------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Sensitivity | TP/(TP+FN) | Detection of depression (important) |
| Specificity | TN/(TN+FP) | Detection of healthy |
| Precision | TP/(TP+FP) | Positive predictive value |
| F1-Score | 2×(Prec×Rec)/(Prec+Rec) | Harmonic mean |
| AUC-ROC | Area under ROC curve | Discrimination ability |
| MCC | Matthews Correlation | Balanced metric for imbalanced data |

### 8.3 Baseline Comparisons

| Method | Architecture | Validation | Reported Accuracy |
|--------|--------------|------------|-------------------|
| Khaleghi et al. (2025) | CNN + SHAP | 5-fold | 98%* |
| Abidi et al. (2024) | SA-Gated DenseNet | Unknown | 96% |
| **Ours** | Transformer+GNN | LOSO | TBD |

*Likely inflated due to data leakage

---

## 9. Results

### 9.1 Training Progress

**Status:** Currently running LOSO cross-validation

```
Fold 1/58: Test subject = H_S1
  Epoch 10/30 - Train Loss: 0.XXXX
  Epoch 20/30 - Train Loss: 0.XXXX
  Epoch 30/30 - Train Loss: 0.XXXX
  Subject H_S1: True=0, Pred=X

[... continues for all 58 folds ...]
```

### 9.2 Final Results

**[TO BE FILLED AFTER TRAINING COMPLETES]**

| Metric | Value |
|--------|-------|
| Sample-level Accuracy | X.XX% |
| Subject-level Accuracy | X.XX% |
| AUC-ROC | X.XXX |
| F1-Score | X.XXX |
| Sensitivity | X.XX% |
| Specificity | X.XX% |
| MCC | X.XXX |

**Confusion Matrix:**
```
              Predicted
              0     1
Actual  0   [TN]  [FP]
        1   [FN]  [TP]
```

### 9.3 Comparison with Baseline

| Metric | Khaleghi et al. | Ours | Notes |
|--------|-----------------|------|-------|
| Accuracy | 98% (5-fold) | X% (LOSO) | LOSO is more honest |
| AUC-ROC | N/A | X.XX | |
| F1-Score | 0.98 | X.XX | |
| Validation | Data leakage risk | No leakage | Subject-wise split |

---

## 10. Explainability Analysis

### 10.1 Integrated Gradients (IG)

**Purpose:** Feature-level attribution

**Method:**
1. Define baseline (zero input = no brain activity)
2. Interpolate from baseline to input
3. Compute gradients at each interpolation point
4. Average gradients × (input - baseline)

**Output:**
- Attribution score for each WPD feature
- Attribution heatmap for scalogram regions

### 10.2 Layer-wise Relevance Propagation (LRP)

**Purpose:** Layer-by-layer explanation

**[TO BE IMPLEMENTED]**

### 10.3 TCAV (Testing with Concept Activation Vectors)

**Purpose:** Concept-level interpretation

**Clinical Concepts to Test:**
1. High frontal alpha asymmetry
2. Elevated theta power
3. Reduced alpha activity
4. High delta/alpha ratio
5. Low signal complexity

**[TO BE IMPLEMENTED]**

---

## 11. Discussion

### 11.1 Key Findings

**[TO BE FILLED AFTER RESULTS]**

### 11.2 Clinical Implications

1. **Objective biomarkers:** Identified EEG features associated with depression
2. **Interpretability:** Clinicians can understand model decisions
3. **Integration:** Compatible with existing EEG infrastructure

### 11.3 Limitations

1. **Dataset size:** 64 subjects is modest for deep learning
2. **Single condition:** Only Eyes Closed analyzed
3. **Binary classification:** No severity grading
4. **Medication effects:** Not controlled for

### 11.4 Future Work

1. Multi-site validation
2. Severity prediction (regression)
3. Longitudinal monitoring
4. Real-time implementation

---

## 12. Conclusion

**[TO BE WRITTEN AFTER RESULTS]**

---

## 13. References

1. World Health Organization. Depressive disorder (depression). March 2023.
2. Khaleghi et al. Interpretable deep learning for depression detection. MethodsX, 2025.
3. Lundberg & Lee. A unified approach to interpreting model predictions. NeurIPS, 2017.
4. Sundararajan et al. Axiomatic Attribution for Deep Networks. ICML, 2017.
5. Kim et al. Interpretability Beyond Feature Attribution: TCAV. ICML, 2018.
6. Acharya et al. Automated EEG-based screening of depression using CNN. CMPB, 2018.

---

## Appendix A: Configuration Files

### A.1 Model Configuration

```yaml
# model_config.yaml
transformer:
  d_model: 128
  nhead: 4
  num_layers: 4
  dim_ff: 512
  dropout: 0.1
  patch_size: [8, 16]

gnn:
  hidden_dim: 128
  num_heads: 4
  num_layers: 3
  dropout: 0.3

fusion:
  dim: 128
  num_heads: 4
  use_gating: true

classifier:
  hidden_dims: [64, 32]
  dropout: [0.5, 0.3]
```

### A.2 Training Configuration

```yaml
# training_config.yaml
optimizer:
  name: AdamW
  lr: 0.0001
  weight_decay: 0.01

scheduler:
  name: CosineAnnealingWarmRestarts
  T_0: 10
  T_mult: 2

training:
  batch_size: 16
  epochs: 100
  early_stopping_patience: 15
  gradient_accumulation: 2
  gradient_clip: 1.0
  mixed_precision: true

cross_validation:
  type: LOSO
```

---

## Appendix B: Command Reference

### Preprocessing
```bash
python preprocess.py
```

### Training
```bash
python scripts/train.py \
    --data_dir data/raw/figshare \
    --output_dir outputs \
    --epochs 100 \
    --batch_size 16 \
    --cv_type loso \
    --mixed_precision
```

### Quick Test (Debug Mode)
```bash
python scripts/train.py --data_dir data/raw/figshare --debug
```

---

*Document generated: 2026-01-31*
*Training status: In progress*
