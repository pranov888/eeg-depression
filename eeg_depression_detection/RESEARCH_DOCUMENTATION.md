# Research Documentation: Advanced EEG-Based Depression Detection System

## Paper Title (Proposed)
**"A Multi-Branch Deep Learning Framework with Transformer-GNN Fusion and Multi-Level Explainability for EEG-Based Depression Detection"**

---

## 1. Introduction & Motivation

### 1.1 Problem Statement
Depression affects over 280 million people worldwide (WHO, 2023). Traditional diagnosis relies on subjective clinical interviews (PHQ-9, BDI-II), which suffer from:
- Inter-rater variability
- Patient recall bias
- Cultural and language barriers
- Lack of objective biomarkers

### 1.2 Why EEG?
EEG provides:
- **Objective measurement** of neural activity
- **Non-invasive** and relatively inexpensive
- **High temporal resolution** (milliseconds)
- **Established biomarkers** for depression:
  - Elevated theta power (4-8 Hz) in frontal regions
  - Reduced alpha power (8-13 Hz)
  - Increased delta/alpha ratio
  - Frontal alpha asymmetry (FAA)
  - Reduced Hjorth complexity

### 1.3 Limitations of Existing Work
The baseline paper (Khaleghi et al., 2025) achieved 98% accuracy using:
- Simple CNN architecture
- SHAP for interpretability
- Pre-extracted features
- Standard k-fold cross-validation (potential data leakage)

**Our improvements address:**
1. **Feature Representation**: Advanced wavelet techniques (WPD + CWT) vs. basic features
2. **Model Architecture**: Transformer + GNN vs. simple CNN
3. **Spatial Modeling**: GNN captures electrode topology and brain connectivity
4. **Temporal Modeling**: Transformer captures long-range time-frequency dependencies
5. **Interpretability**: Multi-level XAI (IG + LRP + TCAV) vs. SHAP only
6. **Evaluation**: Subject-wise cross-validation to prevent data leakage

---

## 2. Dataset Selection & Justification

### 2.1 Primary Dataset: Figshare MDD EEG Dataset

**Source:** https://figshare.com/articles/dataset/EEG_Data_New/4244171

**Characteristics:**
| Property | Value | Justification |
|----------|-------|---------------|
| Subjects | 64 (34 MDD, 30 HC) | Adequate for LOSO CV with 64 folds |
| Electrodes | 19 (10-20 montage) | Standard clinical setup, reproducible |
| Sampling Rate | 256 Hz | Sufficient for 0-45 Hz analysis |
| Duration | 5 min eyes-closed | Resting state, minimal artifacts |
| Diagnosis | Clinical MDD | Validated by psychiatrists |

**Why Figshare over alternatives:**

| Dataset | Pros | Cons | Decision |
|---------|------|------|----------|
| **Figshare MDD** | Raw signals, clinical diagnosis, standard montage | Smaller size (64) | **SELECTED** - enables full WPD+CWT pipeline |
| MODMA | 128 electrodes, high resolution | Requires registration, complex access | Secondary option |
| Kaggle | Large (232), easy access | Pre-extracted features only, limits innovation | Benchmarking only |

### 2.2 Why Raw Signals Matter

Pre-extracted features (like Kaggle dataset) limit:
- Cannot apply custom wavelet families (db4, sym5, coif3)
- Cannot generate CWT scalograms for Transformer input
- Cannot experiment with different preprocessing
- Cannot extract channel-wise features for GNN

**Decision Rationale:** Raw signals enable the full proposed pipeline:
```
Raw EEG → Custom Preprocessing → WPD + CWT → Transformer + GNN → Fusion
```

---

## 3. Architecture Design Decisions

### 3.1 Dual-Branch Architecture Rationale

**Why two branches instead of one?**

EEG signals contain two complementary types of information:
1. **Time-frequency patterns** (what oscillations occur when)
2. **Spatial relationships** (how brain regions interact)

A single model struggles to capture both effectively.

| Aspect | Transformer Branch | GNN Branch |
|--------|-------------------|------------|
| **Input** | CWT scalograms | WPD features per electrode |
| **Captures** | Global time-frequency dependencies | Inter-electrode spatial relationships |
| **Attention** | Self-attention over patches | Graph attention over electrodes |
| **Key for** | Temporal pattern recognition | Connectivity-based biomarkers |

### 3.2 Transformer Design Decisions

**Architecture Choice: Vision Transformer (ViT) Style**

**Rationale:**
- Scalograms are 2D images (time × frequency)
- ViT has proven effective for image classification
- Self-attention captures long-range dependencies that CNNs miss
- Patches reduce sequence length for computational efficiency

**Hyperparameters (optimized for <8GB VRAM):**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| d_model | 128 | Reduced from 256 for memory; still expressive |
| nhead | 4 | d_model / nhead = 32 (reasonable head dim) |
| num_layers | 4 | Balances depth vs. memory; ViT-Tiny uses 12 |
| dim_ff | 512 | 4× d_model standard ratio |
| patch_size | (8, 16) | 64 patches from 64×128 scalogram |
| dropout | 0.1 | Standard for Transformers |

**Why not larger?**
- 8GB VRAM constraint
- Small dataset (64 subjects) risks overfitting with larger models
- Can scale up after validation

### 3.3 GNN Design Decisions

**Architecture Choice: Graph Attention Network (GAT)**

**Why GNN over Bi-LSTM?**

| Criterion | GNN (GAT) | Bi-LSTM |
|-----------|-----------|---------|
| Spatial topology | ✓ Explicitly models electrode positions | ✗ Treats channels as sequence |
| Brain connectivity | ✓ Edge weights = functional connectivity | ✗ No connectivity modeling |
| Non-Euclidean | ✓ Native graph structure | ✗ Assumes Euclidean |
| Interpretability | ✓ Attention reveals important connections | Limited |
| Depression biomarkers | ✓ Can learn frontal asymmetry, connectivity | Temporal only |

**Graph Construction Strategy: Hybrid Adjacency**

We combine two types of edges:

1. **Spatial Adjacency (k=6 nearest neighbors)**
   - Based on 3D electrode positions
   - Captures anatomical proximity
   - Ensures local connectivity

2. **Functional Connectivity (correlation > 0.5)**
   - Based on Pearson correlation between channels
   - Captures functional relationships
   - May reveal depression-specific patterns

**Edge formula:**
```
A_hybrid = max(A_spatial, A_functional)
```

**GAT Hyperparameters:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| num_nodes | 19 | 10-20 montage |
| node_feat_dim | 576 | 3 wavelets × 32 nodes × 6 features |
| hidden_dim | 128 | Matches Transformer for fusion |
| num_heads | 4 | Multi-head attention for diversity |
| num_layers | 3 | Sufficient for 19-node graph |
| dropout | 0.3 | Higher than Transformer (small graph) |

### 3.4 Fusion Strategy

**Why Attention-Based Fusion over Simple Concatenation?**

| Method | Pros | Cons |
|--------|------|------|
| Concatenation | Simple, fast | Equal weight to both branches |
| Addition | Preserves dimension | Requires same dim, loses information |
| **Cross-Attention** | Learns adaptive weights, captures interactions | More parameters |
| Gated Fusion | Selective combination | May ignore one branch |

**Our approach: Cross-Attention + Gating**

1. **Cross-Attention:** Each branch attends to the other
   - Transformer attends to GNN: "Which spatial patterns matter for this time-frequency input?"
   - GNN attends to Transformer: "Which time-frequency patterns matter for this spatial configuration?"

2. **Gating:** Learn when to trust each branch
   - Some samples may have stronger temporal vs. spatial signatures
   - Gate weights are learned, not fixed

---

## 4. Feature Extraction Decisions

### 4.1 Wavelet Packet Decomposition (WPD)

**Why WPD over DWT?**

| Method | Frequency Resolution | Flexibility |
|--------|---------------------|-------------|
| DWT | Low at high frequencies | Fixed bands |
| **WPD** | Uniform across all frequencies | All subbands accessible |

WPD decomposes into 2^L nodes at level L, giving fine-grained frequency analysis.

**Wavelet Selection:**

| Wavelet | Properties | Why Selected |
|---------|------------|--------------|
| **db4** | Good time-frequency localization | Standard for EEG, similar to neural waveforms |
| **sym5** | Near-symmetric, smooth | Reduces boundary effects |
| **coif3** | Symmetric, vanishing moments | Captures polynomial trends |

**Using multiple wavelets captures different aspects:**
- db4: Sharp transients
- sym5: Smooth oscillations
- coif3: Trend components

**Features per node (6 features):**
1. **Energy**: Total power in band
2. **Shannon Entropy**: Complexity/irregularity
3. **Log Energy Entropy**: Alternative entropy measure
4. **Mean**: DC component
5. **Std**: Variability
6. **Skewness/Kurtosis**: Distribution shape

**Total features per channel:**
```
3 wavelets × 32 nodes × 6 features = 576 features/channel
```

### 4.2 Continuous Wavelet Transform (CWT)

**Why CWT for Transformer input?**

- Produces 2D time-frequency representation (scalogram)
- Natural input for vision-style Transformer
- Captures both "when" and "what frequency" simultaneously

**Wavelet Selection: Complex Morlet (cmor1.5-1.0)**

| Property | Value | Rationale |
|----------|-------|-----------|
| Bandwidth | 1.5 | Balances time-frequency resolution |
| Center frequency | 1.0 | Good for EEG frequency range |
| Complex | Yes | Phase information available |

**Frequency Range: 1-45 Hz**
- Covers all relevant EEG bands (delta through gamma)
- Above 45 Hz is mostly noise/EMG artifact

**Output Size: 64 × 128**
- 64 frequency bins (logarithmically spaced)
- 128 time bins (covers 4-second epoch)
- Results in 64 patches for Transformer (8×16 patch size)

---

## 5. Training Strategy Decisions

### 5.1 Cross-Validation: LOSO vs K-Fold

**Critical Decision: Leave-One-Subject-Out (LOSO)**

**Why LOSO is essential:**

Standard k-fold splits samples randomly, causing **data leakage**:
- Same subject's epochs appear in both train and test
- Model memorizes subject-specific patterns, not depression patterns
- Inflated accuracy (often 95-99%) that doesn't generalize

**LOSO ensures:**
- All epochs from one subject in test set only
- Model must generalize to unseen subjects
- Realistic clinical scenario (new patient)

**Expected accuracy drop:**
- K-fold with leakage: 95-99%
- LOSO (realistic): 85-92%

This drop is expected and honest.

### 5.2 Memory Optimization (<8GB VRAM)

| Technique | Memory Saving | Trade-off |
|-----------|--------------|-----------|
| **Mixed Precision (FP16)** | ~40% | Slight numerical instability |
| **Gradient Accumulation** | Linear with steps | Slower training |
| **Gradient Checkpointing** | ~30% | 20% slower |
| **Reduced batch size** | Linear | Noisier gradients |

**Our configuration:**
```python
batch_size = 16
gradient_accumulation_steps = 2  # Effective batch = 32
mixed_precision = True
gradient_checkpointing = True
```

### 5.3 Optimizer Selection: AdamW

**Why AdamW over Adam?**

| Optimizer | Weight Decay | Behavior |
|-----------|-------------|----------|
| Adam + L2 | Applied to gradient | Coupled with momentum, less effective |
| **AdamW** | Applied directly to weights | Decoupled, better generalization |

**Hyperparameters:**
- Learning rate: 1e-4 (conservative for small dataset)
- Weight decay: 0.01 (regularization)
- Betas: (0.9, 0.999) (standard)

### 5.4 Learning Rate Schedule: Cosine Annealing with Warm Restarts

**Why this schedule?**

- Gradual decay helps fine-tuning
- Warm restarts escape local minima
- Works well with Transformers

---

## 6. Explainability Framework

### 6.1 Multi-Level XAI Rationale

Single XAI method limitations:
- SHAP: Global importance, slow for deep networks
- Gradient-based: Local, may be noisy

**Our multi-level approach:**

| Level | Method | What it shows | Clinical value |
|-------|--------|---------------|----------------|
| Feature | Integrated Gradients | Which input features matter | Identify EEG biomarkers |
| Layer | LRP | Information flow through network | Understand model reasoning |
| Concept | TCAV | High-level clinical concepts | Validate against domain knowledge |

### 6.2 Integrated Gradients

**Why IG over vanilla gradients?**

- Satisfies **Sensitivity**: If feature matters, attribution is non-zero
- Satisfies **Implementation Invariance**: Same function = same attribution
- Path integral provides complete attribution

**Baseline choice: Zero**
- Zero EEG signal = no brain activity
- Meaningful reference point

### 6.3 Layer-wise Relevance Propagation (LRP)

**Why LRP?**

- Decomposes prediction into input contributions
- Conserves relevance (what goes in = what comes out)
- Layer-by-layer interpretation

**Rule: LRP-epsilon**
- Numerically stable
- Suitable for ReLU/GELU networks

### 6.4 TCAV (Testing with Concept Activation Vectors)

**Why TCAV?**

- Explains in terms humans understand
- Tests specific clinical hypotheses
- Validates model learns correct patterns

**Clinical concepts to test:**

| Concept | Definition | Expected for Depression |
|---------|------------|------------------------|
| High frontal alpha asymmetry | F4_alpha - F3_alpha > threshold | Positive correlation |
| Elevated theta power | Mean theta > 75th percentile | Positive correlation |
| Reduced alpha activity | Mean alpha < 25th percentile | Positive correlation |
| High delta/alpha ratio | Ratio > 70 | Positive correlation |

---

## 7. Experimental Protocol

### 7.1 Data Preprocessing Pipeline

```
1. Load raw EEG (256 Hz, 19 channels, 5 min)
2. Bandpass filter: 1-45 Hz (FIR, order 101)
3. Notch filter: 50 Hz, 60 Hz (remove power line)
4. ICA artifact removal (Picard algorithm)
5. Re-reference to average
6. Resample to 250 Hz
7. Segment into 4-second epochs (50% overlap)
8. Reject epochs with amplitude > 100 μV
```

### 7.2 Feature Extraction Pipeline

```
For each epoch:
1. WPD extraction:
   - Apply db4, sym5, coif3 wavelets
   - 5-level decomposition (32 nodes)
   - Extract 6 features per node
   - Result: (19 channels, 576 features)

2. CWT extraction:
   - Apply Complex Morlet wavelet
   - 64 frequency scales (1-45 Hz)
   - Resize to 64×128
   - Result: (19 channels, 64, 128)
```

### 7.3 Training Protocol

```
1. LOSO Cross-Validation (64 folds):
   For each subject as test:
   a. Train on remaining 63 subjects
   b. Validate on held-out subject
   c. Record metrics

2. Per-fold training:
   - Max epochs: 100
   - Early stopping: patience 15
   - Best model by validation AUC

3. Aggregate results:
   - Mean ± std across folds
   - 95% confidence intervals
```

### 7.4 Evaluation Metrics

| Metric | Formula | Why Important |
|--------|---------|---------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Sensitivity | TP/(TP+FN) | Catch all depression cases |
| Specificity | TN/(TN+FP) | Avoid false alarms |
| F1-Score | 2×(P×R)/(P+R) | Balance precision/recall |
| AUC-ROC | Area under ROC curve | Threshold-independent |
| MCC | Matthews Correlation | Handles class imbalance |

---

## 8. Implementation Log

### 8.1 Project Structure Created
```
eeg_depression_detection/
├── config/           # YAML configurations
├── data/            # Data loading and preprocessing
├── features/        # WPD, CWT extraction
├── models/          # Transformer, GNN, Fusion
├── explainability/  # IG, LRP, TCAV
├── training/        # Training loop, CV
└── evaluation/      # Metrics, visualization
```

### 8.2 Key Files Implemented

| File | Purpose | Status |
|------|---------|--------|
| `config/model_config.yaml` | Model hyperparameters | ✓ Complete |
| `config/training_config.yaml` | Training settings | ✓ Complete |
| `config/data_config.yaml` | Dataset configuration | ✓ Complete |
| `data/preprocessing/filters.py` | Bandpass, notch filtering | ✓ Complete |
| `data/datasets/figshare_dataset.py` | Dataset loader with caching | ✓ Complete |
| `features/wavelet/wpd_extractor.py` | WPD feature extraction | ✓ Complete |
| `features/wavelet/cwt_extractor.py` | CWT scalogram generation | ✓ Complete |
| `models/branches/transformer_encoder.py` | Transformer branch | ✓ Complete |
| `models/branches/gnn_encoder.py` | GNN branch with ElectrodeGraph | ✓ Complete |
| `models/fusion/attention_fusion.py` | Cross-attention fusion with gating | ✓ Complete |
| `models/full_model.py` | Complete integrated model | ✓ Complete |
| `training/trainer.py` | Training loop with LOSO CV | ✓ Complete |
| `explainability/integrated_gradients.py` | Integrated Gradients XAI | ✓ Complete |
| `scripts/train.py` | Main training script | ✓ Complete |

### 8.3 Implementation Highlights

**Total Lines of Code:** ~3,500+ lines of documented Python

**Key Implementation Features:**
1. **Memory-optimized architecture** for <8GB VRAM
2. **Comprehensive documentation** with design rationale
3. **Modular design** allowing component-wise testing
4. **Cached feature extraction** for faster experimentation
5. **Mixed precision training** for memory efficiency
6. **Gradient accumulation** for effective larger batch sizes

---

## 9. Expected Results & Contributions

### 9.1 Expected Performance

| Metric | Baseline (CNN+SHAP) | Our Method (Expected) |
|--------|--------------------|-----------------------|
| Accuracy | 98%* | 88-92% |
| AUC-ROC | Not reported | 0.90-0.95 |
| Evaluation | K-fold (leaky) | LOSO (honest) |

*Baseline 98% is likely inflated due to data leakage

### 9.2 Key Contributions

1. **Novel Architecture**: First Transformer + GNN fusion for EEG depression
2. **Multi-wavelet Features**: db4 + sym5 + coif3 WPD combined with CWT
3. **Spatial Modeling**: GNN for electrode topology and connectivity
4. **Attention Fusion**: Cross-attention with gating for adaptive combination
5. **Multi-level XAI**: IG + LRP + TCAV for comprehensive interpretability
6. **Rigorous Evaluation**: LOSO cross-validation for honest assessment

---

## 10. References

1. Khaleghi, P. et al. (2025). Interpretable deep learning for depression detection in neurological patients using EEG signals. MethodsX.
2. Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model predictions. NeurIPS.
3. Cai, H. et al. (2023). A pervasive approach to EEG-based depression detection. Complexity.
4. Dosovitskiy, A. et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale.
5. Veličković, P. et al. (2018). Graph attention networks. ICLR.
6. Sundararajan, M. et al. (2017). Axiomatic attribution for deep networks. ICML.
7. Bach, S. et al. (2015). On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. PLoS ONE.
8. Kim, B. et al. (2018). Interpretability beyond feature attribution: Quantitative testing with concept activation vectors. ICML.

---

*Document last updated: During implementation*
*Authors: [To be added]*
