# Explainability Methods for EEG Depression Detection

## Overview

This document details three explainability methods implemented for understanding how our model detects depression from EEG signals.

| Method | Question Answered | Output Type | Best For |
|--------|-------------------|-------------|----------|
| **Integrated Gradients** | "Which input features matter?" | Pixel/feature attributions | Identifying important electrodes, time points |
| **LRP** | "How does each layer contribute?" | Relevance scores | Understanding information flow |
| **TCAV** | "Does model use clinical concepts?" | Concept influence % | Validating clinical relevance |

---

## 1. Integrated Gradients (IG)

### Theory

Integrated Gradients attributes the model's prediction to its input features by computing the path integral of gradients along a straight line from a baseline to the input.

**Mathematical Definition:**
```
IG_i(x) = (x_i - x'_i) × ∫₀¹ (∂F(x' + α(x - x')) / ∂x_i) dα
```

Where:
- `x` = input
- `x'` = baseline (typically zeros)
- `F` = model function
- `α` = interpolation parameter

### Why It Works

1. **Sensitivity:** If a feature changes the output, it gets non-zero attribution
2. **Implementation Invariance:** Same attributions for functionally equivalent networks
3. **Completeness:** Attributions sum to the output difference from baseline

### For EEG

IG reveals:
- **Which electrodes** contribute most to depression prediction
- **Which time points** in the epoch are most informative
- **Which frequency components** (via scalogram) the model uses

### Implementation

```python
from explainability.integrated_gradients import IntegratedGradientsExplainer

explainer = IntegratedGradientsExplainer(model, device='cuda')

# Get attributions for a sample
attributions = explainer.explain(
    scalogram,      # (1, 1, 64, 128)
    wpd_features,   # (1, 19, 576)
    target_class=1  # Explain depression prediction
)

# attributions['scalogram'] -> (64, 128) heatmap
# attributions['wpd'] -> (19, 576) electrode attributions
```

### Interpretation Example

```
Electrode Importance (IG):
  F3:  0.234  ← High importance (left frontal)
  F4:  0.198  ← High importance (right frontal)
  Fz:  0.156
  Cz:  0.089
  O1:  0.023  ← Low importance (occipital)
```

**Clinical Interpretation:** The model focuses on frontal electrodes (F3, F4, Fz), consistent with frontal alpha asymmetry findings in depression literature.

---

## 2. Layer-wise Relevance Propagation (LRP)

### Theory

LRP decomposes the model's output back through the network, assigning relevance scores to each neuron at each layer based on their contribution to the final prediction.

**Conservation Principle:**
```
R_output = Σ R_input_features
```

The total relevance is conserved as it propagates backwards.

**LRP-ε Rule (used in our implementation):**
```
R_i = Σⱼ (aᵢwᵢⱼ / (Σₖ aₖwₖⱼ + ε)) × Rⱼ
```

Where:
- `R_i` = relevance of neuron i
- `a_i` = activation of neuron i
- `w_ij` = weight from i to j
- `ε` = small constant for stability

### LRP Rules

| Rule | Formula | Use Case |
|------|---------|----------|
| **LRP-0** | Basic propagation | Simple networks |
| **LRP-ε** | Adds ε to denominator | Numerical stability |
| **LRP-γ** | w → w + γ×max(0,w) | Favors positive contributions |
| **LRP-αβ** | Separate ± contributions | Detailed analysis |

### For EEG

LRP provides:
- **Electrode relevance:** Which channels carry depression-related information
- **Frequency band relevance:** Delta, theta, alpha, beta, gamma contributions
- **Temporal relevance:** Which time windows matter

### Implementation

```python
from explainability.lrp import create_lrp_analyzer

analyzer = create_lrp_analyzer(model, rule='epsilon', device='cuda')

# Compute electrode importance across samples
electrode_results = analyzer.compute_electrode_importance(dataloader, n_samples=100)

# Results include:
# - mean_relevance: Average relevance per electrode
# - mdd_relevance: Relevance for depression samples
# - healthy_relevance: Relevance for healthy samples
# - relevance_difference: MDD - Healthy (what distinguishes them)
```

### Interpretation Example

```
Frequency Band Relevance (LRP):
  Band      Mean    MDD     Healthy   Diff
  ─────────────────────────────────────────
  delta     0.12    0.15    0.09      +0.06
  theta     0.28    0.35    0.21      +0.14  ← Higher in MDD
  alpha     0.31    0.24    0.38      -0.14  ← Lower in MDD
  beta      0.18    0.16    0.20      -0.04
  gamma     0.11    0.10    0.12      -0.02
```

**Clinical Interpretation:** The model finds higher theta and lower alpha relevance for MDD patients, consistent with the established EEG biomarkers of depression.

---

## 3. TCAV (Testing with Concept Activation Vectors)

### Theory

TCAV tests whether a model uses human-understandable concepts in its predictions. Instead of asking "which pixels matter?", it asks "does the model use this clinical concept?"

**Key Steps:**

1. **Define Concept:** e.g., "frontal alpha asymmetry"
2. **Collect Examples:** Samples WITH and WITHOUT the concept
3. **Train CAV:** Linear classifier in activation space
4. **Compute TCAV Score:** Fraction of inputs where concept increases prediction

**TCAV Score:**
```
TCAV_c = |{x : S_c(x) > 0}| / |X|
```

Where `S_c(x)` is the directional derivative of the model output in the CAV direction.

### EEG Concepts Implemented

| Concept | Description | Depression Pattern |
|---------|-------------|-------------------|
| `alpha_asymmetry` | Right > left frontal alpha | Positive asymmetry |
| `theta_elevation` | Elevated frontal theta | Increased ratio |
| `beta_suppression` | Reduced global beta | Lower power |
| `delta_abnormality` | Excessive delta (>30%) | Higher proportion |
| `alpha_reduction` | Reduced global alpha | Lower power |
| `coherence_reduction` | Reduced interhemispheric coherence | Lower coherence |

### Implementation

```python
from explainability.tcav import create_tcav_analyzer, EEGConceptLibrary

# Create analyzer
analyzer = create_tcav_analyzer(model, layer_name='fusion', device='cuda')

# Run full analysis
results = analyzer.run_full_analysis(
    dataloader,
    concepts=['alpha_asymmetry', 'theta_elevation', 'alpha_reduction'],
    n_cav_examples=50,
    n_tcav_samples=100
)

# Results for each concept:
# - cav_accuracy: How well concept separates in activation space
# - tcav_score: Fraction where concept increases prediction
# - p_value: Statistical significance
```

### Interpretation Example

```
TCAV Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Concept              CAV Acc   MDD TCAV   Healthy TCAV   Significant
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
alpha_asymmetry      0.82      0.78       0.31           Yes (p<0.01)
theta_elevation      0.75      0.65       0.42           Yes (p<0.05)
alpha_reduction      0.79      0.71       0.35           Yes (p<0.01)
coherence_reduction  0.68      0.58       0.47           No  (p=0.12)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Clinical Interpretation:**
- **Alpha asymmetry** has 78% influence on MDD prediction → Model strongly uses this clinically-validated biomarker
- **Theta elevation** has 65% influence → Model uses this secondary biomarker
- **Coherence reduction** not significant → Model doesn't rely on this (may be less reliable marker)

### What TCAV Scores Mean

| TCAV Score | Interpretation |
|------------|----------------|
| **>0.7** | Strong positive influence (model uses this concept) |
| **0.5-0.7** | Moderate influence |
| **~0.5** | No influence (random) |
| **<0.3** | Negative influence (concept decreases prediction) |

---

## Comparison of Methods

### When to Use Each

| Scenario | Best Method |
|----------|-------------|
| "Which electrodes matter most?" | IG or LRP |
| "What frequency bands are important?" | LRP |
| "Does model use alpha asymmetry?" | TCAV |
| "Is model learning artifacts?" | TCAV (test "muscle artifact" concept) |
| "Visualize attention patterns" | IG (scalogram heatmaps) |
| "Validate clinical relevance" | TCAV |

### Complementary Insights

```
               IG                    LRP                  TCAV
              ─────                 ─────                ─────
Question:   "What matters?"     "How flows?"        "Uses concept?"

Output:     Feature heatmap     Layer relevance     Concept score

Strengths:  - Precise           - Decomposable      - Clinical terms
            - Visual            - Layer-wise        - Hypothesis test
            - Any network       - Interpretable     - Human concepts

Weaknesses: - No concepts       - Complex rules     - Needs examples
            - Baseline choice   - Network specific  - Binary concepts
```

---

## Running the Analysis

### Quick Start

```bash
# Run all methods
python explainability/run_explainability.py \
    --data_dir data/raw/figshare \
    --output_dir explainability_results \
    --n_samples 100 \
    --methods ig lrp tcav

# Run specific method
python explainability/run_explainability.py \
    --data_dir data/raw/figshare \
    --methods tcav
```

### Output Files

```
explainability_results/run_YYYYMMDD_HHMMSS/
├── explainability_results.json    # All numerical results
├── electrode_importance.png       # Topographic map (if generated)
├── frequency_importance.png       # Frequency band chart
└── tcav_summary.png              # TCAV scores visualization
```

---

## Clinical Validation Checklist

Use these results to validate your model:

- [ ] **Top electrodes are frontal** (F3, F4, Fz) - consistent with depression literature
- [ ] **Alpha asymmetry TCAV > 0.6** - model uses this established biomarker
- [ ] **Theta elevation TCAV > 0.5** - secondary biomarker is used
- [ ] **Artifact concepts TCAV < 0.3** - model not relying on artifacts
- [ ] **Frequency importance:** Alpha and theta higher than delta/gamma
- [ ] **No suspicious patterns:** e.g., edge electrodes too important (cable artifacts)

---

## References

1. Sundararajan et al. (2017): "Axiomatic Attribution for Deep Networks" - Integrated Gradients
2. Bach et al. (2015): "On Pixel-Wise Explanations by Layer-Wise Relevance Propagation"
3. Kim et al. (2018): "Interpretability Beyond Feature Attribution: TCAV"
4. Montavon et al. (2017): "Explaining Deep Neural Networks"
5. Henriques et al. (1991): "Frontal Alpha Asymmetry and Depression" - Clinical validation
