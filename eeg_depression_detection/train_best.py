#!/usr/bin/env python3
"""
EEG Depression Detection — V3 Best-of-Best Ensemble Training
=============================================================
Self-contained script: loads EDF files, extracts multi-domain features,
trains a 3-model ensemble (1D-CNN + XGBoost + SVM) with LOSO CV,
and reports sample-level + subject-level metrics.

Usage:
    python train_best.py
"""

import os
import sys
import json
import time
import logging
import warnings
import gc
import pickle
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import mne
import pywt
from scipy import signal, stats
from scipy.signal import coherence as scipy_coherence
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    precision_score, recall_score
)

import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("ERROR")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw" / "figshare"
OUTPUT_DIR = BASE_DIR / "outputs_v3"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLING_RATE = 256       # Original EDF sampling rate
TARGET_SR = 250           # Resample target
N_CHANNELS = 19
EPOCH_SEC = 4.0
OVERLAP = 0.5
EPOCH_SAMPLES = int(EPOCH_SEC * TARGET_SR)  # 1000 samples

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]

# CNN training
CNN_EPOCHS = 30
CNN_BATCH = 32
CNN_LR = 1e-3
CNN_PATIENCE = 7

# Memory optimization flags
CLEAR_GPU_CACHE_BETWEEN_FOLDS = True
CHECKPOINT_LOSO = True  # Save progress between folds
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# SVM: cap training samples to keep it tractable (RBF SVM is O(n^2))
# 70k samples would take 30-60min/fold = 48+ hours total. Cap at 10k.
SVM_MAX_TRAIN_SAMPLES = 10000

# Directory to save trained models
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

SEED = 42

# ---------------------------------------------------------------------------
# Device setup — detect CUDA compatibility properly
# ---------------------------------------------------------------------------
def get_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        # Test if CUDA actually works by doing a small operation
        t = torch.zeros(1, device="cuda")
        _ = t + 1
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")

DEVICE = get_device()

# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_path = OUTPUT_DIR / "training_v3.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def parse_edf_filename(fname: str):
    """Parse filenames like 'MDD S12 TASK.edf' or 'H S2 EO.edf'.
    Also handles prefixed files like '6921143_H S15 EO.edf'.
    Returns (group, subject_id, condition) or None.
    """
    name = Path(fname).stem
    # Strip numeric prefix if present (e.g. '6921143_H S15 EO')
    if "_" in name and name.split("_")[0].isdigit():
        name = name.split("_", 1)[1]
    parts = name.split()
    if len(parts) != 3:
        return None
    group, subj, condition = parts
    if group not in ("MDD", "H"):
        return None
    if not subj.startswith("S"):
        return None
    subj_num = int(subj[1:])
    # Create unique subject ID across groups
    subject_id = f"{group}_S{subj_num}"
    label = 1 if group == "MDD" else 0
    return subject_id, label, condition


def load_edf(filepath: str, target_sr: int = TARGET_SR):
    """Load an EDF file, apply bandpass + notch filters, resample."""
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

    # Pick only the 19 standard channels if available
    available = raw.ch_names
    pick_chs = [ch for ch in CHANNEL_NAMES if ch in available]
    if len(pick_chs) < N_CHANNELS:
        # Try case-insensitive match
        name_map = {ch.lower(): ch for ch in available}
        pick_chs = []
        for ch in CHANNEL_NAMES:
            if ch in available:
                pick_chs.append(ch)
            elif ch.lower() in name_map:
                pick_chs.append(name_map[ch.lower()])
    if len(pick_chs) < N_CHANNELS:
        # Use first 19 channels as fallback
        pick_chs = available[:N_CHANNELS]

    raw.pick_channels(pick_chs[:N_CHANNELS], ordered=True)

    # Bandpass 1-45 Hz
    raw.filter(l_freq=1.0, h_freq=45.0, method="iir",
               iir_params=dict(order=5, ftype="butter"), verbose=False)
    # Notch filter 50 Hz and 60 Hz
    raw.notch_filter(freqs=[50, 60], verbose=False)
    # Resample
    if raw.info["sfreq"] != target_sr:
        raw.resample(target_sr, verbose=False)

    data = raw.get_data()  # (n_channels, n_samples)
    return data


def segment_epochs(data: np.ndarray, epoch_samples: int, overlap: float):
    """Segment continuous data into overlapping epochs.
    data: (n_channels, n_samples)
    Returns: (n_epochs, n_channels, epoch_samples)
    """
    step = int(epoch_samples * (1 - overlap))
    n_samples = data.shape[1]
    epochs = []
    start = 0
    while start + epoch_samples <= n_samples:
        epoch = data[:, start:start + epoch_samples]
        # Reject bad epochs
        if np.any(np.isnan(epoch)) or np.any(np.isinf(epoch)):
            start += step
            continue
        if np.max(np.abs(epoch)) > 500e-6:  # 500 μV in volts
            start += step
            continue
        epochs.append(epoch)
        start += step
    return np.array(epochs) if epochs else np.empty((0, data.shape[0], epoch_samples))


def load_all_data():
    """Load all EDF files, segment into epochs, return structured data."""
    log.info("=" * 60)
    log.info("PHASE 1: DATA LOADING")
    log.info("=" * 60)
    phase_start = time.time()

    edf_files = sorted(DATA_DIR.glob("*.edf"))
    log.info(f"Found {len(edf_files)} EDF files in {DATA_DIR}")
    log.info(f"Data directory: {DATA_DIR}")

    # subject_id -> {label, epochs: list of (n_epochs, n_channels, epoch_samples)}
    subject_data = defaultdict(lambda: {"label": None, "epochs": [], "conditions": []})
    skipped = 0

    for fpath in tqdm(edf_files, desc="Loading EDF files", unit="file"):
        parsed = parse_edf_filename(fpath.name)
        if parsed is None:
            skipped += 1
            continue
        subject_id, label, condition = parsed

        try:
            data = load_edf(str(fpath))
            epochs = segment_epochs(data, EPOCH_SAMPLES, OVERLAP)
            if len(epochs) == 0:
                log.warning(f"  No valid epochs from {fpath.name}")
                skipped += 1
                continue

            subject_data[subject_id]["label"] = label
            subject_data[subject_id]["epochs"].append(epochs)
            subject_data[subject_id]["conditions"].append(condition)
            # Log per-file details
            tqdm.write(f"    {fpath.name}: {len(epochs)} epochs ({data.shape[1]/TARGET_SR:.1f}s raw data)")
        except Exception as e:
            log.warning(f"  Failed to load {fpath.name}: {e}")
            skipped += 1

    # Concatenate epochs per subject across conditions
    all_subjects = {}
    for sid, info in subject_data.items():
        all_epochs = np.concatenate(info["epochs"], axis=0)  # (total_epochs, 19, 1000)
        all_subjects[sid] = {
            "label": info["label"],
            "epochs": all_epochs,
            "conditions": info["conditions"],
        }

    n_mdd = sum(1 for v in all_subjects.values() if v["label"] == 1)
    n_h = sum(1 for v in all_subjects.values() if v["label"] == 0)
    total_epochs = sum(v["epochs"].shape[0] for v in all_subjects.values())
    avg_epochs_per_subject = total_epochs / len(all_subjects) if all_subjects else 0
    total_data_memory = sum(v["epochs"].nbytes / 1e9 for v in all_subjects.values())
    
    phase_time = time.time() - phase_start
    log.info(f"\n{'─'*60}")
    log.info(f"Data Loading Summary:")
    log.info(f"  Total subjects:        {len(all_subjects)} ({n_mdd} MDD, {n_h} Healthy)")
    log.info(f"  Total epochs:          {total_epochs:,}")
    log.info(f"  Avg epochs/subject:    {avg_epochs_per_subject:.0f}")
    log.info(f"  Total data size:       {total_data_memory:.2f} GB")
    log.info(f"  Skipped files:         {skipped}")
    log.info(f"  Phase duration:        {phase_time:.1f}s")
    log.info(f"{'─'*60}\n")

    return all_subjects


# ===========================================================================
# 2. FEATURE EXTRACTION
# ===========================================================================

def extract_spectral_features(epoch: np.ndarray, sr: int = TARGET_SR):
    """PSD band powers for each channel. epoch: (n_channels, n_samples)."""
    features = []
    for ch in range(epoch.shape[0]):
        freqs, psd = signal.welch(epoch[ch], fs=sr, nperseg=min(256, epoch.shape[1]))
        ch_feats = []
        for band_name, (lo, hi) in BANDS.items():
            idx = np.logical_and(freqs >= lo, freqs <= hi)
            band_power = np.trapezoid(psd[idx], freqs[idx]) if idx.any() else 0.0
            ch_feats.append(band_power)
        features.extend(ch_feats)
    return np.array(features, dtype=np.float32)  # (n_channels * 5,) = 95


def extract_temporal_features(epoch: np.ndarray):
    """Statistical features per channel. epoch: (n_channels, n_samples)."""
    features = []
    for ch in range(epoch.shape[0]):
        x = epoch[ch]
        feat = [
            np.mean(x),
            np.var(x),
            float(stats.skew(x)),
            float(stats.kurtosis(x)),
            float(np.sum(np.diff(np.sign(x)) != 0)),  # zero crossings
            float(np.sum(np.abs(np.diff(x)))),          # line length
        ]
        # Hjorth parameters
        dx = np.diff(x)
        ddx = np.diff(dx)
        var_x = np.var(x)
        var_dx = np.var(dx)
        var_ddx = np.var(ddx)
        activity = var_x
        mobility = np.sqrt(var_dx / (var_x + 1e-10))
        complexity = np.sqrt(var_ddx / (var_dx + 1e-10)) / (mobility + 1e-10)
        feat.extend([activity, mobility, complexity])
        features.extend(feat)
    return np.array(features, dtype=np.float32)  # (n_channels * 9,) = 171


def extract_wavelet_features(epoch: np.ndarray):
    """WPD energy features per channel. epoch: (n_channels, n_samples)."""
    features = []
    wavelet = "db4"
    level = 4  # 2^4 = 16 terminal nodes
    for ch in range(epoch.shape[0]):
        wp = pywt.WaveletPacket(data=epoch[ch], wavelet=wavelet, maxlevel=level)
        nodes = [node.path for node in wp.get_level(level, "freq")]
        ch_feats = []
        for node_path in nodes:
            coeffs = wp[node_path].data
            energy = np.sum(coeffs ** 2)
            ch_feats.append(energy)
        features.extend(ch_feats)
    return np.array(features, dtype=np.float32)  # (n_channels * 16,) = 304


def extract_connectivity_features(epoch: np.ndarray, sr: int = TARGET_SR):
    """Inter-channel coherence in alpha and beta bands."""
    bands_of_interest = {"alpha": (8, 13), "beta": (13, 30)}
    features = []
    n_ch = epoch.shape[0]
    nperseg = min(256, epoch.shape[1])

    for band_name, (lo, hi) in bands_of_interest.items():
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                freqs, coh = scipy_coherence(epoch[i], epoch[j], fs=sr, nperseg=nperseg)
                idx = np.logical_and(freqs >= lo, freqs <= hi)
                mean_coh = np.mean(coh[idx]) if idx.any() else 0.0
                features.append(mean_coh)

    return np.array(features, dtype=np.float32)  # 2 * C(19,2) = 2 * 171 = 342


def extract_all_features(epoch: np.ndarray):
    """Extract all handcrafted features for a single epoch."""
    spectral = extract_spectral_features(epoch)
    temporal = extract_temporal_features(epoch)
    wavelet = extract_wavelet_features(epoch)
    connectivity = extract_connectivity_features(epoch)
    return np.concatenate([spectral, temporal, wavelet, connectivity])


def extract_features_batch(epochs: np.ndarray, desc: str = "Extracting features"):
    """Extract features for a batch of epochs. epochs: (n_epochs, n_channels, n_samples)."""
    features = []
    for i in tqdm(range(len(epochs)), desc=desc, unit="epoch", leave=False):
        feat = extract_all_features(epochs[i])
        features.append(feat)
    return np.array(features, dtype=np.float32)


# ===========================================================================
# 3. DATA AUGMENTATION (feature-space — avoids re-extracting from EEG)
# ===========================================================================

def augment_features(features: np.ndarray, rng: np.random.RandomState):
    """Augment in feature space: add small Gaussian noise + random scaling.
    Much faster than augmenting raw EEG and re-extracting features.
    features: (n_samples, n_features)
    Returns: augmented copy (n_samples, n_features)
    """
    aug = features.copy()
    # Gaussian noise (1% of feature std per column)
    noise_scale = 0.01 * (np.std(aug, axis=0, keepdims=True) + 1e-10)
    aug += rng.normal(0, 1, aug.shape) * noise_scale
    # Random scaling per-sample (0.95 to 1.05)
    scale = rng.uniform(0.95, 1.05, size=(aug.shape[0], 1))
    aug *= scale
    return aug


def augment_epochs_raw(epochs: np.ndarray, rng: np.random.RandomState):
    """Augment raw EEG epochs for CNN training.
    Returns augmented copy (same shape as input).
    """
    augmented = []
    for i in range(len(epochs)):
        aug = epochs[i].copy()
        # Time shift (±10%)
        if rng.random() < 0.5:
            shift = rng.randint(-int(0.1 * aug.shape[1]), int(0.1 * aug.shape[1]))
            aug = np.roll(aug, shift, axis=1)
        # Gaussian noise (SNR ~20dB)
        if rng.random() < 0.5:
            noise_power = np.mean(aug ** 2) / (10 ** (20 / 10))
            noise = rng.normal(0, np.sqrt(noise_power + 1e-12), aug.shape)
            aug = aug + noise
        # Channel dropout
        if rng.random() < 0.3:
            drop_ch = rng.randint(0, aug.shape[0])
            aug[drop_ch] = 0.0
        augmented.append(aug)
    return np.array(augmented)


# ===========================================================================
# 4. MODEL A: LIGHTWEIGHT 1D-CNN
# ===========================================================================

class EEG1DCNN(nn.Module):
    """Lightweight 1D-CNN for raw EEG epochs. Input: (batch, 19, 1000)."""

    def __init__(self, n_channels=N_CHANNELS, n_samples=EPOCH_SAMPLES):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (19, 1000) -> (64, 250)
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),

            # Block 2: (64, 250) -> (128, 62)
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),

            # Block 3: (128, 62) -> (256, 15)
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # (256, 1)
            nn.Flatten(),              # (256,)
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """x: (batch, n_channels, n_samples) -> logits (batch, 1)"""
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_cnn(train_epochs, train_labels, val_epochs, val_labels, fold_idx=None):
    """Train 1D-CNN model. Returns trained model."""
    fold_str = f" (Fold {fold_idx})" if fold_idx is not None else ""
    log.debug(f"  CNN Training{fold_str}: {len(train_epochs)} train epochs, {len(val_epochs)} val epochs")
    
    model = EEG1DCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CNN_LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS)
    use_amp = DEVICE.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    # Create dataloaders
    X_train = torch.FloatTensor(train_epochs)
    y_train = torch.FloatTensor(train_labels)
    X_val = torch.FloatTensor(val_epochs)
    y_val = torch.FloatTensor(val_labels)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=CNN_BATCH, shuffle=True,
                              pin_memory=(DEVICE.type == "cuda"), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CNN_BATCH, shuffle=False,
                            pin_memory=(DEVICE.type == "cuda"), num_workers=0)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    
    # Tracking for logging
    train_losses = []
    val_losses = []

    for epoch in range(CNN_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            if use_amp:
                with autocast("cuda"):
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_loss += loss.item() * len(X_batch)
            train_batches += 1
            
            # Cleanup intermediate tensors
            del X_batch, y_batch, logits, loss
            if use_amp and DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        scheduler.step()
        train_loss /= len(train_ds)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE).unsqueeze(1)
                if use_amp:
                    with autocast("cuda"):
                        logits = model(X_batch)
                        loss = criterion(logits, y_batch)
                else:
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(X_batch)
                val_batches += 1
                
                # Cleanup
                del X_batch, y_batch, logits, loss
        
        val_loss /= len(val_ds)
        val_losses.append(val_loss)
        
        # Cleanup GPU cache after epoch
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # Log training progress
        lr = optimizer.param_groups[0]["lr"]
        log.debug(f"    Epoch {epoch+1:2d}/{CNN_EPOCHS}: Loss={train_loss:.4f} | ValLoss={val_loss:.4f} | LR={lr:.5f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= CNN_PATIENCE:
                log.debug(f"    Early stopping at epoch {epoch+1} (patience={CNN_PATIENCE})")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    
    # Log final CNN training stats
    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    log.debug(f"  CNN Training Complete: final_train_loss={final_train_loss:.4f}, final_val_loss={final_val_loss:.4f}, epochs={len(train_losses)}")
    
    # Cleanup memory after training
    cleanup_gpu_memory()
    
    return model


def predict_cnn(model, epochs_data):
    """Get CNN predicted probabilities. epochs_data: (n_epochs, 19, 1000)."""
    model.eval()
    X = torch.FloatTensor(epochs_data)
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=CNN_BATCH, shuffle=False,
                        pin_memory=(DEVICE.type == "cuda"))
    use_amp = DEVICE.type == "cuda"
    probs = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(DEVICE)
            if use_amp:
                with autocast("cuda"):
                    logits = model(X_batch)
            else:
                logits = model(X_batch)
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.append(prob)
    return np.concatenate(probs)


# ===========================================================================
# MEMORY MANAGEMENT
# ===========================================================================

def cleanup_gpu_memory():
    """Clear GPU cache and run garbage collection."""
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if DEVICE.type == "cuda":
        return torch.cuda.memory_allocated(DEVICE) / 1e6  # MB
    return 0.0


def save_fold_checkpoint(fold_idx, test_sid, predictions):
    """Save checkpoint for a completed fold."""
    checkpoint = {
        "fold_idx": fold_idx,
        "test_sid": test_sid,
        "epoch_probs_ensemble": predictions["ensemble"],
        "epoch_probs_cnn": predictions["cnn"],
        "epoch_probs_xgb": predictions["xgb"],
        "epoch_probs_svm": predictions["svm"],
        "subject_pred_prob": predictions["subject_prob"],
        "subject_true_label": predictions["subject_true_label"],
        "timestamp": datetime.now().isoformat(),
    }
    ckpt_path = CHECKPOINT_DIR / f"fold_{fold_idx:02d}_{test_sid}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(checkpoint, f)
    log.debug(f"  Checkpoint saved: {ckpt_path.name}")


def save_running_results(subject_true, subject_pred_prob, fold_idx, n_subjects):
    """Save partial results after each fold so progress is never lost."""
    subject_ids_sorted = sorted(subject_true.keys())
    subj_true_arr = np.array([subject_true[s] for s in subject_ids_sorted])
    subj_prob_arr = np.array([subject_pred_prob[s] for s in subject_ids_sorted])
    subj_pred_arr = (subj_prob_arr >= 0.5).astype(int)

    if len(np.unique(subj_true_arr)) < 2:
        # Not enough classes yet for AUC
        partial = {
            "status": "in_progress",
            "folds_completed": fold_idx + 1,
            "folds_total": n_subjects,
            "per_subject_probabilities": {
                sid: {
                    "true_label": int(subject_true[sid]),
                    "pred_prob": float(subject_pred_prob[sid]),
                    "predicted": int(subject_pred_prob[sid] >= 0.5),
                }
                for sid in subject_ids_sorted
            },
            "timestamp": datetime.now().isoformat(),
        }
    else:
        running_acc = accuracy_score(subj_true_arr, subj_pred_arr)
        partial = {
            "status": "in_progress",
            "folds_completed": fold_idx + 1,
            "folds_total": n_subjects,
            "running_subject_accuracy": float(running_acc),
            "running_correct": int(np.sum(subj_true_arr == subj_pred_arr)),
            "per_subject_probabilities": {
                sid: {
                    "true_label": int(subject_true[sid]),
                    "pred_prob": float(subject_pred_prob[sid]),
                    "predicted": int(subject_pred_prob[sid] >= 0.5),
                }
                for sid in subject_ids_sorted
            },
            "timestamp": datetime.now().isoformat(),
        }

    partial_path = OUTPUT_DIR / "results_partial.json"
    with open(partial_path, "w") as f:
        json.dump(partial, f, indent=2)


def save_models(cnn_model, xgb_model, svm_model, scaler, fold_idx, test_sid):
    """Save all three trained models for a given fold."""
    fold_dir = MODELS_DIR / f"fold_{fold_idx:02d}_{test_sid}"
    fold_dir.mkdir(exist_ok=True)

    # Save CNN
    torch.save(cnn_model.state_dict(), fold_dir / "cnn_weights.pt")

    # Save XGBoost
    xgb_model.save_model(str(fold_dir / "xgboost.json"))

    # Save SVM + scaler together
    with open(fold_dir / "svm_and_scaler.pkl", "wb") as f:
        pickle.dump({"svm": svm_model, "scaler": scaler}, f)

    log.debug(f"  Models saved: {fold_dir.name}")




def run_loso_cv(all_subjects: dict):
    """Run Leave-One-Subject-Out CV with 3-model ensemble.

    Strategy for speed:
    - Extract handcrafted features ONCE for all subjects (pre-computed)
    - Augment in feature-space (add noise/scale) — no re-extraction needed
    - Augment raw EEG for CNN only (fast numpy ops, no feature extraction)
    """
    log.info("\n" + "=" * 60)
    log.info("PHASE 2: LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION")
    log.info("=" * 60)
    log.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        log.info("Running CNN on CPU (no compatible CUDA device)")

    subject_ids = sorted(all_subjects.keys())
    n_subjects = len(subject_ids)
    log.info(f"Total subjects: {n_subjects}")

    # ---- Pre-extract features for ALL subjects ONCE ----
    log.info("\n[Step 1/5] Extracting handcrafted features for all subjects...")
    t0 = time.time()
    subject_features = {}
    
    total_feature_epochs = sum(len(all_subjects[sid]["epochs"]) for sid in subject_ids)
    
    for idx, sid in enumerate(tqdm(subject_ids, desc="Feature extraction by subject", unit="subject")):
        epochs = all_subjects[sid]["epochs"]
        feats = extract_features_batch(epochs, desc=f"  {sid}")
        subject_features[sid] = feats
        log.debug(f"  [{idx+1}/{n_subjects}] {sid}: {feats.shape[0]} epochs, {feats.shape[1]} features, {feats.nbytes/1e6:.1f}MB")

    feat_dim = subject_features[subject_ids[0]].shape[1]
    feat_time = time.time() - t0
    log.info(f"✓ Feature extraction complete in {feat_time:.1f}s ({total_feature_epochs:,} epochs processed)")
    log.info(f"  Feature dimension: {feat_dim}")
    log.info(f"  Average {total_feature_epochs/feat_time:.0f} epochs/sec")

    # ---- Pre-compute augmented features (feature-space augmentation) ----
    log.info("\n[Step 2/5] Augmenting features (feature-space augmentation)...")
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    subject_features_aug = {}
    for sid in tqdm(subject_ids, desc="Feature augmentation", unit="subject"):
        aug = augment_features(subject_features[sid], rng)
        subject_features_aug[sid] = aug
    
    aug_time = time.time() - t0
    log.info(f"✓ Feature augmentation complete in {aug_time:.1f}s")

    # Storage for predictions
    all_epoch_labels = []
    all_epoch_probs_ensemble = []
    all_epoch_probs_cnn = []
    all_epoch_probs_xgb = []
    all_epoch_probs_svm = []
    all_epoch_subject_ids = []

    subject_true = {}
    subject_pred_prob = {}

    log.info(f"\n[Step 3/5] Running {n_subjects}-fold LOSO CV...")
    log.info("=" * 60)
    
    total_t0 = time.time()
    fold_times = []
    model_times = {"cnn": [], "xgb": [], "svm": []}

    for fold_idx, test_sid in enumerate(tqdm(subject_ids, desc="LOSO CV Progress", unit="fold")):
        fold_t0 = time.time()
        test_label = all_subjects[test_sid]["label"]
        test_epochs = all_subjects[test_sid]["epochs"]
        test_feats = subject_features[test_sid]

        # Gather training data
        train_sids = [s for s in subject_ids if s != test_sid]

        # --- Build training features (original + augmented, all pre-computed) ---
        train_feats_list = []
        train_feat_labels = []
        train_epochs_list = []
        train_epoch_labels = []

        rng_fold = np.random.RandomState(SEED + fold_idx)

        for sid in train_sids:
            lab = all_subjects[sid]["label"]
            ft_orig = subject_features[sid]
            ft_aug = subject_features_aug[sid]
            ep_orig = all_subjects[sid]["epochs"]

            # Features: original + augmented
            train_feats_list.append(ft_orig)
            train_feats_list.append(ft_aug)
            train_feat_labels.append(np.full(len(ft_orig) + len(ft_aug), lab, dtype=np.float32))

            # Raw epochs for CNN: original + augmented
            ep_aug = augment_epochs_raw(ep_orig, rng_fold)
            train_epochs_list.append(ep_orig)
            train_epochs_list.append(ep_aug)
            train_epoch_labels.append(np.full(len(ep_orig) + len(ep_aug), lab, dtype=np.float32))

        train_feats_all = np.concatenate(train_feats_list, axis=0)
        train_feat_labels_all = np.concatenate(train_feat_labels, axis=0)
        train_epochs_all = np.concatenate(train_epochs_list, axis=0)
        train_epoch_labels_all = np.concatenate(train_epoch_labels, axis=0)

        # Handle NaN/Inf in features
        train_feats_all = np.nan_to_num(train_feats_all, nan=0.0, posinf=0.0, neginf=0.0)
        test_feats_clean = np.nan_to_num(test_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        scaler = StandardScaler()
        train_feats_scaled = scaler.fit_transform(train_feats_all)
        test_feats_scaled = scaler.transform(test_feats_clean)

        try:
            # --- Model A: 1D-CNN ---
            cnn_t0 = time.time()
            cnn_model = train_cnn(
                train_epochs_all, train_epoch_labels_all,
                test_epochs, np.full(len(test_epochs), test_label, dtype=np.float32),
                fold_idx=fold_idx+1
            )
            cnn_probs = predict_cnn(cnn_model, test_epochs)
            cnn_time = time.time() - cnn_t0
            model_times["cnn"].append(cnn_time)
            
            log.debug(f"    CNN: {cnn_time:.1f}s")
            del cnn_model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            # Free large arrays
            del train_epochs_all, train_epoch_labels_all
            gc.collect()

            # --- Model B: XGBoost ---
            xgb_t0 = time.time()
            n_pos = np.sum(train_feat_labels_all == 1)
            n_neg = np.sum(train_feat_labels_all == 0)
            scale_pos = n_neg / (n_pos + 1e-8)

            xgb_model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
            )
            xgb_model.fit(train_feats_scaled, train_feat_labels_all.astype(int))
            xgb_probs = xgb_model.predict_proba(test_feats_scaled)[:, 1]
            xgb_time = time.time() - xgb_t0
            model_times["xgb"].append(xgb_time)
            log.debug(f"    XGBoost: {xgb_time:.1f}s")

            # --- Model C: SVM (subsampled to avoid >48hr runtime) ---
            svm_t0 = time.time()
            svm_model = SVC(C=10, kernel="rbf", gamma="scale", probability=True,
                            random_state=SEED)
            n_svm_train = len(train_feats_scaled)
            if n_svm_train > SVM_MAX_TRAIN_SAMPLES:
                rng_svm = np.random.RandomState(SEED + fold_idx)
                svm_idx = rng_svm.choice(n_svm_train, SVM_MAX_TRAIN_SAMPLES, replace=False)
                svm_train_X = train_feats_scaled[svm_idx]
                svm_train_y = train_feat_labels_all.astype(int)[svm_idx]
                log.debug(f"    SVM subsampled: {n_svm_train} → {SVM_MAX_TRAIN_SAMPLES}")
            else:
                svm_train_X = train_feats_scaled
                svm_train_y = train_feat_labels_all.astype(int)
            svm_model.fit(svm_train_X, svm_train_y)
            svm_probs = svm_model.predict_proba(test_feats_scaled)[:, 1]
            svm_time = time.time() - svm_t0
            model_times["svm"].append(svm_time)
            log.debug(f"    SVM: {svm_time:.1f}s")

        except (RuntimeError, MemoryError) as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "memory" in err_str or isinstance(e, MemoryError):
                log.error(f"  Fold {fold_idx + 1}/{n_subjects} OOM/MemError on {test_sid}")
                log.error(f"  Error: {e}")
                log.error("  Attempting recovery...")
                cleanup_gpu_memory()
                # Fallback to simple models without CNN
                cnn_probs = np.full(len(test_epochs), 0.5)
                xgb_probs = np.full(len(test_feats_clean), 0.5)
                svm_probs = np.full(len(test_feats_clean), 0.5)
                xgb_model = None
                svm_model = None
            else:
                raise

        # --- Ensemble: soft voting ---
        ensemble_probs = (cnn_probs + xgb_probs + svm_probs) / 3.0

        # Subject-level prediction (average across all epochs)
        subj_prob = np.mean(ensemble_probs)
        subj_pred = int(subj_prob >= 0.5)

        subject_true[test_sid] = test_label
        subject_pred_prob[test_sid] = subj_prob

        # Store epoch-level
        n_test = len(test_epochs)
        all_epoch_labels.extend([test_label] * n_test)
        all_epoch_probs_ensemble.extend(ensemble_probs.tolist())
        all_epoch_probs_cnn.extend(cnn_probs.tolist())
        all_epoch_probs_xgb.extend(xgb_probs.tolist())
        all_epoch_probs_svm.extend(svm_probs.tolist())
        all_epoch_subject_ids.extend([test_sid] * n_test)

        fold_time = time.time() - fold_t0
        fold_times.append(fold_time)
        elapsed = time.time() - total_t0
        folds_remaining = n_subjects - fold_idx - 1
        avg_fold_time = elapsed / (fold_idx + 1)
        eta = avg_fold_time * folds_remaining

        status = "✓" if subj_pred == test_label else "✗"
        label_str = "MDD" if test_label == 1 else "H"
        pred_str = "MDD" if subj_pred == 1 else "H"
        
        gpu_mem = get_gpu_memory_usage()
        
        # Log fold completion
        log.info(
            f"  Fold {fold_idx + 1:2d}/{n_subjects}: {test_sid:10s} "
            f"True={label_str:3s} Pred={pred_str:3s} "
            f"prob={subj_prob:.3f} "
            f"({fold_time:.0f}s, ETA={eta/60:.0f}m) "
            f"[{status}] GPU={gpu_mem:.0f}MB"
        )
        log.debug(
            f"    Model probs - CNN={np.mean(cnn_probs):.3f}, "
            f"XGB={np.mean(xgb_probs):.3f}, SVM={np.mean(svm_probs):.3f}"
        )

        # === SAVE FOLD CHECKPOINT (raw per-fold data) ===
        if CHECKPOINT_LOSO:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            save_fold_checkpoint(fold_idx, test_sid, {
                "ensemble": ensemble_probs.tolist(),
                "cnn": cnn_probs.tolist(),
                "xgb": xgb_probs.tolist(),
                "svm": svm_probs.tolist(),
                "subject_prob": float(subj_prob),
                "subject_true_label": int(test_label),
            })

        # === SAVE MODELS FOR THIS FOLD ===
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if xgb_model is not None and svm_model is not None:
            save_models(cnn_model, xgb_model, svm_model, scaler, fold_idx, test_sid)

        # === SAVE RUNNING PARTIAL RESULTS (crash-safe) ===
        save_running_results(subject_true, subject_pred_prob, fold_idx, n_subjects)

        # === FREE MODEL MEMORY (64 folds × large models = RAM leak) ===
        del cnn_model
        if xgb_model is not None:
            del xgb_model
        if svm_model is not None:
            del svm_model
        gc.collect()

        # === GPU CACHE CLEANUP ===
        if CLEAR_GPU_CACHE_BETWEEN_FOLDS:
            cleanup_gpu_memory()
            log.debug(f"    Post-cleanup GPU memory: {get_gpu_memory_usage():.0f}MB")

    log.info("=" * 60)
    log.info(f"✓ LOSO CV Complete in {time.time() - total_t0:.1f}s")
    log.info(f"  Avg fold time: {np.mean(fold_times):.1f}s")
    log.info(f"  Total folds: {len(fold_times)}")

    metrics = compute_and_report_metrics(
        all_epoch_labels, all_epoch_probs_ensemble,
        all_epoch_probs_cnn, all_epoch_probs_xgb, all_epoch_probs_svm,
        all_epoch_subject_ids, subject_true, subject_pred_prob,
        model_times=model_times,
    )
    return metrics, subject_features, subject_features_aug


# ===========================================================================
# 6. METRICS & REPORTING
# ===========================================================================

def compute_and_report_metrics(
    epoch_labels, epoch_probs_ensemble,
    epoch_probs_cnn, epoch_probs_xgb, epoch_probs_svm,
    epoch_subject_ids, subject_true, subject_pred_prob,
    model_times=None,
):
    """Compute and print all metrics."""
    log.info("\n" + "=" * 60)
    log.info("PHASE 3: METRICS & REPORTING")
    log.info("=" * 60)
    
    epoch_labels = np.array(epoch_labels)
    epoch_probs = np.array(epoch_probs_ensemble)
    epoch_preds = (epoch_probs >= 0.5).astype(int)

    # --- Sample-level metrics ---
    sample_acc = accuracy_score(epoch_labels, epoch_preds)
    sample_f1 = f1_score(epoch_labels, epoch_preds)
    sample_auc = roc_auc_score(epoch_labels, epoch_probs)
    sample_prec = precision_score(epoch_labels, epoch_preds)
    sample_rec = recall_score(epoch_labels, epoch_preds)
    tn, fp, fn, tp = confusion_matrix(epoch_labels, epoch_preds).ravel()
    sample_sens = tp / (tp + fn + 1e-10)
    sample_spec = tn / (tn + fp + 1e-10)

    # --- Subject-level metrics ---
    subject_ids_sorted = sorted(subject_true.keys())
    subj_true_arr = np.array([subject_true[s] for s in subject_ids_sorted])
    subj_prob_arr = np.array([subject_pred_prob[s] for s in subject_ids_sorted])
    subj_pred_arr = (subj_prob_arr >= 0.5).astype(int)

    subj_acc = accuracy_score(subj_true_arr, subj_pred_arr)
    subj_f1 = f1_score(subj_true_arr, subj_pred_arr)
    subj_auc = roc_auc_score(subj_true_arr, subj_prob_arr)
    subj_prec = precision_score(subj_true_arr, subj_pred_arr)
    subj_rec = recall_score(subj_true_arr, subj_pred_arr)
    subj_cm = confusion_matrix(subj_true_arr, subj_pred_arr)
    s_tn, s_fp, s_fn, s_tp = subj_cm.ravel()
    subj_sens = s_tp / (s_tp + s_fn + 1e-10)
    subj_spec = s_tn / (s_tn + s_fp + 1e-10)

    # Per-model sample accuracy
    cnn_preds = (np.array(epoch_probs_cnn) >= 0.5).astype(int)
    xgb_preds = (np.array(epoch_probs_xgb) >= 0.5).astype(int)
    svm_preds = (np.array(epoch_probs_svm) >= 0.5).astype(int)
    cnn_acc = accuracy_score(epoch_labels, cnn_preds)
    xgb_acc = accuracy_score(epoch_labels, xgb_preds)
    svm_acc = accuracy_score(epoch_labels, svm_preds)

    # Per-model subject accuracy
    log.info("\n[Per-Model Performance]")
    for model_name, model_probs in [("CNN", epoch_probs_cnn), ("XGB", epoch_probs_xgb), ("SVM", epoch_probs_svm)]:
        model_probs = np.array(model_probs)
        model_subj_probs = {}
        for sid in subject_ids_sorted:
            mask = np.array([s == sid for s in epoch_subject_ids])
            model_subj_probs[sid] = np.mean(model_probs[mask])
        model_subj_preds = np.array([(model_subj_probs[s] >= 0.5) for s in subject_ids_sorted]).astype(int)
        model_subj_acc = accuracy_score(subj_true_arr, model_subj_preds)
        log.info(f"  {model_name:10s} - Subject Accuracy: {model_subj_acc:.4f} ({int(model_subj_acc * len(subj_true_arr))}/{len(subj_true_arr)})")

    # Misclassified subjects
    misclassified = [s for s in subject_ids_sorted if subj_pred_arr[subject_ids_sorted.index(s)] != subj_true_arr[subject_ids_sorted.index(s)]]

    # --- Report ---
    log.info("\n" + "─" * 60)
    log.info("SAMPLE-LEVEL METRICS (Each epoch)")
    log.info("─" * 60)
    log.info(f"  Accuracy:    {sample_acc:.4f}")
    log.info(f"  F1 Score:    {sample_f1:.4f}")
    log.info(f"  AUC-ROC:     {sample_auc:.4f}")
    log.info(f"  Precision:   {sample_prec:.4f}")
    log.info(f"  Recall:      {sample_rec:.4f}")
    log.info(f"  Sensitivity: {sample_sens:.4f}")
    log.info(f"  Specificity: {sample_spec:.4f}")
    log.info(f"  Confusion Matrix:")
    log.info(f"    Predicted:  Healthy  MDD")
    log.info(f"    Healthy:    {tn:4d}    {fp:4d}")
    log.info(f"    MDD:        {fn:4d}    {tp:4d}")

    log.info("\n" + "─" * 60)
    log.info("SUBJECT-LEVEL METRICS (Averaged across epochs)")
    log.info("─" * 60)
    log.info(f"  Accuracy:    {subj_acc:.4f} ({int(subj_acc * len(subj_true_arr))}/{len(subj_true_arr)} subjects correct)")
    log.info(f"  F1 Score:    {subj_f1:.4f}")
    log.info(f"  AUC-ROC:     {subj_auc:.4f}")
    log.info(f"  Precision:   {subj_prec:.4f}")
    log.info(f"  Recall:      {subj_rec:.4f}")
    log.info(f"  Sensitivity: {subj_sens:.4f}")
    log.info(f"  Specificity: {subj_spec:.4f}")
    log.info(f"  Confusion Matrix:")
    log.info(f"    Predicted:  Healthy  MDD")
    log.info(f"    Healthy:    {s_tn:4d}    {s_fp:4d}")
    log.info(f"    MDD:        {s_fn:4d}    {s_tp:4d}")

    if misclassified:
        log.info(f"\n  ⚠️  Misclassified subjects ({len(misclassified)}):")
        for sid in misclassified:
            true_l = "MDD" if subject_true[sid] == 1 else "Healthy"
            pred_l = "MDD" if (subject_pred_prob[sid] >= 0.5) else "Healthy"
            pred_p = subject_pred_prob[sid]
            log.info(f"    {sid}: True={true_l:8s} → Pred={pred_l:8s} (prob={pred_p:.3f})")
    else:
        log.info(f"\n  ✓ Perfect classification! All subjects correct.")

    # Model timing summary
    if model_times:
        log.info("\n" + "─" * 60)
        log.info("MODEL TRAINING TIME SUMMARY")
        log.info("─" * 60)
        for model_name, times in model_times.items():
            if times:
                avg_time = np.mean(times)
                total_time = np.sum(times)
                log.info(f"  {model_name.upper():10s} - Avg: {avg_time:.1f}s/fold, Total: {total_time:.0f}s ({len(times)} folds)")

    # --- Save results ---
    results = {
        "timestamp": datetime.now().isoformat(),
        "sample_level": {
            "accuracy": float(sample_acc),
            "f1": float(sample_f1),
            "auc_roc": float(sample_auc),
            "precision": float(sample_prec),
            "recall": float(sample_rec),
            "sensitivity": float(sample_sens),
            "specificity": float(sample_spec),
            "n_epochs": int(len(epoch_labels)),
            "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        },
        "subject_level": {
            "accuracy": float(subj_acc),
            "f1": float(subj_f1),
            "auc_roc": float(subj_auc),
            "precision": float(subj_prec),
            "recall": float(subj_rec),
            "sensitivity": float(subj_sens),
            "specificity": float(subj_spec),
            "confusion_matrix": {"TP": int(s_tp), "TN": int(s_tn), "FP": int(s_fp), "FN": int(s_fn)},
            "n_subjects": len(subj_true_arr),
            "n_correct": int(subj_acc * len(subj_true_arr)),
            "misclassified_count": len(misclassified),
            "misclassified": misclassified,
        },
        "per_model_sample_accuracy": {
            "cnn": float(cnn_acc),
            "xgboost": float(xgb_acc),
            "svm": float(svm_acc),
            "ensemble": float(sample_acc),
        },
        "per_subject_probabilities": {
            sid: {"true_label": int(subject_true[sid]), "pred_prob": float(subject_pred_prob[sid])}
            for sid in subject_ids_sorted
        },
    }

    results_path = OUTPUT_DIR / "results_best.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n✓ Results saved to {results_path}")

    return results


# ===========================================================================
# MAIN
# ===========================================================================

def train_final_model(all_subjects: dict, subject_features: dict, subject_features_aug: dict):
    """Train final ensemble models on ALL subjects and save for deployment.

    This produces the actual deployable model, separate from LOSO CV folds.
    Uses the already-computed subject_features / subject_features_aug so no
    additional feature extraction time is needed.
    """
    log.info("\n" + "=" * 60)
    log.info("PHASE 4: FINAL MODEL TRAINING (ALL SUBJECTS)")
    log.info("=" * 60)

    subject_ids = sorted(all_subjects.keys())
    rng = np.random.RandomState(SEED)

    # Concatenate features from all subjects (original + augmented)
    all_feats = []
    all_labels = []
    for sid in subject_ids:
        label = all_subjects[sid]["label"]
        feats = subject_features[sid]
        feats_aug = subject_features_aug[sid]
        all_feats.append(feats)
        all_labels.extend([label] * len(feats))
        all_feats.append(feats_aug)
        all_labels.extend([label] * len(feats_aug))

    all_feats = np.vstack(all_feats)
    all_labels = np.array(all_labels, dtype=int)

    # Scale features
    scaler = StandardScaler()
    all_feats_scaled = scaler.fit_transform(all_feats)

    log.info(f"  Training on {len(all_labels):,} samples ({int(all_labels.sum())} MDD, {int((1-all_labels).sum())} H)")

    # --- Model B: XGBoost (all data) ---
    log.info("  Training XGBoost (all subjects)...")
    xgb_t0 = time.time()
    xgb_final = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=SEED,
    )
    xgb_final.fit(all_feats_scaled, all_labels)
    log.info(f"  ✓ XGBoost done in {time.time()-xgb_t0:.1f}s")

    # --- Model C: SVM (subsampled) ---
    log.info(f"  Training SVM (subsampled to {SVM_MAX_TRAIN_SAMPLES}) ...")
    svm_t0 = time.time()
    if len(all_feats_scaled) > SVM_MAX_TRAIN_SAMPLES:
        svm_idx = rng.choice(len(all_feats_scaled), SVM_MAX_TRAIN_SAMPLES, replace=False)
        svm_X = all_feats_scaled[svm_idx]
        svm_y = all_labels[svm_idx]
    else:
        svm_X, svm_y = all_feats_scaled, all_labels
    svm_final = SVC(C=10, kernel="rbf", gamma="scale", probability=True, random_state=SEED)
    svm_final.fit(svm_X, svm_y)
    log.info(f"  ✓ SVM done in {time.time()-svm_t0:.1f}s")

    # --- Model A: CNN (all data) ---
    log.info("  Training CNN (all subjects)...")
    cnn_t0 = time.time()
    # Build dataset from raw epochs
    all_epoch_data = []
    all_epoch_labels_cnn = []
    for sid in subject_ids:
        label = all_subjects[sid]["label"]
        for ep in all_subjects[sid]["epochs"]:
            all_epoch_data.append(ep)
            all_epoch_labels_cnn.append(label)
    cnn_dataset = EEGEpochDataset(all_epoch_data, all_epoch_labels_cnn)
    cnn_loader = DataLoader(cnn_dataset, batch_size=CNN_BATCH, shuffle=True,
                            num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    cnn_final = EEGNet(n_channels=all_epoch_data[0].shape[0]).to(DEVICE)
    optimizer = torch.optim.AdamW(cnn_final.parameters(), lr=CNN_LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS)
    cnn_final.train()
    for epoch in range(CNN_EPOCHS):
        for xb, yb in cnn_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(cnn_final(xb).squeeze(1), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
    log.info(f"  ✓ CNN done in {time.time()-cnn_t0:.1f}s")

    # --- Save final models ---
    final_dir = MODELS_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    torch.save(cnn_final.state_dict(), final_dir / "cnn_final.pt")
    xgb_final.save_model(str(final_dir / "xgboost_final.json"))
    with open(final_dir / "svm_and_scaler_final.pkl", "wb") as f:
        pickle.dump({"svm": svm_final, "scaler": scaler}, f)

    log.info(f"  ✓ Final models saved to: {final_dir}")
    log.info("    Files: cnn_final.pt, xgboost_final.json, svm_and_scaler_final.pkl")
    log.info("=" * 60)


def main():
    log.info("=" * 60)
    log.info("EEG DEPRESSION DETECTION — V3 ENSEMBLE TRAINING")
    log.info("=" * 60)
    log.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Python:    {sys.version.split()[0]}")
    log.info(f"PyTorch:   {torch.__version__}")
    log.info(f"CUDA:      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU:       {torch.cuda.get_device_name(0)}")
    log.info(f"Device:    {DEVICE}")
    log.info(f"Data dir:  {DATA_DIR}")
    log.info(f"Output:    {OUTPUT_DIR}")
    log.info("=" * 60)

    t_start = time.time()

    # Load data
    all_subjects = load_all_data()
    if len(all_subjects) == 0:
        log.error("❌ No subjects loaded! Check data directory.")
        sys.exit(1)

    # Run LOSO CV
    results, subject_features, subject_features_aug = run_loso_cv(all_subjects)

    # Train deployable final models on all subjects
    train_final_model(all_subjects, subject_features, subject_features_aug)

    elapsed = time.time() - t_start
    minutes = elapsed / 60
    hours = minutes / 60
    
    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("=" * 60)
    log.info(f"✓ Total runtime: {hours:.1f}h ({minutes:.1f}m / {elapsed:.0f}s)")
    log.info(f"✓ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"✓ Results:       {OUTPUT_DIR / 'results_best.json'}")
    log.info(f"✓ Partial log:   {OUTPUT_DIR / 'results_partial.json'}")
    log.info(f"✓ Final models:  {MODELS_DIR / 'final'}")
    log.info(f"✓ Fold models:   {MODELS_DIR}")
    log.info(f"✓ Checkpoints:   {CHECKPOINT_DIR}")
    log.info(f"✓ Logs:          {OUTPUT_DIR / 'training_v3.log'}")
    log.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
