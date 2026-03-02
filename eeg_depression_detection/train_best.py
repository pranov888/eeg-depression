#!/usr/bin/env python3
"""
EEG Depression Detection — V3 Best-of-Best Ensemble Training
=============================================================
Self-contained script: loads EDF files, extracts multi-domain features,
trains a 3-model ensemble (1D-CNN + XGBoost + SVM) with LOSO CV,
and reports sample-level + subject-level metrics.

Usage:
    conda run -n eeg_dep python train_best.py
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import mne
import pywt
from scipy import signal, stats
from scipy.signal import coherence as scipy_coherence

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
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
    log.info("LOADING DATA")
    log.info("=" * 60)

    edf_files = sorted(DATA_DIR.glob("*.edf"))
    log.info(f"Found {len(edf_files)} EDF files in {DATA_DIR}")

    # subject_id -> {label, epochs: list of (n_epochs, n_channels, epoch_samples)}
    subject_data = defaultdict(lambda: {"label": None, "epochs": [], "conditions": []})
    skipped = 0

    for fpath in edf_files:
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
            log.info(f"  {fpath.name}: {len(epochs)} epochs")
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

    log.info(f"\nLoaded {len(all_subjects)} subjects: {n_mdd} MDD, {n_h} Healthy")
    log.info(f"Total epochs: {total_epochs}")
    log.info(f"Skipped files: {skipped}")

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
        # Basic stats
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
    """Inter-channel coherence in alpha and beta bands.
    epoch: (n_channels, n_samples). Returns upper-triangle coherence values.
    """
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


def extract_features_batch(epochs: np.ndarray):
    """Extract features for a batch of epochs. epochs: (n_epochs, n_channels, n_samples)."""
    features = []
    for i in range(len(epochs)):
        feat = extract_all_features(epochs[i])
        features.append(feat)
    return np.array(features, dtype=np.float32)


# ===========================================================================
# 3. DATA AUGMENTATION
# ===========================================================================

def augment_epoch(epoch: np.ndarray, rng: np.random.RandomState):
    """Apply random augmentations to a single epoch (n_channels, n_samples)."""
    aug = epoch.copy()

    # Time shift (±10%)
    if rng.random() < 0.5:
        shift = rng.randint(-int(0.1 * aug.shape[1]), int(0.1 * aug.shape[1]))
        aug = np.roll(aug, shift, axis=1)

    # Gaussian noise injection (SNR ~20dB)
    if rng.random() < 0.5:
        noise_power = np.mean(aug ** 2) / (10 ** (20 / 10))
        noise = rng.normal(0, np.sqrt(noise_power + 1e-12), aug.shape)
        aug = aug + noise

    # Channel dropout
    if rng.random() < 0.3:
        drop_ch = rng.randint(0, aug.shape[0])
        aug[drop_ch] = 0.0

    return aug


def augment_epochs(epochs: np.ndarray, n_augmented: int = 1):
    """Create augmented copies of epochs.
    Returns original + augmented epochs concatenated.
    """
    rng = np.random.RandomState(SEED)
    augmented = []
    for _ in range(n_augmented):
        for i in range(len(epochs)):
            augmented.append(augment_epoch(epochs[i], rng))
    return np.concatenate([epochs, np.array(augmented)], axis=0)


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


def train_cnn(train_epochs, train_labels, val_epochs, val_labels):
    """Train 1D-CNN model. Returns trained model."""
    model = EEG1DCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CNN_LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS)
    scaler = GradScaler("cuda") if DEVICE.type == "cuda" else None

    # Create dataloaders
    X_train = torch.FloatTensor(train_epochs)
    y_train = torch.FloatTensor(train_labels)
    X_val = torch.FloatTensor(val_epochs)
    y_val = torch.FloatTensor(val_labels)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=CNN_BATCH, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CNN_BATCH, shuffle=False,
                            pin_memory=True, num_workers=0)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(CNN_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            if scaler is not None:
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

        scheduler.step()
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE).unsqueeze(1)
                if scaler is not None:
                    with autocast("cuda"):
                        logits = model(X_batch)
                        loss = criterion(logits, y_batch)
                else:
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(X_batch)
        val_loss /= len(val_ds)

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= CNN_PATIENCE:
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    return model


def predict_cnn(model, epochs_data):
    """Get CNN predicted probabilities. epochs_data: (n_epochs, 19, 1000)."""
    model.eval()
    X = torch.FloatTensor(epochs_data)
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=CNN_BATCH, shuffle=False, pin_memory=True)
    probs = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(DEVICE)
            if DEVICE.type == "cuda":
                with autocast("cuda"):
                    logits = model(X_batch)
            else:
                logits = model(X_batch)
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.append(prob)
    return np.concatenate(probs)


# ===========================================================================
# 5. LOSO CROSS-VALIDATION WITH ENSEMBLE
# ===========================================================================

def run_loso_cv(all_subjects: dict):
    """Run Leave-One-Subject-Out CV with 3-model ensemble."""
    log.info("\n" + "=" * 60)
    log.info("LOSO CROSS-VALIDATION — ENSEMBLE (CNN + XGBoost + SVM)")
    log.info("=" * 60)
    log.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    subject_ids = sorted(all_subjects.keys())
    n_subjects = len(subject_ids)

    # Pre-extract features for all subjects (to avoid repeated computation)
    log.info("\nExtracting handcrafted features for all subjects...")
    t0 = time.time()
    subject_features = {}
    for sid in subject_ids:
        epochs = all_subjects[sid]["epochs"]
        feats = extract_features_batch(epochs)
        subject_features[sid] = feats
        log.info(f"  {sid}: {feats.shape[0]} epochs, {feats.shape[1]} features")

    feat_dim = subject_features[subject_ids[0]].shape[1]
    log.info(f"Feature extraction done in {time.time() - t0:.1f}s. Feature dim: {feat_dim}")

    # Storage for predictions
    all_epoch_labels = []
    all_epoch_probs_ensemble = []
    all_epoch_probs_cnn = []
    all_epoch_probs_xgb = []
    all_epoch_probs_svm = []
    all_epoch_subject_ids = []

    subject_true = {}
    subject_pred_prob = {}

    for fold_idx, test_sid in enumerate(subject_ids):
        test_label = all_subjects[test_sid]["label"]
        test_epochs = all_subjects[test_sid]["epochs"]
        test_feats = subject_features[test_sid]

        # Gather training data
        train_sids = [s for s in subject_ids if s != test_sid]
        train_epochs_list = []
        train_labels_list = []
        train_feats_list = []
        train_feat_labels = []

        for sid in train_sids:
            ep = all_subjects[sid]["epochs"]
            lab = all_subjects[sid]["label"]
            ft = subject_features[sid]
            n_ep = ep.shape[0]

            # Augment training data
            ep_aug = augment_epochs(ep, n_augmented=1)  # 2x data
            ft_aug = extract_features_batch(ep_aug[n_ep:])  # Features for augmented only
            ft_combined = np.concatenate([ft, ft_aug], axis=0)

            train_epochs_list.append(ep_aug)
            train_labels_list.append(np.full(len(ep_aug), lab, dtype=np.float32))
            train_feats_list.append(ft_combined)
            train_feat_labels.append(np.full(len(ft_combined), lab, dtype=np.float32))

        train_epochs_all = np.concatenate(train_epochs_list, axis=0)
        train_labels_all = np.concatenate(train_labels_list, axis=0)
        train_feats_all = np.concatenate(train_feats_list, axis=0)
        train_feat_labels_all = np.concatenate(train_feat_labels, axis=0)

        # Handle NaN/Inf in features
        train_feats_all = np.nan_to_num(train_feats_all, nan=0.0, posinf=0.0, neginf=0.0)
        test_feats_clean = np.nan_to_num(test_feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        scaler = StandardScaler()
        train_feats_scaled = scaler.fit_transform(train_feats_all)
        test_feats_scaled = scaler.transform(test_feats_clean)

        # --- Model A: 1D-CNN ---
        cnn_model = train_cnn(
            train_epochs_all, train_labels_all,
            test_epochs, np.full(len(test_epochs), test_label, dtype=np.float32),
        )
        cnn_probs = predict_cnn(cnn_model, test_epochs)
        del cnn_model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # --- Model B: XGBoost ---
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

        # --- Model C: SVM ---
        svm_model = SVC(C=10, kernel="rbf", gamma="scale", probability=True,
                        random_state=SEED)
        svm_model.fit(train_feats_scaled, train_feat_labels_all.astype(int))
        svm_probs = svm_model.predict_proba(test_feats_scaled)[:, 1]

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

        status = "CORRECT" if subj_pred == test_label else "WRONG"
        label_str = "MDD" if test_label == 1 else "H"
        pred_str = "MDD" if subj_pred == 1 else "H"
        log.info(
            f"  Fold {fold_idx + 1:2d}/{n_subjects}: {test_sid:10s} "
            f"True={label_str:3s} Pred={pred_str:3s} "
            f"prob={subj_prob:.3f} "
            f"(CNN={np.mean(cnn_probs):.3f} XGB={np.mean(xgb_probs):.3f} SVM={np.mean(svm_probs):.3f}) "
            f"[{status}]"
        )

    return compute_and_report_metrics(
        all_epoch_labels, all_epoch_probs_ensemble,
        all_epoch_probs_cnn, all_epoch_probs_xgb, all_epoch_probs_svm,
        all_epoch_subject_ids, subject_true, subject_pred_prob,
    )


# ===========================================================================
# 6. METRICS & REPORTING
# ===========================================================================

def compute_and_report_metrics(
    epoch_labels, epoch_probs_ensemble,
    epoch_probs_cnn, epoch_probs_xgb, epoch_probs_svm,
    epoch_subject_ids, subject_true, subject_pred_prob,
):
    """Compute and print all metrics."""
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
    for model_name, model_probs in [("CNN", epoch_probs_cnn), ("XGB", epoch_probs_xgb), ("SVM", epoch_probs_svm)]:
        model_probs = np.array(model_probs)
        model_subj_probs = {}
        for sid in subject_ids_sorted:
            mask = np.array([s == sid for s in epoch_subject_ids])
            model_subj_probs[sid] = np.mean(model_probs[mask])
        model_subj_preds = np.array([(model_subj_probs[s] >= 0.5) for s in subject_ids_sorted]).astype(int)
        model_subj_acc = accuracy_score(subj_true_arr, model_subj_preds)
        log.info(f"  {model_name} subject-level accuracy: {model_subj_acc:.4f}")

    # Misclassified subjects
    misclassified = [s for s in subject_ids_sorted if subj_pred_arr[subject_ids_sorted.index(s)] != subj_true_arr[subject_ids_sorted.index(s)]]

    # --- Report ---
    log.info("\n" + "=" * 60)
    log.info("FINAL RESULTS")
    log.info("=" * 60)

    log.info("\n--- Sample-Level Metrics ---")
    log.info(f"  Accuracy:    {sample_acc:.4f}")
    log.info(f"  F1 Score:    {sample_f1:.4f}")
    log.info(f"  AUC-ROC:     {sample_auc:.4f}")
    log.info(f"  Precision:   {sample_prec:.4f}")
    log.info(f"  Recall:      {sample_rec:.4f}")
    log.info(f"  Sensitivity: {sample_sens:.4f}")
    log.info(f"  Specificity: {sample_spec:.4f}")
    log.info(f"  Confusion:   TP={tp} TN={tn} FP={fp} FN={fn}")

    log.info("\n--- Per-Model Sample Accuracy ---")
    log.info(f"  CNN:     {cnn_acc:.4f}")
    log.info(f"  XGBoost: {xgb_acc:.4f}")
    log.info(f"  SVM:     {svm_acc:.4f}")
    log.info(f"  Ensemble:{sample_acc:.4f}")

    log.info("\n--- Subject-Level Metrics ---")
    log.info(f"  Accuracy:    {subj_acc:.4f} ({int(subj_acc * len(subj_true_arr))}/{len(subj_true_arr)} subjects)")
    log.info(f"  F1 Score:    {subj_f1:.4f}")
    log.info(f"  AUC-ROC:     {subj_auc:.4f}")
    log.info(f"  Precision:   {subj_prec:.4f}")
    log.info(f"  Recall:      {subj_rec:.4f}")
    log.info(f"  Sensitivity: {subj_sens:.4f}")
    log.info(f"  Specificity: {subj_spec:.4f}")
    log.info(f"  Confusion Matrix:")
    log.info(f"    Predicted:  H    MDD")
    log.info(f"    True H:   {s_tn:3d}  {s_fp:3d}")
    log.info(f"    True MDD: {s_fn:3d}  {s_tp:3d}")

    if misclassified:
        log.info(f"\n  Misclassified subjects ({len(misclassified)}):")
        for sid in misclassified:
            true_l = "MDD" if subject_true[sid] == 1 else "H"
            pred_p = subject_pred_prob[sid]
            log.info(f"    {sid}: true={true_l}, prob={pred_p:.3f}")

    # --- Save results ---
    results = {
        "sample_level": {
            "accuracy": float(sample_acc),
            "f1": float(sample_f1),
            "auc_roc": float(sample_auc),
            "precision": float(sample_prec),
            "recall": float(sample_rec),
            "sensitivity": float(sample_sens),
            "specificity": float(sample_spec),
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
    log.info(f"\nResults saved to {results_path}")

    return results


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    log.info("=" * 60)
    log.info("EEG Depression Detection — V3 Ensemble Training")
    log.info("=" * 60)
    log.info(f"Python: {sys.version}")
    log.info(f"PyTorch: {torch.__version__}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"Device: {DEVICE}")
    log.info(f"Data dir: {DATA_DIR}")
    log.info(f"Output dir: {OUTPUT_DIR}")

    t_start = time.time()

    # Load data
    all_subjects = load_all_data()
    if len(all_subjects) == 0:
        log.error("No subjects loaded! Check data directory.")
        sys.exit(1)

    # Run LOSO CV
    results = run_loso_cv(all_subjects)

    elapsed = time.time() - t_start
    log.info(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    log.info("Done.")

    return results


if __name__ == "__main__":
    main()
