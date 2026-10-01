"""Pixel-level feature engineering, training-pixel sampling and on-disk feature caching.

Every pixel becomes one sample whose features are its full Sentinel-2 time series:
10 reflectance bands x T dates, plus per-date spectral indices. This relies on all patches
sharing one acquisition calendar (true for this single-tile subset), so feature `B8_t20`
means the same date everywhere.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import S2_BAND_NAMES, PastisSubset

BAND_INDEX = {b: i for i, b in enumerate(S2_BAND_NAMES)}
INDEX_DEFINITIONS = {
    "NDVI": ("B8", "B4"),    # vegetation vigour
    "NDMI": ("B8", "B11"),   # canopy / soil moisture
}


def to_reflectance(s2: np.ndarray, scale: float, clip: tuple[float, float]) -> np.ndarray:
    """int16 digital numbers -> float32 surface reflectance, clipped to a physical range."""
    refl = s2.astype(np.float32) / scale
    return np.clip(refl, clip[0], clip[1], out=refl)


def normalized_difference(refl: np.ndarray, band_a: str, band_b: str) -> np.ndarray:
    """(T, B, H, W) reflectance -> (T, H, W) index (a - b) / (a + b)."""
    a, b = refl[:, BAND_INDEX[band_a]], refl[:, BAND_INDEX[band_b]]
    total = a + b
    return np.divide(a - b, total, out=np.zeros_like(total), where=total > 0)


def feature_names(n_dates: int, indices: list[str]) -> list[str]:
    names = [f"{band}_t{t:02d}" for t in range(n_dates) for band in S2_BAND_NAMES]
    names += [f"{idx}_t{t:02d}" for idx in indices for t in range(n_dates)]
    return names


def patch_features(s2: np.ndarray, feature_cfg: dict) -> np.ndarray:
    """(T, B, H, W) int16 patch -> (H*W, T*B + T*len(indices)) float32 feature matrix."""
    refl = to_reflectance(s2, feature_cfg["reflectance_scale"], feature_cfg["clip"])
    n_dates, n_bands, height, width = refl.shape
    blocks = [refl.transpose(2, 3, 0, 1).reshape(height * width, n_dates * n_bands)]
    for name in feature_cfg["indices"]:
        index = normalized_difference(refl, *INDEX_DEFINITIONS[name])
        blocks.append(index.transpose(1, 2, 0).reshape(height * width, n_dates))
    return np.concatenate(blocks, axis=1)


def sampling_probabilities(class_totals: pd.Series, max_per_class: int, ignore_index: int) -> np.ndarray:
    """Per-class keep probability so that each class contributes at most ~max_per_class pixels."""
    totals = class_totals.to_numpy(dtype=float)
    probs = np.divide(max_per_class, totals, out=np.zeros_like(totals), where=totals > 0).clip(max=1.0)
    probs[ignore_index] = 0.0
    return probs


def class_weights(y: np.ndarray, power: float) -> dict[int, float]:
    """Inverse-frequency weights ** power, normalised so the mean sample weight is 1."""
    classes, counts = np.unique(y, return_counts=True)
    weights = counts.astype(float) ** -power
    weights *= counts.sum() / (weights * counts).sum()
    return dict(zip(classes.tolist(), weights.tolist()))


def sample_training_pixels(
    ds: PastisSubset,
    patch_ids: list[int],
    class_counts: pd.DataFrame,
    feature_cfg: dict,
    max_per_class: int,
    ignore_index: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class-capped random pixel sample from the given patches -> (X, y, source patch_id)."""
    probs = sampling_probabilities(class_counts.loc[patch_ids].sum(), max_per_class, ignore_index)
    rng = np.random.default_rng(seed)

    masks = {}
    for pid in patch_ids:
        y = ds.load_target(pid).ravel()
        masks[pid] = rng.random(y.size) < probs[y]
    n_total = sum(int(m.sum()) for m in masks.values())

    n_features = len(feature_names(len(ds.dates_for(patch_ids[0])), feature_cfg["indices"]))
    X = np.empty((n_total, n_features), dtype=np.float32)
    y_out = np.empty(n_total, dtype=np.uint8)
    source = np.empty(n_total, dtype=np.int32)
    start = 0
    for pid, mask in masks.items():
        n = int(mask.sum())
        if n == 0:
            continue
        X[start:start + n] = patch_features(ds.load_s2(pid), feature_cfg)[mask]
        y_out[start:start + n] = ds.load_target(pid).ravel()[mask]
        source[start:start + n] = pid
        start += n
    return X, y_out, source


def dense_patch_features(
    ds: PastisSubset, patch_ids: list[int], feature_cfg: dict, out_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Features for every pixel of every patch, written to a (n_patches, H*W, F) .npy memmap.
    Returns (features memmap opened read-only, labels (n_patches, H*W) uint8)."""
    first = ds.load_s2(patch_ids[0])
    n_pixels = first.shape[2] * first.shape[3]
    n_features = len(feature_names(first.shape[0], feature_cfg["indices"]))
    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(len(patch_ids), n_pixels, n_features)
    )
    for i, pid in enumerate(patch_ids):
        out[i] = patch_features(ds.load_s2(pid), feature_cfg)
    out.flush()
    del out
    labels = np.stack([ds.load_target(pid).ravel() for pid in patch_ids])
    return np.load(out_path, mmap_mode="r"), labels


def _cache_key(*parts) -> str:
    return hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]


def cached_training_sample(cfg: dict, ds: PastisSubset, patch_ids: list[int], class_counts: pd.DataFrame):
    sampling = cfg["sampling"]
    key = _cache_key(patch_ids, cfg["features"], sampling["max_pixels_per_class"], cfg["seed"])
    path = cfg["paths"]["cache_dir"] / f"train_sample_{key}.npz"
    if path.exists():
        data = np.load(path)
        return data["X"], data["y"], data["source"]
    X, y, source = sample_training_pixels(
        ds, patch_ids, class_counts, cfg["features"],
        sampling["max_pixels_per_class"], cfg["labels"]["ignore_index"], cfg["seed"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y, source=source)
    return X, y, source


def cached_dense_features(cfg: dict, ds: PastisSubset, patch_ids: list[int], split_name: str):
    key = _cache_key(patch_ids, cfg["features"])
    features_path = cfg["paths"]["cache_dir"] / f"{split_name}_features_{key}.npy"
    labels_path = cfg["paths"]["cache_dir"] / f"{split_name}_labels_{key}.npy"
    if features_path.exists() and labels_path.exists():
        return np.load(features_path, mmap_mode="r"), np.load(labels_path)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features, labels = dense_patch_features(ds, patch_ids, cfg["features"], features_path)
    np.save(labels_path, labels)
    return features, labels
