"""Segmentation metrics (void pixels excluded), patch-wise prediction and model evaluation.

Usage:
    python -m src.evaluate --config configs/config.yaml               # all saved models, val + test
    python -m src.evaluate --splits test --models random_forest

Outputs (outputs/metrics): evaluation_<split>.csv (overall metrics per model),
evaluation_<split>_per_class.csv and evaluation_<split>_confusion.npz (one 20x20 matrix per
model, rows = ground truth); predictions are saved as outputs/predictions/<split>_<model>.npy.
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG, load_config
from src.data_loading import CLASS_NAMES, PastisSubset
from src.preprocessing import cached_dense_features
from src.splits import load_split

MODEL_ORDER = [
    "majority", "logistic_regression", "svm_rbf", "random_forest", "xgboost", "lightgbm",
]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows = ground truth, columns = prediction."""
    flat = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    return np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def metrics_from_confusion(cm: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Overall + per-class metrics; macro scores average over classes present in the ground truth."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    n = cm.sum()

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    union = support + predicted - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)

    present = support > 0
    expected_agreement = (support * predicted).sum() / n**2
    overall_accuracy = tp.sum() / n
    summary = {
        "overall_accuracy": overall_accuracy,
        "macro_f1": f1[present].mean(),
        "weighted_f1": (f1 * support).sum() / support.sum(),
        "miou": iou[present].mean(),
        "kappa": (overall_accuracy - expected_agreement) / (1 - expected_agreement),
        "n_classes_evaluated": int(present.sum()),
        "n_pixels": int(n),
    }
    per_class = pd.DataFrame({
        "class_id": np.arange(len(cm)),
        "class_name": [CLASS_NAMES[i] for i in range(len(cm))],
        "precision": precision, "recall": recall, "f1": f1, "iou": iou,
        "support": support.astype(int), "predicted": predicted.astype(int),
    })
    return summary, per_class[present | (predicted > 0)].reset_index(drop=True)


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, ignore_index: int
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    valid = y_true != ignore_index
    cm = confusion_matrix(y_true[valid], y_pred[valid], n_classes)
    summary, per_class = metrics_from_confusion(cm)
    per_class = per_class[per_class["class_id"] != ignore_index].reset_index(drop=True)
    return summary, per_class, cm


def predict_patches(model, features, classes: np.ndarray) -> np.ndarray:
    """Predicts patch by patch from the (n_patches, H*W, F) memmap -> PASTIS class ids."""
    return np.stack([classes[model.predict(np.asarray(features[i]))] for i in range(len(features))])


def upsert_csv(df: pd.DataFrame, path, key: str = "model") -> pd.DataFrame:
    """Replaces the rows of the models in `df` and keeps the others, in MODEL_ORDER."""
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[~old[key].isin(df[key].unique())], df], ignore_index=True)
    df["_order"] = df[key].map({m: i for i, m in enumerate(MODEL_ORDER)})
    df = df.sort_values(["_order"], kind="stable").drop(columns="_order")
    df.to_csv(path, index=False)
    return df


def main(config_path=DEFAULT_CONFIG, splits=("val", "test"), models: list[str] | None = None) -> None:
    cfg = load_config(config_path)
    paths, labels_cfg = cfg["paths"], cfg["labels"]
    paths["predictions_dir"].mkdir(parents=True, exist_ok=True)
    ds = PastisSubset(paths["data_root"])
    split_ids = load_split(paths["splits_dir"])
    classes = np.array(json.loads((paths["models_dir"] / "label_classes.json").read_text()))
    models = models or [m for m in MODEL_ORDER if (paths["models_dir"] / f"{m}.joblib").exists()]

    features = {s: cached_dense_features(cfg, ds, split_ids[s], s) for s in splits}
    for name in models:
        model = joblib.load(paths["models_dir"] / f"{name}.joblib")
        for s in splits:
            X, y = features[s]
            t = time.time()
            pred = predict_patches(model, X, classes)
            predict_s = time.time() - t
            np.save(paths["predictions_dir"] / f"{s}_{name}.npy", pred.reshape(len(pred), 128, 128).astype(np.uint8))

            summary, per_class, cm = evaluate_predictions(
                y.ravel(), pred.ravel(), labels_cfg["n_classes"], labels_cfg["ignore_index"])
            upsert_csv(pd.DataFrame([{"model": name, **summary, "predict_seconds": round(predict_s, 1)}]),
                       paths["metrics_dir"] / f"evaluation_{s}.csv")
            upsert_csv(per_class.assign(model=name), paths["metrics_dir"] / f"evaluation_{s}_per_class.csv")
            cm_path = paths["metrics_dir"] / f"evaluation_{s}_confusion.npz"
            matrices = dict(np.load(cm_path)) if cm_path.exists() else {}
            matrices[name] = cm
            np.savez(cm_path, **matrices)
            print(f"[{name}] {s}: OA={summary['overall_accuracy']:.3f} macroF1={summary['macro_f1']:.3f} "
                  f"mIoU={summary['miou']:.3f} ({predict_s:.0f}s)", flush=True)
        del model
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER)
    args = parser.parse_args()
    main(args.config, args.splits, args.models)
