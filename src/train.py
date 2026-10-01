"""Train pixel-level crop classifiers and compare them on the validation split.

Usage:
    python -m src.train --config configs/config.yaml
    python -m src.train --models lightgbm random_forest     # (re)train a subset

Outputs: fitted models (outputs/models), validation predictions (outputs/predictions),
validation metrics + training log (outputs/metrics). The test split is not touched here.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from itertools import product

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import DEFAULT_CONFIG, load_config
from src.data_loading import CLASS_NAMES, PastisSubset, class_pixel_counts
from src.evaluate import (
    MODEL_ORDER, confusion_matrix, evaluate_predictions, metrics_from_confusion, predict_patches,
    upsert_csv,
)
from src.preprocessing import (
    cached_dense_features, cached_training_sample, class_weights, feature_names,
)
from src.splits import load_split

SAMPLE_WEIGHT_STEP = {"logistic_regression": "logisticregression", "svm_rbf": "svc"}


def build_model(name: str, params: dict, seed: int, n_jobs: int, majority_label: int):
    p = dict(params)
    if name == "majority":
        return DummyClassifier(strategy="constant", constant=majority_label)
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=p["max_iter"]))
    if name == "svm_rbf":
        return make_pipeline(
            StandardScaler(), PCA(p["pca_components"], random_state=seed),
            SVC(C=p["C"], gamma=p["gamma"], kernel="rbf", cache_size=1000, random_state=seed),
        )
    if name == "random_forest":
        return RandomForestClassifier(**p, n_jobs=n_jobs, random_state=seed)
    if name == "xgboost":
        return XGBClassifier(**p, tree_method="hist", n_jobs=n_jobs, random_state=seed)
    if name == "lightgbm":
        p.pop("early_stopping_rounds")
        return lgb.LGBMClassifier(**p, n_jobs=n_jobs, random_state=seed, verbose=-1)
    raise ValueError(f"Unknown model: {name}")


def fit_model(name: str, model, X, y, w, eval_set, params: dict):
    if name == "majority":                   # does not support sample weights
        return model.fit(X, y)
    if name in SAMPLE_WEIGHT_STEP:
        return model.fit(X, y, **{f"{SAMPLE_WEIGHT_STEP[name]}__sample_weight": w})
    if name == "xgboost":
        return model.fit(X, y, sample_weight=w, eval_set=[eval_set], verbose=False)
    if name == "lightgbm":
        stop = lgb.early_stopping(params["early_stopping_rounds"], first_metric_only=True, verbose=False)
        return model.fit(X, y, sample_weight=w, eval_X=eval_set[0], eval_y=eval_set[1], callbacks=[stop])
    return model.fit(X, y, sample_weight=w)


def model_details(name: str, model) -> dict:
    if name == "xgboost":
        return {"n_boosting_rounds": int(model.best_iteration) + 1}      # best_iteration is 0-based
    if name == "lightgbm":
        return {"n_boosting_rounds": int(model.best_iteration_)}          # best_iteration_ is 1-based
    if name == "svm_rbf":
        return {"n_support_vectors": int(model[-1].n_support_.sum()),
                "pca_explained_variance": float(model[1].explained_variance_ratio_.sum())}
    if name == "logistic_regression":
        return {"n_iter": int(np.max(model[-1].n_iter_))}
    return {}


def feature_importance(name: str, model) -> np.ndarray | None:
    if name in ("random_forest", "xgboost"):
        return np.asarray(model.feature_importances_, dtype=float)
    if name == "lightgbm":
        gain = model.booster_.feature_importance("gain")
        return gain / gain.sum()
    return None


def param_grid(search: dict) -> list[dict]:
    """{"C": [1, 10], "gamma": [0.01]} -> [{"C": 1, "gamma": 0.01}, {"C": 10, "gamma": 0.01}]."""
    keys = list(search)
    return [dict(zip(keys, values)) for values in product(*(search[k] for k in keys))]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    summary, _ = metrics_from_confusion(confusion_matrix(y_true, y_pred, n_classes))
    return float(summary["macro_f1"])


def early_stopping_curve(name: str, model) -> pd.DataFrame | None:
    """Per-round validation-subsample metrics recorded during boosting."""
    if name == "xgboost":
        curves = model.evals_result()["validation_0"]
        rename = {"mlogloss": "logloss", "merror": "error"}
    elif name == "lightgbm":
        curves = model.evals_result_["valid_0"]
        rename = {"multi_logloss": "logloss", "multi_error": "error"}
    else:
        return None
    df = pd.DataFrame(curves).rename(columns=rename)
    df.insert(0, "round", np.arange(1, len(df) + 1))
    return df


def capped_subsample(y: np.ndarray, max_per_class: int, rng: np.random.Generator) -> np.ndarray:
    idx = [
        rng.choice(np.flatnonzero(y == c), min(max_per_class, int((y == c).sum())), replace=False)
        for c in np.unique(y)
    ]
    return np.sort(np.concatenate(idx))


def main(config_path=DEFAULT_CONFIG, models: list[str] | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    paths, labels_cfg, sampling = cfg["paths"], cfg["labels"], cfg["sampling"]
    for key in ("models_dir", "metrics_dir", "predictions_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    models = models or [m for m in MODEL_ORDER if m in cfg["models"]]
    rng = np.random.default_rng(cfg["seed"])

    ds = PastisSubset(paths["data_root"])
    split = load_split(paths["splits_dir"])
    counts = class_pixel_counts(ds)

    t0 = time.time()
    X, y_raw, _ = cached_training_sample(cfg, ds, split["train"], counts)
    val_X, val_y = cached_dense_features(cfg, ds, split["val"], "val")
    print(f"features ready in {time.time() - t0:.0f}s: train sample {X.shape}, val {val_X.shape}", flush=True)

    classes = np.unique(y_raw)                      # PASTIS ids seen in training
    y = np.searchsorted(classes, y_raw)             # contiguous 0..K-1 labels (required by XGBoost)
    weights_by_class = class_weights(y_raw, sampling["class_weight_power"])
    w = np.array([weights_by_class[c] for c in classes])[y]
    majority_label = int(np.searchsorted(classes, counts.loc[split["train"], classes.tolist()].sum().idxmax()))

    val_flat_y = val_y.ravel()
    es_pool = np.flatnonzero(np.isin(val_flat_y, classes))
    es_idx = es_pool[capped_subsample(val_flat_y[es_pool], sampling["early_stopping_pixels_per_class"], rng)]
    es_X = np.asarray(val_X.reshape(-1, val_X.shape[2])[es_idx])
    es_set = (es_X, np.searchsorted(classes, val_flat_y[es_idx]))

    names = feature_names(len(ds.dates_for(split["train"][0])), cfg["features"]["indices"])
    (paths["models_dir"] / "label_classes.json").write_text(json.dumps(classes.tolist()))
    (paths["models_dir"] / "feature_names.json").write_text(json.dumps(names))

    log_path = paths["metrics_dir"] / "training_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"models": {}}
    log["training_sample"] = {
        "n_pixels": int(len(y)), "n_features": int(X.shape[1]),
        "class_counts": {CLASS_NAMES[c]: int((y_raw == c).sum()) for c in classes},
        "class_weights": {CLASS_NAMES[c]: round(weights_by_class[c], 4) for c in classes},
        "early_stopping_pixels": int(len(es_idx)),
    }

    for name in models:
        params = cfg["models"][name]
        print(f"[{name}] training ...", flush=True)
        if "max_pixels_per_class" in params:
            idx = capped_subsample(y_raw, params["max_pixels_per_class"], rng)
            Xf, yf = X[idx], y[idx]
            wf_by_class = class_weights(y_raw[idx], sampling["class_weight_power"])
            wf = np.array([wf_by_class[c] for c in classes])[yf]
        else:
            Xf, yf, wf = X, y, w
        base_params = {k: v for k, v in params.items() if k not in ("search", "max_pixels_per_class")}

        candidates = param_grid(params.get("search", {}))
        best, search_rows, search_start = None, [], time.time()
        for overrides in candidates:
            model = build_model(name, {**base_params, **overrides}, cfg["seed"], cfg["n_jobs"], majority_label)
            t = time.time()
            fit_model(name, model, Xf, yf, wf, es_set, params)
            fit_s = time.time() - t
            if len(candidates) == 1:
                best = (None, overrides, model, fit_s)
                break
            score = macro_f1(es_set[1], model.predict(es_set[0]), len(classes))
            search_rows.append({"model": name, **overrides, "val_subsample_macro_f1": round(score, 4),
                                "fit_seconds": round(fit_s, 1)})
            print(f"[{name}]   {overrides} -> val-subsample macro-F1 {score:.3f} ({fit_s:.0f}s)", flush=True)
            if best is None or score > best[0]:
                best = (score, overrides, model, fit_s)
        _, best_overrides, model, fit_s = best
        if search_rows:
            upsert_csv(pd.DataFrame(search_rows), paths["metrics_dir"] / "hyperparameter_search.csv")

        t = time.time()
        val_pred = predict_patches(model, val_X, classes)
        predict_s = time.time() - t

        model_path = paths["models_dir"] / f"{name}.joblib"
        joblib.dump(model, model_path, compress=3)
        np.save(paths["predictions_dir"] / f"val_{name}.npy", val_pred.reshape(len(val_pred), 128, 128).astype(np.uint8))

        summary, per_class, _ = evaluate_predictions(
            val_flat_y, val_pred.ravel(), labels_cfg["n_classes"], labels_cfg["ignore_index"]
        )
        row = {"model": name, **summary, "fit_seconds": round(fit_s, 1),
               "predict_seconds": round(predict_s, 1), "n_train_pixels": int(len(yf)),
               "model_size_mb": round(model_path.stat().st_size / 1e6, 2)}
        upsert_csv(pd.DataFrame([row]), paths["metrics_dir"] / "val_model_comparison.csv")
        upsert_csv(per_class.assign(model=name), paths["metrics_dir"] / "val_per_class_metrics.csv")

        curve = early_stopping_curve(name, model)
        if curve is not None:
            curve.to_csv(paths["metrics_dir"] / f"early_stopping_curve_{name}.csv", index=False)

        importance = feature_importance(name, model)
        if importance is not None:
            pd.DataFrame({"feature": names, "importance": importance}).to_csv(
                paths["metrics_dir"] / f"feature_importance_{name}.csv", index=False)

        log["models"][name] = {
            "params": params, "selected_params": {**base_params, **best_overrides},
            **model_details(name, model), "fit_seconds": row["fit_seconds"],
            "search_seconds": round(time.time() - search_start - predict_s, 1) if search_rows else 0.0,
            "predict_seconds": row["predict_seconds"],
        }
        log_path.write_text(json.dumps(log, indent=2))
        print(f"[{name}] fit {fit_s:.0f}s, predict {predict_s:.0f}s | OA={summary['overall_accuracy']:.3f} "
              f"macroF1={summary['macro_f1']:.3f} mIoU={summary['miou']:.3f}", flush=True)
        del model
        gc.collect()

    return pd.read_csv(paths["metrics_dir"] / "val_model_comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER)
    args = parser.parse_args()
    print(main(args.config, args.models).round(4).to_string(index=False))
