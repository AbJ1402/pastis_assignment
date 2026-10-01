"""Patch-level train / validation / test split built from the official PASTIS folds.

Whole patches (never individual pixels) are assigned to a split, so pixels of the same parcel
can never appear in both training and evaluation data.

Usage: python -m src.splits --config configs/config.yaml
"""
from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import geopandas as gpd
import pandas as pd
from scipy.spatial.distance import jensenshannon

from src.config import DEFAULT_CONFIG, load_config
from src.data_loading import PastisSubset, class_pixel_counts

SPLIT_NAMES = ("train", "val", "test")


def assign_splits(folds: pd.Series, val_fold: int, test_fold: int) -> pd.Series:
    """folds: Fold number indexed by patch_id -> split name indexed by patch_id."""
    return folds.map(lambda f: "val" if f == val_fold else "test" if f == test_fold else "train")


def split_class_totals(counts: pd.DataFrame, assignment: pd.Series, ignore_index: int) -> pd.DataFrame:
    """Pixel count per class (rows) and split (columns), void class removed."""
    totals = counts.drop(columns=ignore_index).groupby(assignment).sum().T
    return totals.reindex(columns=list(SPLIT_NAMES))


def score_fold_pairs(counts: pd.DataFrame, folds: pd.Series, ignore_index: int) -> pd.DataFrame:
    """Scores every (val_fold, test_fold) choice; the remaining folds form the training set."""
    overall = counts.drop(columns=ignore_index).sum()
    p_overall = overall / overall.sum()
    rows = []
    for val_fold, test_fold in permutations(sorted(folds.unique()), 2):
        assignment = assign_splits(folds, val_fold, test_fold)
        totals = split_class_totals(counts, assignment, ignore_index)
        row = {"val_fold": val_fold, "test_fold": test_fold}
        for name in SPLIT_NAMES:
            col = totals[name]
            row[f"n_patches_{name}"] = int((assignment == name).sum())
            row[f"n_classes_{name}"] = int((col > 0).sum())
            row[f"jsd_{name}"] = float(jensenshannon(col / col.sum(), p_overall, base=2))
        rows.append(row)
    scores = pd.DataFrame(rows)
    scores["min_classes_val_test"] = scores[["n_classes_val", "n_classes_test"]].min(axis=1)
    scores["jsd_val_plus_test"] = scores["jsd_val"] + scores["jsd_test"]
    return scores


def choose_fold_pair(scores: pd.DataFrame) -> tuple[int, int]:
    """Lexicographic rule: most classes seen in training, then most classes evaluable in both
    val and test, then val/test class distributions closest to the overall distribution."""
    ranked = scores.sort_values(
        ["n_classes_train", "min_classes_val_test", "jsd_val_plus_test"],
        ascending=[False, False, True],
    )
    best = ranked.iloc[0]
    return int(best["val_fold"]), int(best["test_fold"])


def nearest_train_distance(geometries: gpd.GeoSeries, assignment: pd.Series) -> pd.Series:
    """Distance (m, EPSG:2154) from each val/test patch to the closest training patch.
    `geometries` is indexed by patch_id."""
    train_geoms = geometries[assignment[assignment == "train"].index]
    held_out = assignment[assignment != "train"].index
    return pd.Series(
        [train_geoms.distance(geometries[pid]).min() for pid in held_out], index=held_out, name="dist_m"
    )


def save_split(assignment: pd.Series, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in SPLIT_NAMES:
        ids = sorted(assignment[assignment == name].index)
        paths[name] = out_dir / f"{name}_ids.txt"
        paths[name].write_text("\n".join(str(i) for i in ids) + "\n")
    return paths


def load_split(splits_dir: Path) -> dict[str, list[int]]:
    return {
        name: [int(line) for line in (splits_dir / f"{name}_ids.txt").read_text().split()]
        for name in SPLIT_NAMES
    }


def main(config_path: str | Path = DEFAULT_CONFIG) -> pd.Series:
    cfg = load_config(config_path)
    ignore_index = cfg["labels"]["ignore_index"]
    ds = PastisSubset(cfg["paths"]["data_root"])
    counts = class_pixel_counts(ds)
    folds = ds.metadata.set_index("ID_PATCH")["Fold"]

    val_fold, test_fold = cfg["split"]["val_fold"], cfg["split"]["test_fold"]
    if val_fold is None or test_fold is None:
        val_fold, test_fold = choose_fold_pair(score_fold_pairs(counts, folds, ignore_index))

    assignment = assign_splits(folds, val_fold, test_fold)
    paths = save_split(assignment, cfg["paths"]["splits_dir"])
    print(f"val_fold={val_fold}, test_fold={test_fold}")
    for name, path in paths.items():
        print(f"{name}: {(assignment == name).sum()} patches -> {path}")
    return assignment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
