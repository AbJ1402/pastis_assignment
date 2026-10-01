"""Data loading utilities for the PASTIS subset (Sentinel-2 + annotations).

Dataset layout expected under `data_root`:
    PASTIS_subset/
        DATA_S2/S2_<patch_id>.npy        -> (T, 10, H, W) int16
        ANNOTATIONS/TARGET_<patch_id>.npy -> (1, H, W) uint8
        metadata.geojson                  -> per-patch geometry + acquisition dates
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

S2_BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

CLASS_NAMES = {
    0: "Background",
    1: "Meadow",
    2: "Soft winter wheat",
    3: "Corn",
    4: "Winter barley",
    5: "Winter rapeseed",
    6: "Spring barley",
    7: "Sunflower",
    8: "Grapevine",
    9: "Beet",
    10: "Winter triticale",
    11: "Winter durum wheat",
    12: "Fruits, vegetables, flowers",
    13: "Potatoes",
    14: "Leguminous fodder",
    15: "Soybeans",
    16: "Orchard",
    17: "Mixed cereal",
    18: "Sorghum",
    19: "Void label",
}


class PastisSubset:
    """Lightweight accessor for the PASTIS subset used in this assignment."""

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self.s2_dir = self.data_root / "DATA_S2"
        self.ann_dir = self.data_root / "ANNOTATIONS"
        self.metadata_path = self.data_root / "metadata.geojson"

        with open(self.metadata_path) as f:
            self._geojson = json.load(f)

        self.metadata = self._build_metadata_table(self._geojson)
        self.patch_ids = sorted(self.metadata["ID_PATCH"].tolist())

    @staticmethod
    def _build_metadata_table(geojson: dict) -> pd.DataFrame:
        rows = []
        for feat in geojson["features"]:
            props = dict(feat["properties"])
            dates_raw = props.pop("dates-S2")
            dates = dates_raw if isinstance(dates_raw, dict) else json.loads(
                dates_raw.replace("'", '"')
            )
            props["dates"] = {int(k): v for k, v in dates.items()}
            props["n_dates"] = len(dates)
            props["geometry"] = feat["geometry"]
            rows.append(props)
        df = pd.DataFrame(rows)
        df["ID_PATCH"] = df["ID_PATCH"].astype(int)
        return df

    def s2_path(self, patch_id: int) -> Path:
        return self.s2_dir / f"S2_{patch_id}.npy"

    def target_path(self, patch_id: int) -> Path:
        return self.ann_dir / f"TARGET_{patch_id}.npy"

    def load_s2(self, patch_id: int) -> np.ndarray:
        """Returns (T, 10, H, W) int16 array."""
        return np.load(self.s2_path(patch_id))

    def load_target(self, patch_id: int) -> np.ndarray:
        """Returns (1, H, W) uint8 array (0th annotation layer only)."""
        return np.load(self.target_path(patch_id))

    def dates_for(self, patch_id: int) -> list[int]:
        row = self.metadata.loc[self.metadata["ID_PATCH"] == patch_id].iloc[0]
        return [row["dates"][k] for k in sorted(row["dates"])]

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __iter__(self):
        return iter(self.patch_ids)


def class_pixel_counts(
    ds: PastisSubset, patch_ids: list[int] | None = None, n_classes: int = len(CLASS_NAMES)
) -> pd.DataFrame:
    """Pixel count of every class in every patch: index = patch_id, columns = class ids."""
    patch_ids = ds.patch_ids if patch_ids is None else patch_ids
    counts = np.stack([
        np.bincount(ds.load_target(pid).ravel(), minlength=n_classes) for pid in patch_ids
    ])
    return pd.DataFrame(counts, index=pd.Index(patch_ids, name="patch_id"), columns=range(n_classes))


def discover_patch_ids(data_root: str | Path) -> list[int]:
    """Cross-check patch ids present in both DATA_S2 and ANNOTATIONS folders."""
    data_root = Path(data_root)
    s2_ids = {
        int(re.match(r"S2_(\d+)\.npy", p.name).group(1))
        for p in (data_root / "DATA_S2").glob("S2_*.npy")
    }
    target_ids = {
        int(re.match(r"TARGET_(\d+)\.npy", p.name).group(1))
        for p in (data_root / "ANNOTATIONS").glob("TARGET_*.npy")
    }
    missing_targets = s2_ids - target_ids
    missing_s2 = target_ids - s2_ids
    if missing_targets:
        raise ValueError(f"Patches missing annotations: {sorted(missing_targets)}")
    if missing_s2:
        raise ValueError(f"Patches missing S2 data: {sorted(missing_s2)}")
    return sorted(s2_ids)
