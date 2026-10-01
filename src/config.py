from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Loads the YAML config; entries under `paths` become absolute Paths rooted at the project."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: (PROJECT_ROOT / v).resolve() for k, v in cfg["paths"].items()}
    return cfg
