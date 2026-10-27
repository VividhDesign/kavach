"""Load the YAML configs: runtime settings (configs/default.yaml) and a site file with zones."""

from __future__ import annotations

from pathlib import Path

import yaml

from .mandown import ManDownConfig
from .ppe import PPEConfig
from .zones import Zone


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["ppe_cfg"] = PPEConfig(**cfg.get("ppe", {}))
    cfg["mandown_cfg"] = ManDownConfig(**cfg.get("mandown", {}))
    return cfg


def load_zones(path: str | Path | None) -> list[Zone]:
    if not path:
        return []
    site = yaml.safe_load(Path(path).read_text()) or {}
    zones = [Zone(**z) for z in site.get("zones", [])]
    for z in zones:
        if len(z.points) < 3 or not all(0 <= c <= 1 for pt in z.points for c in pt):
            raise ValueError(f"zone {z.name!r}: need >= 3 points with normalised (0-1) coordinates")
    return zones
