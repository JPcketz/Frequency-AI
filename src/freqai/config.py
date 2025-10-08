"""
Simple YAML config loader + deep-merge for Frequency AI (M0).
We'll wire this into the CLI next so you can do: --config my_preset.yaml
"""
from __future__ import annotations
from typing import Any, Dict, Mapping
from pathlib import Path
import os
import yaml

ALLOWED_TOP_LEVEL = {
    "key", "mode", "bpm", "anchor", "anchor_bars",
    "groove", "quantize", "humanize", "instruments",
    "length", "marker", "stems", "midi", "wav",
    "csv_structure", "outdir", "outfile"
}

def load_yaml_config(path: Path | str | None) -> Dict[str, Any]:
    """
    Load a YAML config file if provided; else check env FREQAI_CONFIG.
    Returns {} if nothing found. Values are not validated here.
    """
    cfg_path: Path | None = None
    if path:
        cfg_path = Path(path)
    else:
        env_path = os.getenv("FREQAI_CONFIG")
        if env_path:
            cfg_path = Path(env_path)

    if not cfg_path:
        return {}

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Keep only known keys (avoid surprises)
    clean = {k: v for k, v in data.items() if k in ALLOWED_TOP_LEVEL}
    return clean

def deep_merge(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Shallow-merge by default; recursively merge nested dicts.
    Values in `b` (override) take precedence over `a` (base).
    """
    out: Dict[str, Any] = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out