"""
Stems export + simple stereo mixdown (M0).

Inputs:
  parts_events: dict[str, list[NoteEvent]]  e.g. {"melody":[...], "bass":[...]}
  render_opts (optional): {
      "sr": 44100,
      "defaults": {"wave": "saw", "gain": 0.22, "pan": 0.5},
      "per_part": {
          "melody": {"wave": "triangle", "gain": 0.22, "pan": 0.65},
          "bass":   {"wave": "saw",      "gain": 0.28, "pan": 0.35},
      }
  }

Outputs:
  - Per-part mono WAV stems in outdir (e.g., melody.wav, bass.wav)
  - Stereo mixdown WAV (mix.wav) using constant-power panning + peak-safe normalization
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import math
import numpy as np
import soundfile as sf

from ..synthesis.renderer import render_events_to_array

NoteEvent = Tuple[Union[int, str], float, float, int]

DEFAULTS = {
    "sr": 44100,
    "defaults": {"wave": "saw", "gain": 0.22, "pan": 0.5},
    "per_part": {
        "melody": {"wave": "triangle", "gain": 0.22, "pan": 0.65},
        "bass":   {"wave": "saw",      "gain": 0.28, "pan": 0.35},
    },
}

def _get_cfg(render_opts: Dict[str, Any] | None) -> Dict[str, Any]:
    if not render_opts:
        return DEFAULTS
    # shallow-merge
    cfg = dict(DEFAULTS)
    for k, v in render_opts.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            merged = dict(cfg[k]); merged.update(v); cfg[k] = merged
        else:
            cfg[k] = v
    # also merge per_part sub-dicts
    if "per_part" in render_opts:
        pp = dict(DEFAULTS["per_part"])
        pp.update(render_opts["per_part"])
        cfg["per_part"] = pp
    return cfg

def _pan_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    """
    Constant-power pan: pan in [0,1], 0=left, 1=right, 0.5=center.
    Returns stereo (2, N).
    """
    pan = float(min(max(pan, 0.0), 1.0))
    theta = pan * (math.pi / 2.0)
    l = math.cos(theta)
    r = math.sin(theta)
    stereo = np.vstack([mono * l, mono * r]).astype(np.float32)
    return stereo

def _peak_normalize(stereo: np.ndarray, peak_target: float = 0.99) -> np.ndarray:
    peak = np.max(np.abs(stereo)) if stereo.size else 0.0
    if peak > peak_target and peak > 0:
        stereo = (stereo / peak * peak_target).astype(np.float32)
    return stereo

def export_stems_and_mix(
    parts_events: Dict[str, List[NoteEvent]],
    bpm: int,
    outdir: Union[str, Path],
    base: str = "demo",
    render_opts: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Renders each part to a mono WAV stem and writes a stereo mixdown.
    Returns dict with file paths and basic stats.
    """
    cfg = _get_cfg(render_opts)
    sr = int(cfg.get("sr", 44100))
    defaults = cfg.get("defaults", {})
    per_part = cfg.get("per_part", {})

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stems: Dict[str, np.ndarray] = {}
    written: Dict[str, str] = {}

    # Render each part
    for name, events in parts_events.items():
        pp_cfg = dict(defaults)
        pp_cfg.update(per_part.get(name, {}))
        wave = pp_cfg.get("wave", "saw")
        gain = float(pp_cfg.get("gain", 0.22))

        audio = render_events_to_array(events, bpm=bpm, sr=sr, wave=wave, gain=gain)
        stems[name] = audio

        # write mono stem
        stem_path = outdir / f"{base}.{name}.wav"
        sf.write(str(stem_path), audio, sr, subtype="PCM_16")
        written[f"stem_{name}"] = str(stem_path)

    # Build stereo mix with panning
    max_len = max((len(a) for a in stems.values()), default=0)
    mix = np.zeros((2, max_len), dtype=np.float32)

    for name, mono in stems.items():
        pp_cfg = dict(defaults)
        pp_cfg.update(per_part.get(name, {}))
        pan = float(pp_cfg.get("pan", 0.5))

        if len(mono) < max_len:
            pad = np.zeros(max_len - len(mono), dtype=np.float32)
            mono = np.concatenate([mono, pad])

        stereo = _pan_stereo(mono, pan)
        mix += stereo

    # Peak-safe normalization
    mix = _peak_normalize(mix, peak_target=0.99)

    # Write stereo mix
    mix_path = outdir / f"{base}.mix.wav"
    # (C, N) -> (N, C) for soundfile
    sf.write(str(mix_path), mix.T, sr, subtype="PCM_16")
    written["mix"] = str(mix_path)

    # Simple stats
    stats = {
        "sr": sr,
        "duration_sec": round(max_len / sr, 3) if max_len else 0.0,
        "stems": {k: {"length": len(v)} for k, v in stems.items()},
    }
    written["stats"] = stats
    return written