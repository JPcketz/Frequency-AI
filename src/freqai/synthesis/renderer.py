from __future__ import annotations
from typing import Iterable, Tuple, Union, Dict
from pathlib import Path
import numpy as np
import soundfile as sf
import pretty_midi

NoteEvent = Tuple[Union[int, str], float, float, int]

def _note_to_midi(n: Union[int, str]) -> int:
    return int(pretty_midi.note_name_to_number(n)) if isinstance(n, str) else int(n)

def _midi_to_freq(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

def _adsr(n: int, sr: int, a=0.005, d=0.05, s=0.85, r=0.05):
    env = np.ones(n, np.float32) * s
    a_s, d_s, r_s = int(a*sr), int(d*sr), int(r*sr)
    if a_s > 1: env[:a_s] = np.linspace(0.0, 1.0, a_s, dtype=np.float32)
    if d_s > 0 and a_s + d_s < n:
        env[a_s:a_s+d_s] = np.linspace(1.0, s, d_s, dtype=np.float32)
    if r_s > 0:
        rs = max(n - r_s, 0)
        env[rs:] = np.linspace(env[rs], 0.0, n - rs, dtype=np.float32)
    return env

def _osc(wave: str, f: float, t: np.ndarray) -> np.ndarray:
    ph = 2*np.pi*f*t
    if wave == "sine":    return np.sin(ph, dtype=np.float32)
    if wave == "square":  return np.sign(np.sin(ph)).astype(np.float32)
    if wave == "triangle":
        frac = (f*t) % 1.0
        return (2*np.abs(2*frac - 1) - 1).astype(np.float32)
    if wave == "saw":
        frac = (f*t) % 1.0
        return (2*frac - 1).astype(np.float32)
    rng = np.random.default_rng(42)
    return rng.uniform(-1, 1, t.shape).astype(np.float32)

def render_events_to_array(
    events: Iterable[NoteEvent], bpm: int, sr: int = 44100,
    wave: str = "saw", gain: float = 0.22, adsr: Dict[str,float] | None = None
) -> np.ndarray:
    spb = 60.0 / float(bpm)
    ev = list(events)
    if not ev: return np.zeros(1, np.float32)
    total_sec = max(float(e[2]) for e in ev) * spb
    n = int(np.ceil(total_sec * sr)) + 1
    out = np.zeros(n, np.float32)
    if adsr is None: adsr = {"a":0.005,"d":0.05,"s":0.85,"r":0.05}
    for pitch, sb, eb, vel in ev:
        f = _midi_to_freq(_note_to_midi(pitch))
        s_sec, e_sec = float(sb)*spb, float(eb)*spb
        e_sec = max(e_sec, s_sec + 1e-3)
        s_idx, e_idx = int(s_sec*sr), min(int(np.ceil(e_sec*sr)), n)
        t = np.arange(e_idx - s_idx, dtype=np.float32) / sr
        seg = _osc(wave, f, t) * _adsr(len(t), sr, **adsr)
        seg = (gain * (max(1, min(int(vel),127))/127.0) * seg).astype(np.float32)
        out[s_idx:e_idx] += seg
    peak = np.max(np.abs(out))
    return out if peak <= 0.99 else (out/peak*0.99).astype(np.float32)

def write_wav_from_events(events, bpm, out_path: str | Path, sr=44100, wave="saw", gain=0.22, adsr=None) -> str:
    audio = render_events_to_array(events, bpm=bpm, sr=sr, wave=wave, gain=gain, adsr=adsr)
    out_path = str(out_path)
    sf.write(out_path, audio, sr, subtype="PCM_16")
    return out_path