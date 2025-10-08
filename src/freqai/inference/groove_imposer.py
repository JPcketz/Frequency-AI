"""
Groove extraction & imposition (M0).
- extract_groove_template(midi_path, quantize='1/16') -> {'grid': '1/16', 'offsets_ms': [..]}
- impose_groove_on_events(events, bpm, template, max_ms=12) -> events with micro-timing applied

'events' are (pitch, start_beat, end_beat, velocity) like our NoteEvent elsewhere.
"""
from __future__ import annotations
from typing import Iterable, Tuple, Union, List, Dict
import bisect
import numpy as np
import pretty_midi

NoteEvent = Tuple[Union[int, str], float, float, int]

_GRID_MAP = {
    "1/4": 4,
    "1/8": 8,
    "1/12": 12,
    "1/16": 16,
    "1/24": 24,
    "1/32": 32,
}

def _grid_steps(quantize: str) -> int:
    if quantize not in _GRID_MAP:
        raise ValueError(f"Unsupported quantize '{quantize}'. Use one of {sorted(_GRID_MAP)}")
    return _GRID_MAP[quantize]

def _time_to_beat(t: float, beat_times: List[float]) -> float:
    """Piecewise-linear time->beat using pretty_midi beat grid."""
    if not beat_times:
        return 0.0
    if t <= beat_times[0]:
        # extrapolate before first beat
        if len(beat_times) >= 2:
            sec_per_beat = beat_times[1] - beat_times[0]
        else:
            sec_per_beat = 60.0 / 120.0
        return (t - beat_times[0]) / max(sec_per_beat, 1e-9)
    i = bisect.bisect_right(beat_times, t) - 1
    i = max(0, min(i, len(beat_times) - 2))
    sec_per_beat = beat_times[i + 1] - beat_times[i]
    frac = (t - beat_times[i]) / max(sec_per_beat, 1e-9)
    return i + frac

def _local_spb(beat_idx: int, beat_times: List[float]) -> float:
    """Seconds per beat around beat_idx."""
    i = max(0, min(beat_idx, len(beat_times) - 2))
    return max(beat_times[i + 1] - beat_times[i], 1e-9)

def extract_groove_template(midi_path: str, quantize: str = "1/16") -> Dict[str, object]:
    """
    Build average micro-timing offsets (in milliseconds) for each grid slot in a beat.
    Works on any notes it finds (drums or pitched). For a 1-bar loop this is ideal.
    """
    steps = _grid_steps(quantize)
    pm = pretty_midi.PrettyMIDI(midi_path)
    beat_times = pm.get_beats()
    if len(beat_times) < 2:
        # Fallback tempo estimate if no beats detected
        tempo = pm.estimate_tempo()
        # fabricate a short beat grid
        start = 0.0
        beat_times = [start + i * (60.0 / max(tempo, 1e-3)) for i in range(128)]

    # Collect all note-on times
    onsets = []
    for inst in pm.instruments:
        for note in inst.notes:
            onsets.append(note.start)
    if not onsets:
        # No notes? return zero offsets
        return {"grid": quantize, "offsets_ms": [0.0] * steps}

    # Compute offsets for each onset relative to nearest gridpoint
    per_slot_offsets = [[] for _ in range(steps)]
    for t in onsets:
        b = _time_to_beat(t, beat_times)  # continuous beat
        # nearest gridpoint in units of beats
        g = round(b * steps) / steps
        # local seconds per beat at this region
        base_beat = int(b)
        spb = _local_spb(base_beat, beat_times)
        # offset in ms (actual - ideal)
        offset_ms = (b - g) * spb * 1000.0
        # slot index within a beat (0..steps-1)
        slot = int(round((b % 1.0) * steps)) % steps
        per_slot_offsets[slot].append(offset_ms)

    # Aggregate (median) and clamp to reasonable human range
    offsets_ms = []
    for lst in per_slot_offsets:
        if lst:
            offsets_ms.append(float(np.median(lst)))
        else:
            offsets_ms.append(0.0)

    return {"grid": quantize, "offsets_ms": offsets_ms}

def impose_groove_on_events(
    events: Iterable[NoteEvent],
    bpm: int,
    template: Dict[str, object],
    max_ms: float = 12.0,
) -> List[NoteEvent]:
    """
    Shift note onsets according to groove template (by grid slot).
    Only adjusts start time; keeps end time relative to start.
    """
    quantize = str(template.get("grid", "1/16"))
    steps = _grid_steps(quantize)
    offsets_ms: List[float] = list(template.get("offsets_ms", [0.0] * steps))
    # Clamp offsets to +-max_ms
    offsets_ms = [float(np.clip(v, -abs(max_ms), abs(max_ms))) for v in offsets_ms]

    spb = 60.0 / float(bpm)  # seconds per beat

    out: List[NoteEvent] = []
    for pitch, s_beat, e_beat, vel in events:
        slot = int(round((s_beat % 1.0) * steps)) % steps
        shift_beats = (offsets_ms[slot] / 1000.0) / spb
        new_start = s_beat + shift_beats
        dur = max(e_beat - s_beat, 1e-4)
        new_end = new_start + dur
        out.append((pitch, new_start, new_end, vel))
    return out