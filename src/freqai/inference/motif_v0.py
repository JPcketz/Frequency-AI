"""
motif_v0: repeat a short melody motif at target sections (e.g., Chorus).

- Extract the first N bars of melody as the motif (default 2 bars).
- For each target section start (in seconds), snap to the nearest bar, then
  insert the motif there, replacing existing melody events in that window.

Assumptions:
- 4/4 time (beats_per_bar = 4)
- parts dict contains "melody": List[NoteEvent]
- NoteEvent = (pitch: int|str, start_beat: float, end_beat: float, velocity: int)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Union, Iterable

NoteEvent = Tuple[Union[int, str], float, float, int]

def _clone_evt(evt: NoteEvent, dt_beats: float) -> NoteEvent:
    p, s, e, v = evt
    return (p, s + dt_beats, e + dt_beats, v)

def _clip_evt(evt: NoteEvent, start_b: float, end_b: float) -> List[NoteEvent]:
    """Clip evt to [start_b, end_b); return [] if outside."""
    p, s, e, v = evt
    s2 = max(s, start_b)
    e2 = min(e, end_b)
    if e2 <= s2:
        return []
    return [(p, s2, e2, v)]

def _remove_range(events: List[NoteEvent], start_b: float, end_b: float) -> List[NoteEvent]:
    out: List[NoteEvent] = []
    for ev in events:
        p, s, e, v = ev
        # keep events fully outside the window
        if e <= start_b or s >= end_b:
            out.append(ev)
        else:
            # if overlapping, drop it (simple replace policy)
            pass
    return out

def apply_motif_repetition(
    parts: Dict[str, List[NoteEvent]],
    bpm: int,
    sections: List[Dict[str, int]],
    target_names: Iterable[str] = ("Chorus", "Chorus/Outro"),
    motif_bars: int = 2,
    beats_per_bar: int = 4,
) -> Dict[str, List[NoteEvent]]:
    """Return updated parts with melody motif repeated at section starts."""
    melody = parts.get("melody", [])
    if not melody:
        return parts

    # Find motif window from earliest melody event
    motif_start_b = min(ev[1] for ev in melody)
    motif_len_b = float(motif_bars * beats_per_bar)
    motif_end_b = motif_start_b + motif_len_b

    # Extract motif events (clip to window)
    motif_events: List[NoteEvent] = []
    for ev in melody:
        motif_events.extend(_clip_evt(ev, motif_start_b, motif_end_b))
    if not motif_events:
        return parts

    # Seconds -> beats helper
    spb = 60.0 / float(bpm)

    # Build list of target starts (in beats), snapped to nearest bar
    target_starts_b: List[float] = []
    for sec in sections:
        name = str(sec.get("name", ""))
        if not any(tn.lower() in name.lower() for tn in target_names):
            continue
        start_sec = float(sec.get("start", 0))
        start_b = start_sec / spb
        # snap to nearest bar
        snapped = round(start_b / beats_per_bar) * beats_per_bar
        # avoid placing motif back on itself at the original spot
        if abs(snapped - motif_start_b) < 0.5:
            continue
        target_starts_b.append(snapped)

    if not target_starts_b:
        return parts

    # Replace melody in those windows with the cloned motif
    new_melody = melody[:]
    for tgt_b in target_starts_b:
        window_start = tgt_b
        window_end = tgt_b + motif_len_b
        new_melody = _remove_range(new_melody, window_start, window_end)
        dt = tgt_b - motif_start_b
        new_melody.extend([_clone_evt(ev, dt) for ev in motif_events])

    # Keep ordering tidy
    new_melody.sort(key=lambda ev: (ev[1], ev[2], ev[0] if isinstance(ev[0], int) else 0))
    parts = dict(parts)
    parts["melody"] = new_melody
    return parts