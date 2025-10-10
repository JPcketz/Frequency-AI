"""
arrange_v0: light arrangement polish
- Drum fills at section transitions (snare 16ths + crash, kick pickup)
- Simple dynamics (velocity scaling) by section (Intro/Verse/Chorus/Bridge/Outro)

Assumptions:
- 4/4 time (beats_per_bar = 4)
- Sections are [{"name": str, "start": sec, "end": sec}, ...]
- parts contains "drums" optionally (if not, drum fills are skipped)
- NoteEvent = (pitch: int|str, start_beat: float, end_beat: float, velocity: int)
"""
from typing import Dict, List, Tuple, Union

NoteEvent = Tuple[Union[int, str], float, float, int]

# GM drum notes
KICK  = 36
SNARE = 38
CHH   = 42
OHH   = 46
CRASH = 49  # Crash Cymbal 1

def _sec_to_beat(sec: float, bpm: int) -> float:
    return float(sec) / (60.0 / float(bpm))

def _snap_to_bar(beat: float, beats_per_bar: int = 4) -> float:
    return round(beat / beats_per_bar) * beats_per_bar

def _add_evt(events: List[NoteEvent], pitch: int, s: float, e: float, v: int):
    events.append((int(pitch), float(s), float(max(e, s + 1e-4)), max(1, min(int(v), 127))))

def _apply_dynamics_to_window(events: List[NoteEvent], start_b: float, end_b: float, scale: float) -> List[NoteEvent]:
    out: List[NoteEvent] = []
    for p, s, e, v in events:
        if e <= start_b or s >= end_b:
            out.append((p, s, e, v))
        else:
            nv = int(max(1, min(round(v * scale), 127)))
            out.append((p, s, e, nv))
    return out

def _section_gain(name: str) -> float:
    n = name.lower()
    if "intro" in n:
        return 0.88
    if "chorus" in n:
        return 1.15
    if "bridge" in n:
        return 0.96
    if "outro" in n:
        return 0.92
    # verse / default
    return 1.00

def apply_arrangement(parts: Dict[str, List[NoteEvent]], bpm: int,
                      sections: List[Dict[str, int]], beats_per_bar: int = 4) -> Dict[str, List[NoteEvent]]:
    """
    Returns updated parts with:
      - drum fills at section boundaries (except the very first section)
      - simple section-based velocity scaling for all parts
    """
    if not sections:
        return parts

    new = {k: v[:] for k, v in parts.items()}
    drums = new.get("drums", None)

    # ---- A) Section dynamics
    for sec in sections:
        start_b = _sec_to_beat(float(sec.get("start", 0)), bpm)
        end_b   = _sec_to_beat(float(sec.get("end",   0)), bpm)
        gain = _section_gain(str(sec.get("name", "")))

        for part_name in list(new.keys()):
            evs = new[part_name]
            # lighter touch on drums
            g = gain if part_name in ("melody", "bass") else (0.5*gain + 0.5)
            new[part_name] = _apply_dynamics_to_window(evs, start_b, end_b, g)

    # ---- B) Drum fills at section changes
    if drums is not None and len(sections) > 1:
        for i in range(1, len(sections)):
            # Section start (snap to bar)
            start_b = _snap_to_bar(_sec_to_beat(float(sections[i]["start"]), bpm), beats_per_bar)
            prev_bar_start = start_b - beats_per_bar
            if prev_bar_start < 0:
                continue

            # Snare 16th fill across last beat of previous bar: 4 hits on 3.0/3.25/3.5/3.75
            for ofs in (3.0, 3.25, 3.5, 3.75):
                t = prev_bar_start + ofs
                _add_evt(drums, SNARE, t, t + 0.20, 110)

            # Kick pickup on the & of 4 (3.5)
            t_pick = prev_bar_start + 3.5
            _add_evt(drums, KICK, t_pick, t_pick + 0.18, 115)

            # Crash on section downbeat, ring ~1.5 beats
            _add_evt(drums, CRASH, start_b + 0.0, start_b + 1.50, 118)

        new["drums"] = drums

    return new
