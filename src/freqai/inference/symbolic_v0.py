"""
symbolic_v0: simple, deterministic melody+bass generator.
- Key/mode-aware scale
- Chord-tone emphasis on strong beats
- Passing tones on weak beats
- Outputs NoteEvents [(pitch, start_beat, end_beat, velocity)]
"""

from __future__ import annotations
from typing import List, Tuple, Union, Dict
import re
import math
import pretty_midi

NoteEvent = Tuple[Union[int, str], float, float, int]

# Mode intervals in semitones from tonic (Ionian major, Aeolian minor, etc.)
MODE_INTERVALS: Dict[str, List[int]] = {
    "ionian":      [0,2,4,5,7,9,11],  # major
    "dorian":      [0,2,3,5,7,9,10],
    "phrygian":    [0,1,3,5,7,8,10],
    "lydian":      [0,2,4,6,7,9,11],
    "mixolydian":  [0,2,4,5,7,9,10],
    "aeolian":     [0,2,3,5,7,8,10],  # natural minor
    "locrian":     [0,1,3,5,6,8,10],
}

_CHORD_RE = re.compile(r"^([A-Ga-g](?:#|b)?)(.*)$")

def _pc(note_name: str) -> int:
    """Pitch class (0..11) using pretty_midi for robust spelling."""
    return pretty_midi.note_name_to_number(note_name.replace("♯","#").replace("♭","b")+"4") % 12

def _parse_chord(ch: str) -> Tuple[int, str]:
    """Return (root_pc, quality) with quality in {'maj','min','dim','aug'} (best-effort)."""
    m = _CHORD_RE.match(ch.strip())
    if not m:
        return (0, "maj")
    root = m.group(1).replace("♯","#").replace("♭","b")
    tail = m.group(2).lower()
    q = "maj"
    if "dim" in tail or "o" in tail:
        q = "dim"
    elif "aug" in tail or "+" in tail:
        q = "aug"
    elif "m" in tail and "maj" not in tail:  # covers m, min, m7, etc.
        q = "min"
    return (_pc(root), q)

def _triad_pcs(root_pc: int, quality: str) -> List[int]:
    if quality == "min": return [(root_pc + i) % 12 for i in (0,3,7)]
    if quality == "dim": return [(root_pc + i) % 12 for i in (0,3,6)]
    if quality == "aug": return [(root_pc + i) % 12 for i in (0,4,8)]
    return [(root_pc + i) % 12 for i in (0,4,7)]  # maj

def _scale_pcs(key: str, mode: str) -> List[int]:
    tonic_pc = _pc(key)
    return [ (tonic_pc + iv) % 12 for iv in MODE_INTERVALS.get(mode, MODE_INTERVALS["ionian"]) ]

def _closest_pitch_in_pc(target: int, pc_set: List[int], lo: int, hi: int) -> int:
    """Pick pitch (MIDI number) within [lo,hi] whose pitch-class is in pc_set and nearest to target."""
    best = None
    best_d = 10**9
    for p in range(lo, hi+1):
        if p % 12 in pc_set:
            d = abs(p - target)
            if d < best_d:
                best_d = d
                best = p
    if best is None:
        # fallback: clamp target to range
        return max(lo, min(hi, target))
    return best

def _name(p: int) -> str:
    """Prefer returning MIDI int (our exporter accepts ints); kept for debugging if needed."""
    return pretty_midi.note_number_to_name(p)

def generate_melody_bass(
    anchor: List[str],
    key: str,
    mode: str,
    beats_per_bar: int = 4,
    melody_register: Tuple[int,int] = (60, 76),  # C4..E5
    bass_register: Tuple[int,int] = (40, 55),    # E2..G3
) -> Dict[str, List[NoteEvent]]:
    """
    Returns {'melody': [...], 'bass': [...]}
    - Melody: quarter notes per bar with chord tones on beats 1 & 3, passing tones on 2 & 4
    - Bass: half-note root then fifth each bar
    """
    if not anchor:
        return {"melody": [], "bass": []}

    scale = _scale_pcs(key, mode)
    melody_lo, melody_hi = melody_register
    bass_lo, bass_hi = bass_register

    melody: List[NoteEvent] = []
    bass: List[NoteEvent] = []

    prev_m = None  # previous melody MIDI
    for i, ch in enumerate(anchor):
        r_pc, qual = _parse_chord(ch)
        triad = _triad_pcs(r_pc, qual)

        # Choose a bar center reference: chord root near middle of melody register
        center = (melody_lo + melody_hi) // 2
        root_pitch = _closest_pitch_in_pc(center, [r_pc], melody_lo, melody_hi)
        third_pitch = _closest_pitch_in_pc(root_pitch + 3 if qual in ("min","dim") else root_pitch + 4,
                                           [triad[1] if len(triad)>1 else ((r_pc+4)%12)], melody_lo, melody_hi)
        fifth_pitch = _closest_pitch_in_pc(root_pitch + 7, [triad[2] if len(triad)>2 else ((r_pc+7)%12)], melody_lo, melody_hi)

        # Strong beats: 0 and 2 -> alternate root, third/fifth for contour
        strong_targets = [root_pitch, third_pitch if (i % 2 == 0) else fifth_pitch]
        # Weak beats: choose nearby scale tone (stepwise from previous)
        beat_notes: List[int] = []
        for b in range(beats_per_bar):
            if b % 2 == 0:  # strong
                target = strong_targets[b//2]
                note = _closest_pitch_in_pc(prev_m if prev_m is not None else target, [target % 12], melody_lo, melody_hi)
            else:  # weak: pick a scale tone within a step or two of last
                last = beat_notes[-1] if beat_notes else (prev_m if prev_m is not None else root_pitch)
                # try up or down by 1–2 scale steps
                candidates_pcs = []
                for k in (-2,-1,1,2):
                    idxs = [j for j,pc in enumerate(scale) if pc == (last % 12)]
                    if idxs:
                        j0 = idxs[0]
                        cand_pc = scale[(j0 + k) % len(scale)]
                        candidates_pcs.append(cand_pc)
                # ensure uniqueness
                candidates_pcs = list(dict.fromkeys(candidates_pcs))
                # choose closest pitch class candidate
                choices = [_closest_pitch_in_pc(last, [pc], melody_lo, melody_hi) for pc in candidates_pcs]
                # break ties by closeness to root_pitch (keeps phrase centered)
                if choices:
                    note = min(choices, key=lambda p: (abs(p - last), abs(p - root_pitch)))
                else:
                    note = _closest_pitch_in_pc(last, scale, melody_lo, melody_hi)
            beat_notes.append(note)
            prev_m = note

        # Emit quarter notes for the bar
        bar_start = i * beats_per_bar
        for b, p in enumerate(beat_notes):
            melody.append((int(p), float(bar_start + b), float(bar_start + b + 1), 96 if b % 2 == 0 else 84))

        # Bass: half notes root -> fifth
        # Root in bass register
        bass_root = _closest_pitch_in_pc((bass_lo + bass_hi)//2, [r_pc], bass_lo, bass_hi)
        bass_fifth = _closest_pitch_in_pc(bass_root + 7, [ (r_pc + 7) % 12 ], bass_lo, bass_hi)
        bass.append((int(bass_root), float(bar_start + 0), float(bar_start + 2), 104))
        bass.append((int(bass_fifth), float(bar_start + 2), float(bar_start + 4), 104))

    return {"melody": melody, "bass": bass}