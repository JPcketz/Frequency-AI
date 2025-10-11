"""
voiceleading_v0: simple harmonic cleanup + smoother motion.

- Prioritizes chord tones on strong beats (bass esp. on beats 1 & 3 of each 4/4 bar).
- Chooses nearest chord-tone pitch to minimize leaps (voice-leading).
- Keeps bass in a sensible range (E1..C3 by default).
- Optionally nudges melody strong-beat notes toward chord tones (light touch).

NoteEvent = (pitch: int|str, start_beat: float, end_beat: float, velocity: int)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Union

NoteEvent = Tuple[Union[int, str], float, float, int]

# --- pitch helpers -----------------------------------------------------------

NOTE2PC = {
    "C":0, "C#":1, "Db":1, "D":2, "D#":3, "Eb":3, "E":4, "F":5,
    "F#":6, "Gb":6, "G":7, "G#":8, "Ab":8, "A":9, "A#":10, "Bb":10, "B":11
}

MODE_PCS = {
    "ionian":      [0,2,4,5,7,9,11],  # major
    "dorian":      [0,2,3,5,7,9,10],
    "phrygian":    [0,1,3,5,7,8,10],
    "lydian":      [0,2,4,6,7,9,11],
    "mixolydian":  [0,2,4,5,7,9,10],
    "aeolian":     [0,2,3,5,7,8,10],  # natural minor
    "locrian":     [0,1,3,5,6,8,10],
}

def _pc_from_note(n: Union[int, str]) -> int:
    if isinstance(n, int):
        return n % 12
    n = n.strip().replace("♯","#").replace("♭","b")
    if len(n) >= 2 and n[1] in {"#","b"}:
        key = n[0].upper() + n[1]
    else:
        key = n[0].upper()
    return NOTE2PC.get(key, 0)

def _root_pc(key: str) -> int:
    return _pc_from_note(key)

def _scale_pcs_for_key_mode(key: str, mode: str) -> List[int]:
    root = _root_pc(key)
    degrees = MODE_PCS.get(mode.lower(), MODE_PCS["ionian"])
    return [ (root + d) % 12 for d in degrees ]

def _parse_chord(ch: str) -> Tuple[int, str]:
    """
    Accepts things like 'Am', 'Amin', 'A', 'C', 'G7' -> treat '7' as major triad base.
    Returns (root_pc, 'maj'|'min'|'dim')
    """
    ch = ch.strip()
    if not ch:
        return (0, "maj")
    r = ch[0].upper()
    q = ch[1:].strip()
    if len(ch) >= 2 and ch[1] in "#b":
        r = ch[:2].title()
        q = ch[2:].strip()
    root = _pc_from_note(r)
    ql = q.lower()
    if ql.startswith(("m","min","-")) and not ql.startswith("maj"):
        qual = "min"
    elif "dim" in ql or "o" in ql:
        qual = "dim"
    else:
        qual = "maj"
    return (root, qual)

def _chord_tones(root_pc: int, quality: str) -> List[int]:
    if quality == "min":
        ints = [0,3,7]          # m triad
    elif quality == "dim":
        ints = [0,3,6]          # diminished
    else:
        ints = [0,4,7]          # M triad (default)
    return [ (root_pc + i) % 12 for i in ints ]

def _nearest_pc_pitch(prev: int, pc_set: List[int], lo: int, hi: int) -> int:
    """
    Choose MIDI note with pitch-class in pc_set nearest to prev (within [lo,hi]).
    """
    best = None
    best_dist = 10**9
    for pc in pc_set:
        # find octave so it's near prev
        base = prev - ((prev - pc) % 12)
        candidates = [base + k*12 for k in (-2,-1,0,1,2,3)]
        for c in candidates:
            if c < lo: continue
            if c > hi: continue
            d = abs(c - prev)
            if d < best_dist:
                best_dist = d
                best = c
    if best is None:
        # fall back to clamping prev into range
        best = min(max(prev, lo), hi)
    return best

def _is_near_integer(x: float, eps: float = 1e-3) -> bool:
    return abs(x - round(x)) <= eps

# --- main API ----------------------------------------------------------------

def improve_voice_leading(
    parts: Dict[str, List[NoteEvent]],
    anchor: List[str],
    key: str,
    mode: str,
    beats_per_bar: int = 4,
    bass_lo: int = 40,     # E1
    bass_hi: int = 60,     # C4-? keep conservative for tightness
    melody_lo: int = 60,   # C4
    melody_hi: int = 84,   # C6
    strong_beats: Tuple[int,int] = (0,2),
    adjust_melody_on_strong_beats: bool = False,
) -> Dict[str, List[NoteEvent]]:
    """
    Enforce chord tones on strong beats and minimize leaps.
    Returns a NEW parts dict; original is not mutated.
    """
    if not anchor:
        return parts

    # Precompute chord tone pcs per bar
    chord_pcs_by_bar: List[List[int]] = []
    for ch in anchor:
        r, q = _parse_chord(ch)
        chord_pcs_by_bar.append(_chord_tones(r, q))

    scale_pcs = _scale_pcs_for_key_mode(key, mode)

    out: Dict[str, List[NoteEvent]] = {k: v[:] for k, v in parts.items()}

    # --- Bass cleanup ---
    if "bass" in out and out["bass"]:
        new_bass: List[NoteEvent] = []
        prev = None
        for p, s, e, v in out["bass"]:
            # derive bar and beat index
            bar = int(s // beats_per_bar) if beats_per_bar > 0 else 0
            chord_pcs = chord_pcs_by_bar[bar % len(chord_pcs_by_bar)]
            # pick target pitch
            if isinstance(p, str):
                cur = 45  # arbitrary seed if string; will be replaced
            else:
                cur = int(p)
            if prev is None:
                # first note: choose a chord tone near current (or center of range)
                seed = cur if bass_lo <= cur <= bass_hi else (bass_lo + bass_hi)//2
                tgt = _nearest_pc_pitch(seed, chord_pcs, bass_lo, bass_hi)
            else:
                # strong beats -> snap to chord tone nearest previous; weak beats -> keep or gentle move
                beat_in_bar = s - bar * beats_per_bar
                if int(round(beat_in_bar)) in strong_beats and _is_near_integer(beat_in_bar):
                    tgt = _nearest_pc_pitch(prev, chord_pcs, bass_lo, bass_hi)
                else:
                    # keep within range, optionally drift a step toward nearest scale tone
                    target_pc = min(scale_pcs, key=lambda pc: min((abs((prev % 12) - pc), 12-abs((prev % 12) - pc))))
                    tgt = _nearest_pc_pitch(prev, [target_pc], bass_lo, bass_hi)
            prev = tgt
            new_bass.append((tgt, s, e, v))
        out["bass"] = new_bass

    # --- (Optional) Melody nudge on strong beats ---
    if adjust_melody_on_strong_beats and "melody" in out and out["melody"]:
        new_melody: List[NoteEvent] = []
        prev_m = None
        for p, s, e, v in out["melody"]:
            bar = int(s // beats_per_bar) if beats_per_bar > 0 else 0
            chord_pcs = chord_pcs_by_bar[bar % len(chord_pcs_by_bar)]
            if isinstance(p, int):
                cur = p
            else:
                cur = 72  # seed if string
            beat_in_bar = s - bar * beats_per_bar
            if int(round(beat_in_bar)) in strong_beats and _is_near_integer(beat_in_bar):
                # snap to nearest chord tone in melody range (keeps hook stable)
                tgt = _nearest_pc_pitch(cur if prev_m is None else prev_m, chord_pcs, melody_lo, melody_hi)
                prev_m = tgt
                new_melody.append((tgt, s, e, v))
            else:
                # keep as-is (or clamp range)
                tgt = min(max(cur, melody_lo), melody_hi)
                prev_m = tgt
                new_melody.append((tgt, s, e, v))
        out["melody"] = new_melody

    return out
