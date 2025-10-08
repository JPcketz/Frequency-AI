"""
Simple anchor→demo melody writer (M0).
Takes a 4-bar chord anchor and writes one sustained root note per bar as a MIDI.
"""
from typing import List, Tuple, Union
import re
from ..export.midi_export import write_melody_midi

NoteEvent = Tuple[Union[int, str], float, float, int]

_ROOT_RE = re.compile(r"^([A-Ga-g](?:#|b)?)(.*)$")

def _root_name(chord: str) -> str:
    m = _ROOT_RE.match(chord.strip())
    if not m:
        return "C"  # fallback
    root = m.group(1).replace("♯", "#").replace("♭", "b")
    # Normalize to proper case like "Bb", "C#", "F"
    root = root[0].upper() + (root[1:] if len(root) > 1 else "")
    return root

def anchor_to_demo_melody(anchor: List[str], beats_per_bar: int = 4, octave: int = 3) -> List[NoteEvent]:
    """Map each chord to a sustained root note for one bar."""
    events: List[NoteEvent] = []
    beat = 0.0
    for ch in anchor:
        root = _root_name(ch)
        note_name = f"{root}{octave}"
        start = beat
        end = beat + beats_per_bar
        events.append((note_name, start, end, 100))
        beat = end
    return events

def write_anchor_demo_midi(anchor: List[str], bpm: int, out_path: str) -> str:
    events = anchor_to_demo_melody(anchor)
    write_melody_midi(events, bpm=bpm, out_path=out_path, instrument_name="anchor_demo", program=0)
    return out_path