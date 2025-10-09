"""
drums_v0: simple rock/pop 8th-note kit pattern over 4/4.
- Kick on 1 & 3
- Snare on 2 & 4 (backbeat)
- Closed hi-hat on 8ths (short notes)
Returns NoteEvents: (midi_pitch, start_beat, end_beat, velocity)
General MIDI drum notes (channel semantics handled elsewhere):
  36 = Kick, 38 = Snare, 42 = Closed Hat, 46 = Open Hat
"""

from __future__ import annotations
from typing import List, Tuple, Union

NoteEvent = Tuple[Union[int, str], float, float, int]

KICK = 36
SNARE = 38
CHH = 42
OHH = 46

def generate_drums_v0(anchor: List[str], beats_per_bar: int = 4) -> List[NoteEvent]:
    """
    Make a basic kit groove with length = len(anchor) bars (4/4).
    For now we ignore chord labels and just track bar count.
    """
    bars = len(anchor) if anchor else 4
    events: List[NoteEvent] = []

    for i in range(bars):
        bar_start = i * beats_per_bar

        # --- Backbeat snare on beats 2 & 4 (indices 1,3) ---
        events.append((SNARE, float(bar_start + 1.0), float(bar_start + 1.25), 108))
        events.append((SNARE, float(bar_start + 3.0), float(bar_start + 3.25), 112))

        # --- Kick on beats 1 & 3 (indices 0,2) ---
        events.append((KICK, float(bar_start + 0.0), float(bar_start + 0.25), 118))
        events.append((KICK, float(bar_start + 2.0), float(bar_start + 2.25), 112))

        # (Optional) small pickup before beat 3 every other bar
        if i % 2 == 1:
            events.append((KICK, float(bar_start + 1.75), float(bar_start + 1.875), 96))

        # --- Hi-hats on 8ths across the bar (every 0.5 beat) ---
        # Short notes so our simple renderer feels percussive enough
        for n in range(8):  # 8x 8th-notes
            t = bar_start + n * 0.5
            vel = 94 if (n % 2 == 0) else 84  # slight accent on the downbeats
            events.append((CHH, float(t), float(t + 0.25), vel))

        # (Optional) open hat on the & of 4 to lead into next bar
        events.append((OHH, float(bar_start + 3.5), float(bar_start + 3.875), 96))

    return events