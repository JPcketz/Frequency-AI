"""
song_v0: expand a short anchor (e.g., 4 bars) to a full song length by
tiling the chords to the required bar count, then generating parts.

- Computes total bars from (length_sec, bpm) assuming 4/4 (4 beats/bar).
- Repeats the anchor to fill those bars.
- Generates melody + bass (symbolic_v0) and optional drums (drums_v0).
- Returns a dict of parts: {"melody":[...], "bass":[...], "drums":[...]?}
"""

from __future__ import annotations
from math import ceil
from typing import List, Dict, Tuple, Union

NoteEvent = Tuple[Union[int, str], float, float, int]

def _bars_from_length(length_sec: int | float, bpm: int, beats_per_bar: int = 4) -> int:
    """How many 4/4 bars fit in the requested length? (ceil to avoid under-run)"""
    spb = 60.0 / float(bpm)                # seconds per beat
    spbar = spb * float(beats_per_bar)     # seconds per bar
    bars = int(ceil(float(length_sec) / spbar))
    return max(bars, 1)

def _tile_anchor(anchor: List[str], total_bars: int) -> List[str]:
    """Repeat the anchor progression to match total_bars."""
    if not anchor:
        return []
    tiled = [anchor[i % len(anchor)] for i in range(total_bars)]
    return tiled

def generate_song_v0(
    anchor: List[str],
    key: str,
    mode: str,
    length_sec: int | float,
    bpm: int,
    include_drums: bool = True,
    beats_per_bar: int = 4,
) -> Dict[str, List[NoteEvent]]:
    """
    Expand the anchor to the number of bars implied by (length_sec, bpm),
    then generate melody/bass (and drums) over the whole structure.
    """
    total_bars = _bars_from_length(length_sec, bpm, beats_per_bar=beats_per_bar)
    tiled = _tile_anchor(anchor, total_bars)
    if not tiled:
        return {"melody": [], "bass": []}

    # Lazy imports to avoid circulars
    from .symbolic_v0 import generate_melody_bass
    parts = generate_melody_bass(tiled, key=key, mode=mode)

    if include_drums:
        try:
            from .drums_v0 import generate_drums_v0
            parts["drums"] = generate_drums_v0(tiled, beats_per_bar=beats_per_bar)
        except Exception:
            # Drums optional; swallow if not available
            pass

    return parts