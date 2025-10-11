# -*- coding: utf-8 -*-
"""
MIDI export utilities for Frequency AI.

- write_melody_midi(events, bpm, out_path, instrument_name="melody", program=0)
- write_multitrack_midi(parts, bpm, out_path, programs=None, drum_flags=None)

Notes:
- All times are expressed in BEATS in the input events and converted to seconds via 60/bpm.
- NoteEvent = (pitch: int|str, start_beat: float, end_beat: float, velocity: int)
- For drums, pass drum_flags={"drums": True} (GM mapping expected, e.g., 36 kick, 38 snare, 42 hat).
"""

from typing import Dict, List, Tuple, Union, Optional
import os
import time

import pretty_midi

NoteEvent = Tuple[Union[int, str], float, float, int]


# ---------------------------------------------------------------------------
# Safer write on Windows: write tmp, then atomic-ish replace, with retries
# ---------------------------------------------------------------------------
def _safe_pm_write(pm: pretty_midi.PrettyMIDI, out_path: str,
                   retries: int = 3, delay: float = 0.5) -> str:
    """
    Write PrettyMIDI `pm` to `out_path` more safely on Windows:
      1) write to out_path + ".tmp"
      2) attempt to remove existing out_path (ignore failures)
      3) os.replace(tmp, out_path)
      4) retry on PermissionError
      5) fallback to timestamped filename if all retries fail

    Returns the path actually written.
    """
    tmp = out_path + ".tmp"
    for _ in range(retries):
        try:
            pm.write(tmp)
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    # If removal fails (locked), still try replace below.
                    pass
            os.replace(tmp, out_path)
            return out_path
        except PermissionError:
            time.sleep(delay)
        except Exception:
            # Best effort cleanup of tmp on unexpected errors
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    base, ext = os.path.splitext(out_path)
    alt = f"{base}.{int(time.time())}{ext}"
    pm.write(alt)
    return alt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _beats_to_seconds(b: float, bpm: int) -> float:
    return float(b) * (60.0 / float(bpm))


def _add_note(instr: pretty_midi.Instrument, pitch: int, s_beats: float, e_beats: float,
              velocity: int, bpm: int) -> None:
    start = _beats_to_seconds(s_beats, bpm)
    end = _beats_to_seconds(e_beats, bpm)
    if end <= start:
        end = start + 1e-4
    velocity = max(1, min(int(velocity), 127))
    note = pretty_midi.Note(velocity=velocity, pitch=int(pitch), start=start, end=end)
    instr.notes.append(note)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def write_melody_midi(
    events: List[NoteEvent],
    bpm: int,
    out_path: str,
    instrument_name: str = "melody",
    program: int = 0,
) -> str:
    """
    Write a single-track MIDI from a flat list of events.
    Non-int pitches are skipped.
    """
    pm = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(program=int(program), name=instrument_name, is_drum=False)
    for p, s, e, v in events:
        if isinstance(p, int):
            _add_note(instr, p, s, e, v, bpm)
    pm.instruments.append(instr)
    written = _safe_pm_write(pm, out_path)
    return written


def write_multitrack_midi(
    parts: Dict[str, List[NoteEvent]],
    bpm: int,
    out_path: str,
    programs: Optional[Dict[str, int]] = None,
    drum_flags: Optional[Dict[str, bool]] = None,
) -> str:
    """
    Write a multitrack MIDI file from named parts.

    Args:
        parts: dict like {"melody":[...], "bass":[...], "drums":[...]}
        bpm: tempo
        out_path: destination path
        programs: optional GM programs per part (ignored for drums)
        drum_flags: e.g., {"drums": True}

    Returns:
        The path actually written (may differ if fallback name used).
    """
    programs = programs or {}
    drum_flags = drum_flags or {}
    pm = pretty_midi.PrettyMIDI()

    for name, events in parts.items():
        is_drum = bool(drum_flags.get(name, False))
        program = int(programs.get(name, 0)) if not is_drum else 0
        instr = pretty_midi.Instrument(program=program, name=name, is_drum=is_drum)

        for p, s, e, v in events:
            if isinstance(p, int):
                _add_note(instr, p, s, e, v, bpm)
            # If pitch is a string, skip (Or you can add parsing if needed)

        pm.instruments.append(instr)

    written = _safe_pm_write(pm, out_path)
    return written
