"""
Minimal MIDI export helper for Frequency AI (M0).
Converts note events (pitch, start_beat, end_beat, velocity) to a MIDI file.

pitch: int MIDI note number (e.g., 60 for C4) OR note name string ("C4", "G#3")
start_beat/end_beat: beats from start (float allowed)
velocity: 1..127
"""
from typing import Iterable, Tuple, Union
import pretty_midi

NoteEvent = Tuple[Union[int, str], float, float, int]

def write_melody_midi(
    note_events: Iterable[NoteEvent],
    bpm: int,
    out_path: str,
    instrument_name: str = "melody",
    program: int = 0,
) -> None:
    """
    Write a single-track melody MIDI using the given BPM.

    Example:
        events = [
            ("A3", 0.0, 1.0, 96),
            ("G3", 1.0, 2.0, 96),
            ("C4", 2.0, 3.0, 100),
            ("F3", 3.0, 4.0, 100),
        ]
        write_melody_midi(events, bpm=112, out_path="demo.mid")
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    inst = pretty_midi.Instrument(program=program, name=instrument_name)

    sec_per_beat = 60.0 / float(bpm)

    for pitch, s_beat, e_beat, vel in note_events:
        if isinstance(pitch, str):
            pitch_num = pretty_midi.note_name_to_number(pitch)
        else:
            pitch_num = int(pitch)
        start_sec = float(s_beat) * sec_per_beat
        end_sec = float(e_beat) * sec_per_beat
        end_sec = max(end_sec, start_sec + 1e-3)  # avoid zero-length
        vel = max(1, min(int(vel), 127))
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch_num, start=start_sec, end=end_sec))

    pm.instruments.append(inst)
    pm.write(out_path)