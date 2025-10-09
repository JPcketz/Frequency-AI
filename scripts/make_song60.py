from pathlib import Path
from freqai.inference.song_v0 import generate_song_v0
from freqai.inference.groove_imposer import extract_groove_template, impose_groove_on_events
from freqai.export.stems import export_stems_and_mix
from freqai.export.midi_export import write_multitrack_midi

anchor = ["Am","G","C","F"]
key, mode, bpm, length_sec = "D", "dorian", 112, 60

# generate full-length parts
parts = generate_song_v0(anchor, key=key, mode=mode, length_sec=length_sec, bpm=bpm, include_drums=True)

# optional groove
try:
    tpl = extract_groove_template("demo_anchor.mid", "1/16")
    for k in list(parts.keys()):
        parts[k] = impose_groove_on_events(parts[k], bpm=bpm, template=tpl, max_ms=12)
except Exception as e:
    print("Skipping groove:", e)

Path("outputs").mkdir(exist_ok=True)

# stems + stereo mix
out = export_stems_and_mix(
    parts, bpm=bpm, outdir="outputs", base="song60",
    render_opts={
        "sr": 44100,
        "defaults": {"wave": "saw", "gain": 0.22, "pan": 0.5},
        "per_part": {
            "melody": {"wave": "triangle", "gain": 0.22, "pan": 0.65},
            "bass":   {"wave": "saw",      "gain": 0.28, "pan": 0.35},
            "drums":  {"wave": "noise",    "gain": 0.14, "pan": 0.50},
        },
    }
)
print("mix:", out["mix"])

# multitrack MIDI
write_multitrack_midi(
    parts, bpm=bpm, out_path="outputs/song60.parts.mid",
    programs={"melody": 73, "bass": 34},
    drum_flags={"drums": True}
)
print("midi:", "outputs/song60.parts.mid")