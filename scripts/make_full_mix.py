from pathlib import Path
from freqai.inference.symbolic_v0 import generate_melody_bass
from freqai.inference.drums_v0 import generate_drums_v0
from freqai.export.stems import export_stems_and_mix
from freqai.export.midi_export import write_multitrack_midi

# 4 bars over Am-G-C-F, D Dorian, 112 BPM
anchor = ["Am","G","C","F"]
parts = generate_melody_bass(anchor, key="D", mode="dorian")
parts["drums"] = generate_drums_v0(anchor)  # kick/snare/hat 8ths

# write stems + stereo mix
Path("outputs").mkdir(exist_ok=True)
out = export_stems_and_mix(
    parts, bpm=112, outdir="outputs", base="v0_full",
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

# multitrack MIDI (melody/bass + drums as drum track)
write_multitrack_midi(
    parts, bpm=112, out_path="outputs/v0_full.parts.mid",
    programs={"melody": 73, "bass": 34},
    drum_flags={"drums": True}
)
print("midi:", "outputs/v0_full.parts.mid")