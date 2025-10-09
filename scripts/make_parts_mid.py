from pathlib import Path
from freqai.inference.symbolic_v0 import generate_melody_bass
from freqai.inference.groove_imposer import extract_groove_template, impose_groove_on_events
from freqai.export.midi_export import write_multitrack_midi

anchor = ["Am","G","C","F"]
parts = generate_melody_bass(anchor, "D", "dorian")
tpl = extract_groove_template("demo_anchor.mid", "1/16")
parts = {k: impose_groove_on_events(v, 112, tpl, 12) for k, v in parts.items()}

Path("outputs").mkdir(exist_ok=True)
out = write_multitrack_midi(parts, bpm=112, out_path="outputs/v0.parts.mid",
                            programs={"melody":73, "bass":34})
print("wrote", out)