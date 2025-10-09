import re
import json
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import click
from rich import print
from rich.console import Console
from rich.table import Table

# Symbolic generation (v0), groove, exports
from .inference.symbolic_v0 import generate_melody_bass
from .inference.groove_imposer import extract_groove_template, impose_groove_on_events
from .export.midi_export import write_melody_midi
from .synthesis.renderer import write_wav_from_events
from .export.stems import export_stems_and_mix  # <— NEW

# Config support
from .config import load_yaml_config, deep_merge

console = Console()

# --- Canonical keys & modes ---
KEYS = {
    "C","C#","Db","D","D#","Eb","E","F","F#","Gb","G","G#","Ab","A","A#","Bb","B"
}
MODE_ALIASES = {
    "ionian": "ionian", "major": "ionian",
    "dorian": "dorian",
    "phrygian": "phrygian",
    "lydian": "lydian",
    "mixolydian": "mixolydian",
    "aeolian": "aeolian", "minor": "aeolian",
    "locrian": "locrian",
}
QUANTIZE_ALLOWED = {"1/4","1/8","1/12","1/16","1/24","1/32"}

def _canon_key(k: str) -> str:
    k = k.strip().replace("♯","#").replace("♭","b")
    if len(k) >= 2 and k[1] in {"b","#"}:
        k = k[0].upper() + k[1]
    else:
        k = k.upper()
    if k not in KEYS:
        raise click.BadParameter(f"Unsupported key '{k}'. Try one of: {sorted(KEYS)}")
    return k

def _canon_mode(m: str) -> str:
    m = m.strip().lower()
    if m not in MODE_ALIASES:
        raise click.BadParameter(
            f"Unsupported mode '{m}'. Try: major, minor, dorian, phrygian, lydian, mixolydian, aeolian, ionian, locrian"
        )
    return MODE_ALIASES[m]

def _parse_instruments(s: Optional[str]) -> List[str]:
    if not s:
        return []
    items = [x.strip() for x in s.split(",") if x.strip()]
    if len(items) > 4:
        raise click.BadParameter("Please specify at most 4 instruments (comma-separated).")
    return items

def _parse_anchor(s: Optional[str]) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,\-\s]+", s.strip())
    return [p for p in parts if p]

def _parse_duration(s: str) -> int:
    s = s.strip().lower()
    if re.fullmatch(r"\d+", s):
        return int(s)
    if s.endswith("s") and s[:-1].isdigit():
        return int(s[:-1])
    m = re.fullmatch(r"(\d+)m(?:(\d+)s)?", s)
    if m:
        return int(m.group(1))*60 + int(m.group(2) or 0)
    m = re.fullmatch(r"(\d+):([0-5]\d)", s)
    if m:
        return int(m.group(1))*60 + int(m.group(2))
    raise click.BadParameter("Length must look like 60, 60s, 1m30s, 1:00, or 00:45")

def _parse_markers(markers: Tuple[str, ...]) -> List[Tuple[int, str]]:
    out = []
    for m in markers:
        if ":" not in m:
            raise click.BadParameter(f"Marker '{m}' must be 'time:label' (e.g., 30:motif or 00:45:motif)")
        time_str, label = m.rsplit(":", 1)
        time_str, label = time_str.strip().lower(), label.strip()
        if time_str.endswith("s") or re.fullmatch(r"\d+:\d{2}", time_str) or time_str.isdigit():
            t = _parse_duration(time_str)
        else:
            raise click.BadParameter(f"Bad time format in marker '{m}'")
        out.append((t, label))
    out.sort(key=lambda x: x[0])
    return out

def _sec_to_mss(t: int) -> str:
    return f"{t//60:02d}:{t%60:02d}"

def _propose_sections(total_sec: int) -> List[Dict[str, int]]:
    if total_sec < 20:
        return [{"name":"A","start":0,"end":total_sec}]
    cuts = [0,
            round(total_sec*0.10),
            round(total_sec*0.45),
            round(total_sec*0.70),
            round(total_sec*0.85),
            total_sec]
    names = ["Intro","Verse","Chorus","Bridge","Chorus/Outro"]
    sections = []
    for i in range(len(names)):
        s, e = cuts[i], cuts[i+1]
        if e - s > 0:
            sections.append({"name": names[i], "start": s, "end": e})
    return sections

@click.group()
def main():
    """Frequency AI — CLI (M0: symbolic v0 + groove + MIDI/WAV/STEMS)."""
    pass

@main.command("generate")
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="YAML config preset. CLI flags override config values.")
@click.option("--key", required=False, help="Musical key (e.g., C, D#, F#, Bb).")
@click.option("--mode", required=False, help="Mode (major/minor or modal: dorian, lydian, etc.)")
@click.option("--bpm", type=int, required=False, help="Tempo in beats per minute.")
@click.option("--anchor", default="", help='4-bar chord progression, e.g. "Am-G-C-F"')
@click.option("--anchor-bars", type=int, default=4, show_default=True, help="Bars covered by the anchor progression.")
@click.option("--groove", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Path to a reference MIDI loop for groove extraction.")
@click.option("--quantize", default="1/16", show_default=True, help="Groove grid (e.g., 1/8, 1/16).")
@click.option("--humanize", type=float, default=12.0, show_default=True, help="Max micro-timing shift (ms).")
@click.option("--instruments", default="", help='Comma-separated ≤4 instruments (labels only for now).')
@click.option("--length", required=False, help='Total length (e.g., 60s, 1:00, 1m30s).')
@click.option("--marker", multiple=True, help='Repeatable: "time:label" (e.g., 30:motif or 00:45:filter_sweep).')
@click.option("--stems/--no-stems", default=False, help="Export per-instrument stems (melody/bass) and a stereo mix.")
@click.option("--midi/--no-midi", default=False, help="Export MIDI (symbolic_v0 melody+bass).")
@click.option("--wav/--no-wav", default=False, help="Export WAV (symbolic_v0 mix of parts).")
@click.option("--csv-structure/--no-csv-structure", default=False, help="Export CSV of sections.")
@click.option("--outdir", default="outputs", show_default=True, help="Directory for outputs.")
@click.option("--outfile", default=None, help="Base filename (no extension). If omitted, a default is chosen.")
@click.option("--waveform", type=click.Choice(["sine","saw","square","triangle","noise"]), default="saw", show_default=True,
              help="Waveform used for demo audio rendering.")
@click.option("--sr", type=int, default=44100, show_default=True, help="Sample rate for WAV rendering.")
@click.option("--gain", type=float, default=0.22, show_default=True, help="Output gain (0..1) for demo audio.")
def cmd_generate(config, key, mode, bpm, anchor, anchor_bars, groove, quantize, humanize,
                 instruments, length, marker, stems, midi, wav, csv_structure, outdir, outfile,
                 waveform, sr, gain):
    """Plan and export: symbolic_v0 (melody+bass) → optional groove → MIDI/WAV + CSV + STEMS."""

    # Load config and merge with CLI (CLI wins)
    cfg = load_yaml_config(config)
    cli_args: Dict[str, Any] = {
        "key": key, "mode": mode, "bpm": bpm,
        "anchor": anchor, "anchor_bars": anchor_bars, "groove": str(groove) if groove else None,
        "quantize": quantize, "humanize": humanize, "instruments": instruments,
        "length": length, "marker": list(marker) if marker else None,
        "stems": stems, "midi": midi, "wav": wav, "csv_structure": csv_structure,
        "outdir": outdir, "outfile": outfile,
        "waveform": waveform, "sr": sr, "gain": gain,
    }
    cli_args = {k: v for k, v in cli_args.items() if v is not None and v != ""}
    args = deep_merge(cfg, cli_args)

    # Required
    for req in ("key","mode","bpm","length"):
        if req not in args:
            raise click.UsageError("Please specify --key, --mode, --bpm, and --length (CLI or --config).")

    key = _canon_key(str(args["key"]))
    mode = _canon_mode(str(args["mode"]))
    bpm = int(args["bpm"])
    total_sec = _parse_duration(str(args["length"]))

    anchor_chords = _parse_anchor(str(args.get("anchor", "")))
    anchor_bars = int(args.get("anchor_bars", 4))
    instruments = _parse_instruments(str(args.get("instruments", "")))

    groove_path = args.get("groove", None)
    quantize = str(args.get("quantize", "1/16"))
    humanize = float(args.get("humanize", 12.0))
    if quantize not in QUANTIZE_ALLOWED:
        raise click.BadParameter(f"quantize must be one of {sorted(QUANTIZE_ALLOWED)}")

    # Markers
    markers_arg = args.get("marker", [])
    if isinstance(markers_arg, (list, tuple)):
        marker_tuple: Tuple[str, ...] = tuple(str(x) for x in markers_arg)
    elif isinstance(markers_arg, str) and markers_arg:
        marker_tuple = (markers_arg,)
    else:
        marker_tuple = tuple()
    markers = _parse_markers(marker_tuple) if marker_tuple else []

    # Booleans & output
    stems = bool(args.get("stems", False))
    midi = bool(args.get("midi", False))
    wav = bool(args.get("wav", False))
    csv_structure = bool(args.get("csv_structure", False))
    outdir = str(args.get("outdir", "outputs"))
    outfile = args.get("outfile", None)
    waveform = str(args.get("waveform", "saw"))
    sr = int(args.get("sr", 44100))
    gain = float(args.get("gain", 0.22))

    # Plan
    plan = {
        "controls": {
            "key": key, "mode": mode, "bpm": bpm,
            "anchor": anchor_chords, "anchor_bars": anchor_bars,
            "groove_midi": groove_path,
            "quantize": quantize, "humanize_ms": humanize,
            "instruments": instruments,
            "length_sec": total_sec,
            "markers": [{"time_sec": t, "label": lab} for t, lab in markers],
            "exports": {"stems": stems, "midi": midi, "wav": wav, "csv_structure": csv_structure},
            "render": {"waveform": waveform, "sr": sr, "gain": gain}
        },
        "structure": _propose_sections(total_sec),
        "notes": [
            "symbolic_v0: key/mode-aware melody+bass over your anchor (quarter-note melody, half-note bass).",
            "Groove (if provided) imposes micro-timing on parts.",
            "Audio uses a placeholder synth; stems and mixdown available."
        ]
    }

    # Pretty tables
    table = Table(title="Planned Sections", show_lines=False)
    table.add_column("Section", no_wrap=True)
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    for s in plan["structure"]:
        table.add_row(s["name"], _sec_to_mss(s["start"]), _sec_to_mss(s["end"]))
    console.print(table)

    if markers:
        mtab = Table(title="Markers", show_lines=False)
        mtab.add_column("Time"); mtab.add_column("Label")
        for t, lab in markers:
            mtab.add_row(_sec_to_mss(t), lab)
        console.print(mtab)

    # Ensure out dir
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # CSV structure
    if csv_structure:
        base = outfile or "structure"
        csv_path = outdir_path / f"{base}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["section","start_m:ss","end_m:ss","start_sec","end_sec"])
            for s in plan["structure"]:
                writer.writerow([s["name"], _sec_to_mss(s["start"]), _sec_to_mss(s["end"]), s["start"], s["end"]])
        console.print(f"[green]Wrote[/green] CSV structure → {csv_path}")

    # === Generate symbolic_v0 parts ===
    if not anchor_chords:
        console.print("[yellow]No --anchor provided; nothing to render/export.[/yellow]")
        return

    parts = generate_melody_bass(anchor_chords, key=key, mode=mode)  # {"melody":[...], "bass":[...]}

    # Apply groove per-part (so stems + mix share the same feel)
    if groove_path:
        tpl = extract_groove_template(str(groove_path), quantize=quantize)
        for name, evs in list(parts.items()):
            parts[name] = impose_groove_on_events(evs, bpm=bpm, template=tpl, max_ms=humanize)
        console.print(f"[cyan]Applied groove from[/cyan] {groove_path} [cyan]({quantize}, ±{humanize}ms)[/cyan]")

    # Flattened events for single-track MIDI/WAV (kept for convenience)
    events = parts["melody"] + parts["bass"]

    # MIDI export (single-track for now)
    if midi:
        base = outfile or "demo"
        midi_path = outdir_path / f"{base}.mid"
        write_melody_midi(events, bpm=bpm, out_path=str(midi_path), instrument_name="melody_bass_v0", program=0)
        console.print(f"[green]Wrote[/green] MIDI (melody+bass v0) → {midi_path}")

    # WAV export (mix of parts with the basic renderer)
    if wav:
        base = outfile or "demo"
        wav_path = outdir_path / f"{base}.wav"
        write_wav_from_events(events, bpm=bpm, out_path=str(wav_path), sr=sr, wave=waveform, gain=gain)
        console.print(f"[green]Wrote[/green] WAV ({waveform}, {sr} Hz) → {wav_path}")

    # STEMS export + stereo mixdown
    if stems:
        base = outfile or "demo"
        render_opts = {
            "sr": sr,
            "defaults": {"wave": waveform, "gain": gain, "pan": 0.5},  # CLI waveform/gain apply unless per-part overrides
            # per_part left to stems.py defaults (melody triangle @0.22/pan0.65; bass saw @0.28/pan0.35)
        }
        result = export_stems_and_mix(parts, bpm=bpm, outdir=outdir_path, base=base, render_opts=render_opts)
        console.print(f"[green]Wrote[/green] stems & mix → {result['mix']}")
        for k, v in result.items():
            if k.startswith("stem_"):
                console.print(f"  - {k.replace('stem_','')}: {v}")

    # Summary
    console.rule("[bold]Controls")
    print(json.dumps(plan["controls"], indent=2))
    console.rule("[bold]Next steps")
    for n in plan["notes"]:
        print(f"- {n}")

if __name__ == "__main__":
    main()
