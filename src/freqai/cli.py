import re
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import click
from rich import print
from rich.console import Console
from rich.table import Table

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
    k = k.upper()
    if k in {"DB","EB","GB","AB","BB"}:  # normalize flat caps to title-case
        k = k[0] + "b"
    if k not in KEYS:
        raise click.BadParameter(f"Unsupported key '{k}'. Try one of: {sorted(KEYS)}")
    return k

def _canon_mode(m: str) -> str:
    m = m.strip().lower()
    if m not in MODE_ALIASES:
        raise click.BadParameter(
            f"Unsupported mode '{m}'. Try one of: major, minor, dorian, phrygian, lydian, mixolydian, aeolian, ionian, locrian"
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
    # allow: "Am-G-C-F" or "Am G C F" or "Am, G, C, F"
    parts = re.split(r"[,\-\s]+", s.strip())
    parts = [p for p in parts if p]
    return parts

def _parse_duration(s: str) -> int:
    """
    Accepts: "60" (seconds), "60s", "1m", "1m30s", "1:00", "00:45", "2:03"
    Returns seconds (int)
    """
    s = s.strip().lower()
    if re.fullmatch(r"\d+", s):
        return int(s)
    if s.endswith("s") and s[:-1].isdigit():
        return int(s[:-1])
    m = re.fullmatch(r"(\d+)m(?:(\d+)s)?", s)
    if m:
        mins = int(m.group(1))
        secs = int(m.group(2) or 0)
        return mins*60 + secs
    m = re.fullmatch(r"(\d+):([0-5]\d)", s)
    if m:
        return int(m.group(1))*60 + int(m.group(2))
    raise click.BadParameter("Length must look like 60, 60s, 1m30s, 1:00, or 00:45")

def _parse_markers(markers: Tuple[str]) -> List[Tuple[int, str]]:
    """
    Accepts repeated --marker like:
      --marker "30:intro_motif" or --marker "00:45:filter_sweep" or --marker "30s:drop"
    Returns list of (time_sec, label)
    """
    out = []
    for m in markers:
        if ":" not in m:
            raise click.BadParameter(f"Marker '{m}' must be 'time:label' (e.g., 30:motif or 00:45:motif)")
        time_str, label = m.split(":", 1)
        time_str = time_str.strip().lower()
        label = label.strip()
        if time_str.endswith("s"):
            t = _parse_duration(time_str)
        elif re.fullmatch(r"\d+:\d{2}", time_str) or time_str.isdigit():
            t = _parse_duration(time_str)
        else:
            raise click.BadParameter(f"Bad time format in marker '{m}'")
        out.append((t, label))
    # sort by time
    out.sort(key=lambda x: x[0])
    return out

def _sec_to_mss(t: int) -> str:
    return f"{t//60:02d}:{t%60:02d}"

def _propose_sections(total_sec: int) -> List[Dict]:
    """
    Very simple sectioning heuristic just for the dry-run:
    Intro (10%), Verse (35%), Chorus (25%), Bridge (15%), Chorus/Outro (15%)
    """
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
        if e - s <= 0:  # guard
            continue
        sections.append({"name": names[i], "start": s, "end": e})
    return sections

@click.group()
def main():
    """Frequency AI — CLI (M0 dry-run planner)."""
    pass

@main.command("generate")
@click.option("--key", required=True, help="Musical key (e.g., C, D#, F#, Bb).")
@click.option("--mode", required=True, help="Mode (major/minor or modal: dorian, lydian, etc.)")
@click.option("--bpm", type=int, required=True, help="Tempo in beats per minute.")
@click.option("--anchor", default="", help='4-bar chord progression, e.g. "Am-G-C-F"')
@click.option("--anchor-bars", type=int, default=4, show_default=True, help="Bars covered by the anchor progression.")
@click.option("--groove", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Path to a reference MIDI loop for groove extraction.")
@click.option("--quantize", default="1/16", show_default=True, help="Quantization grid (e.g., 1/8, 1/16).")
@click.option("--humanize", type=float, default=12.0, show_default=True, help="Humanize micro-timing (ms).")
@click.option("--instruments", default="", help='Comma-separated ≤4 instruments, e.g. "analog_bass,e_gtr_bigverb,jazz_kit,cin_pad"')
@click.option("--length", required=True, help='Total length (e.g., 60s, 1:00, 1m30s).')
@click.option("--marker", multiple=True, help='Repeatable: "time:label" (e.g., 30:motif or 00:45:filter_sweep).')
@click.option("--stems/--no-stems", default=False, help="Export per-instrument stems (when audio is implemented).")
@click.option("--midi/--no-midi", default=False, help="Export MIDI (when generation is implemented).")
@click.option("--wav/--no-wav", default=False, help="Export final WAV (when audio is implemented).")
@click.option("--csv-structure/--no-csv-structure", default=False, help="Export CSV of sections/chords/motifs.")
def cmd_generate(key, mode, bpm, anchor, anchor_bars, groove, quantize, humanize,
                 instruments, length, marker, stems, midi, wav, csv_structure):
    """Plan a generation (no audio yet). Validates inputs and prints a structured plan."""
    # Validate key/mode
    key = _canon_key(key)
    mode = _canon_mode(mode)
    # Validate quantize
    if quantize not in QUANTIZE_ALLOWED:
        raise click.BadParameter(f"quantize must be one of {sorted(QUANTIZE_ALLOWED)}")
    # Parse simple args
    insts = _parse_instruments(instruments)
    anchor_chords = _parse_anchor(anchor)
    total_sec = _parse_duration(length)
    markers = _parse_markers(marker) if marker else []

    plan = {
        "controls": {
            "key": key, "mode": mode, "bpm": bpm,
            "anchor": anchor_chords, "anchor_bars": anchor_bars,
            "groove_midi": str(groove) if groove else None,
            "quantize": quantize, "humanize_ms": humanize,
            "instruments": insts,
            "length_sec": total_sec,
            "markers": [{"time_sec": t, "label": lab} for t, lab in markers],
            "exports": {
                "stems": stems, "midi": midi, "wav": wav, "csv_structure": csv_structure
            }
        },
        "structure": _propose_sections(total_sec),
        "notes": [
            "This is a dry-run. No audio is generated at M0.",
            "Next milestones: symbolic generation (MIDI), groove imposition, then audio rendering."
        ]
    }

    # Pretty table for sections
    table = Table(title="Planned Sections", show_lines=False)
    table.add_column("Section", no_wrap=True)
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    for s in plan["structure"]:
        table.add_row(s["name"], _sec_to_mss(s["start"]), _sec_to_mss(s["end"]))
    console.print(table)

    # Markers table (if any)
    if markers:
        mtab = Table(title="Markers", show_lines=False)
        mtab.add_column("Time")
        mtab.add_column("Label")
        for t, lab in markers:
            mtab.add_row(_sec_to_mss(t), lab)
        console.print(mtab)

    # Controls summary JSON
    console.rule("[bold]Controls")
    print(json.dumps(plan["controls"], indent=2))
    console.rule("[bold]Next steps")
    for n in plan["notes"]:
        print(f"- {n}")

if __name__ == "__main__":
    main()