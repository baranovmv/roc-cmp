"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys

from .audio import load_wav
from .compare import Result, compare


def _fmt(r: Result, events: bool) -> str:
    lines = [f"score: {r.score:.1f} / 100"]
    if not r.detected:
        lines.append(f"verdict: {r.reason}")
        return "\n".join(lines)
    lines += [
        f"reference: {r.ref_duration:.1f}s, dump: {r.dump_duration:.1f}s",
        f"active dump span: {r.active_start:.2f}s .. {r.active_end:.2f}s"
        f" (reference offset {r.ref_offset:+.3f}s, drift {r.drift_ms:.1f} ms)",
        f"reference coverage: {100 * r.coverage:.1f}%",
        f"glitches: {r.glitch_count} ({100 * r.bad_fraction:.2f}% of active time affected),"
        f" median spectral distance {r.median_lsd:.2f} dB",
    ]
    if r.jumps:
        lines.append(f"timeline jumps: {r.jumps}")
    if events and r.glitches:
        lines.append("glitch events (dump time -> reference time):")
        for g in r.glitches:
            lines.append(
                f"  {g.start:8.3f}s .. {g.end:8.3f}s  ref {g.ref_start:8.3f}s  {g.kind:10s} {g.lsd:5.1f} dB"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="roc-cmp",
        description="Score how much a dumped (received) WAV file was distorted relative to the "
        "reference WAV. 100 = identical, 0 = no transmission detected. Leading/trailing silence "
        "and a partially sent file are not penalised; gaps and distortion inside the received "
        "span are.",
    )
    p.add_argument("reference", help="reference WAV file")
    p.add_argument("dump", help="dumped (received) WAV file")
    p.add_argument("--min-score", type=float, metavar="N", help="exit with status 1 if score is below N")
    p.add_argument("--events", "-e", action="store_true", help="list every detected glitch")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p.add_argument(
        "--threshold",
        type=float,
        default=2.5,
        metavar="DB",
        help="log-spectral distance (dB) above which a frame is a glitch (default: %(default)s)",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.1,
        metavar="SEC",
        help="listening time ruined on each side of a glitch (default: %(default)s)",
    )
    a = p.parse_args(argv)

    try:
        ref, dump = load_wav(a.reference), load_wav(a.dump)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    r = compare(ref, dump, lsd_threshold=a.threshold, margin_s=a.margin)
    if a.json:
        print(json.dumps(r.to_dict(), indent=2))
    else:
        print(_fmt(r, a.events))
    if a.min_score is not None and r.score < a.min_score:
        return 1
    return 0
