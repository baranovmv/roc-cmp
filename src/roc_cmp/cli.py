"""Command line interface."""

import argparse
import json
import sys
from typing import List, Optional

from .audio import load_wav
from .compare import Result, compare


def _fmt(r: Result, events: bool) -> str:
    lines = ["score: {:.1f} / 100".format(r.score)]
    if not r.detected:
        lines.append("verdict: {}".format(r.reason))
        return "\n".join(lines)
    lines += [
        "reference: {:.1f}s, dump: {:.1f}s".format(r.ref_duration, r.dump_duration),
        "active dump span: {:.2f}s .. {:.2f}s (reference offset {:+.3f}s, drift {:.1f} ms)".format(
            r.active_start, r.active_end, r.ref_offset, r.drift_ms
        ),
        "reference coverage: {:.1f}%".format(100 * r.coverage),
        "glitches: {} ({:.2f}% of active time affected), median spectral distance {:.2f} dB".format(
            r.glitch_count, 100 * r.bad_fraction, r.median_lsd
        ),
    ]
    if r.jumps:
        lines.append("timeline jumps: {}".format(r.jumps))
    if events and r.glitches:
        lines.append("glitch events (dump time -> reference time):")
        for g in r.glitches:
            lines.append(
                "  {:8.3f}s .. {:8.3f}s  ref {:8.3f}s  {:10s} {:5.1f} dB".format(
                    g.start, g.end, g.ref_start, g.kind, g.lsd
                )
            )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
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
        print("error: {}".format(e), file=sys.stderr)
        return 2

    r = compare(ref, dump, lsd_threshold=a.threshold, margin_s=a.margin)
    if a.json:
        print(json.dumps(r.to_dict(), indent=2))
    else:
        print(_fmt(r, a.events))
    if a.min_score is not None and r.score < a.min_score:
        return 1
    return 0
