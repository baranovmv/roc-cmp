"""Locate the dump inside the reference and track slow time drift.

The dump normally keeps the reference timeline (lost packets become zeros), but it
starts at an arbitrary point, may be silent at the edges, and drifts by a few ms
over a minute because of clock/latency compensation in the receiver. So we find
one global lag from a few probe windows and then track the lag block by block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

SILENT_MS = 1e-9  # mean square below which a block is considered digital silence
# A block mostly made of padding silence correlates by chance, so it may not (re)acquire the lag.
MIN_SOLID_LOCAL = 0.25
MIN_SOLID_GLOBAL = 0.75


@dataclass
class Alignment:
    ok: bool
    lags: np.ndarray  # per block, in samples of the matching rate; NaN if unknown
    ncc: np.ndarray  # per block best correlation (0 where not matched)
    block: int  # block length in samples of the matching rate
    jumps: int  # number of unexplained large lag changes


def _ncc(ref_pad: np.ndarray, ref_c2: np.ndarray, seg: np.ndarray, start: int, stop: int):
    """Normalised cross-correlation of seg against ref_pad for start positions [start, stop)."""
    n = len(seg)
    win = ref_pad[start : stop + n - 1]
    cc = fftconvolve(win, seg[::-1], mode="valid")
    en = ref_c2[start + n : stop + n] - ref_c2[start : stop]
    den = np.sqrt(np.maximum(en, 0.0) * float(seg @ seg))
    return np.where(den > 1e-12, cc / np.maximum(den, 1e-12), 0.0)


class _Ref:
    def __init__(self, ref: np.ndarray, pad: int):
        self.pad = pad
        self.x = np.concatenate([np.zeros(pad, ref.dtype), ref, np.zeros(pad, ref.dtype)])
        self.c2 = np.concatenate([[0.0], np.cumsum(self.x.astype(np.float64) ** 2)])

    def search(self, seg: np.ndarray, lo: int, hi: int):
        """Best (ref position, ncc) for the start of `seg`, searched in [lo, hi] (unpadded ref coordinates)."""
        lo = max(lo, -self.pad)
        hi = min(hi, len(self.x) - self.pad - len(seg))
        if hi < lo:
            return 0, 0.0
        scores = _ncc(self.x, self.c2, seg, lo + self.pad, hi + self.pad + 1)
        k = int(np.argmax(scores))
        return lo + k, float(scores[k])


def align(
    ref: np.ndarray,
    dump: np.ndarray,
    rate: int,
    block_s: float = 0.5,
    local_s: float = 0.1,
    accept: float = 0.5,
    probes: int = 16,
    probe_s: float = 1.0,
) -> Alignment:
    """`ref` and `dump` are mono at the matching rate `rate`."""
    block = max(int(block_s * rate), 16)
    n_blocks = max(len(dump) // block, 1)
    lags = np.full(n_blocks, np.nan)
    ncc = np.zeros(n_blocks)

    def block_seg(b):
        return dump[b * block : (b + 1) * block]

    energy = np.array([float(np.mean(block_seg(b).astype(np.float64) ** 2)) for b in range(n_blocks)])
    active = np.where(energy > SILENT_MS)[0]
    if len(active) == 0 or len(ref) < block:
        return Alignment(False, lags, ncc, block, 0)

    rf = _Ref(ref, pad=int(rate * 2) + len(dump) // 4 + block * 4)

    # Global acquisition from loud blocks spread over the dump.
    loud = active[energy[active] >= np.median(energy[active])]
    picks = loud[np.linspace(0, len(loud) - 1, min(probes, len(loud))).astype(int)]
    plen = int(probe_s * rate)
    cands = []
    for b in picks:
        seg = dump[b * block : b * block + plen]
        if len(seg) < plen // 2 or float(seg @ seg) <= 0:
            continue
        pos, score = rf.search(seg, -len(ref), len(ref))
        if score >= accept:
            cands.append((pos - b * block, score))
    if not cands:
        return Alignment(False, lags, ncc, block, 0)
    tol = int(0.05 * rate)
    cl = np.array([c[0] for c in cands])
    support = [(np.abs(cl - v) <= tol).sum() for v in cl]
    anchor = float(np.median(cl[np.abs(cl - cl[int(np.argmax(support))]) <= tol]))

    cur = int(round(anchor))
    loc = int(local_s * rate)
    jumps = 0
    for b in active:
        seg = block_seg(b)
        solid = float(np.mean(np.abs(seg) > 1e-5))  # share of non-silent samples
        if solid < MIN_SOLID_LOCAL:
            continue
        start = b * block
        pos, score = rf.search(seg, start + cur - loc, start + cur + loc)
        if score < accept and solid >= MIN_SOLID_GLOBAL:
            gpos, gscore = rf.search(seg, -len(ref), len(ref))
            if gscore >= max(accept, 0.7):
                pos, score = gpos, gscore
                if abs(pos - start - cur) > tol:
                    jumps += 1
        if score >= accept:
            cur = pos - start
            lags[b] = cur
            ncc[b] = score

    return Alignment(True, lags, ncc, block, jumps)


def fill_lags(lags: np.ndarray) -> np.ndarray:
    """Replace unknown lags by interpolation / nearest known value."""
    known = ~np.isnan(lags)
    idx = np.arange(len(lags))
    return np.interp(idx, idx[known], lags[known])
