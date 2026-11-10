"""Frame-wise spectral comparison of an aligned dump against the reference and scoring."""

from collections import OrderedDict, namedtuple
from typing import List, Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided

from .align import align, fill_lags
from .audio import Audio, match_channels, resample

ANALYSIS_RATE = 16000
MATCH_DECIM = 4  # alignment runs at ANALYSIS_RATE / MATCH_DECIM
N_FFT = 512
HOP = 256
N_BANDS = 24
FLOOR_DB = -80.0  # band power floor, dBFS; anything quieter is treated as silence
SILENCE_DB = -80.0  # samples below this level are "silent" in the dump
MIN_DROPOUT_S = 0.0015  # shortest digital-silence run counted as a dropout
JUMP_TOL_S = 0.003  # lag change between blocks above this is a timeline jump, not drift
REF_AUDIBLE = 10 ** (-60 / 20)  # the reference must be louder than this for a gap to count

# start, end: dump time, seconds
# ref_start: corresponding position in the reference
# kind: "dropout" or "distortion"
# lsd: peak log-spectral distance inside, dB
Glitch = namedtuple("Glitch", "start end ref_start kind lsd")


class Result:
    # ref_offset: reference time that dump time 0 corresponds to
    # coverage: fraction of the reference present and matching in the dump
    _FIELDS = (
        "score", "detected", "reason", "ref_duration", "dump_duration", "active_start",
        "active_end", "ref_offset", "coverage", "bad_fraction", "glitch_count", "glitches",
        "median_lsd", "drift_ms", "jumps",
    )

    def __init__(
        self,
        score,
        detected,
        reason="",
        ref_duration=0.0,
        dump_duration=0.0,
        active_start=0.0,
        active_end=0.0,
        ref_offset=0.0,
        coverage=0.0,
        bad_fraction=0.0,
        glitch_count=0,
        glitches=None,
        median_lsd=0.0,
        drift_ms=0.0,
        jumps=0,
    ):
        self.score = score
        self.detected = detected
        self.reason = reason
        self.ref_duration = ref_duration
        self.dump_duration = dump_duration
        self.active_start = active_start
        self.active_end = active_end
        self.ref_offset = ref_offset
        self.coverage = coverage
        self.bad_fraction = bad_fraction
        self.glitch_count = glitch_count
        self.glitches = [] if glitches is None else glitches
        self.median_lsd = median_lsd
        self.drift_ms = drift_ms
        self.jumps = jumps

    def to_dict(self) -> dict:
        d = OrderedDict((name, getattr(self, name)) for name in self._FIELDS)
        d["glitches"] = [g._asdict() for g in self.glitches]
        return d


def _band_matrix() -> np.ndarray:
    """(N_BANDS, N_FFT//2+1) averaging matrix over log-spaced bands, 100 Hz to Nyquist."""
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / ANALYSIS_RATE)
    edges = np.geomspace(100.0, ANALYSIS_RATE / 2, N_BANDS + 1)
    m = np.zeros((N_BANDS, len(freqs)))
    for i in range(N_BANDS):
        sel = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if not sel.any():  # narrow low band: take nearest bin
            sel[np.argmin(np.abs(freqs - np.sqrt(edges[i] * edges[i + 1])))] = True
        m[i, sel] = 1.0 / sel.sum()
    return m


_BANDS = _band_matrix()
_WINDOW = np.hanning(N_FFT).astype(np.float32)


def _frames(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    """(n_frames, n) read-only view of overlapping frames of 1-D `x`, `hop` samples apart."""
    n_frames = (len(x) - n) // hop + 1
    return as_strided(
        x, shape=(n_frames, n), strides=(hop * x.strides[0], x.strides[0]), writeable=False
    )


def _frames_db(frames: np.ndarray) -> np.ndarray:
    """(frames, N_BANDS) band power in dBFS for a (frames, N_FFT) array."""
    spec = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)) ** 2
    spec /= (_WINDOW.sum() / 2) ** 2  # a full-scale sine -> power 1 (0 dBFS)
    return 10 * np.log10(np.maximum(spec @ _BANDS.T, 10 ** (FLOOR_DB / 10)))


def _make_lag_fn(lags: np.ndarray, block: int, jump_tol: int):
    """lag(pos): reference-minus-dump offset at dump sample `pos`.

    Lags are interpolated between block centres to follow slow clock drift, but a
    change larger than `jump_tol` samples is a timeline discontinuity and stays a step.
    """
    delta = np.concatenate([[0.0], np.diff(lags)])
    steps = np.where(np.abs(delta) > jump_tol, delta, 0.0)
    offset = np.cumsum(steps)
    smooth = lags - offset
    centres = (np.arange(len(lags)) + 0.5) * block

    def lag(pos: np.ndarray) -> np.ndarray:
        b = np.clip(np.asarray(pos) // block, 0, len(lags) - 1).astype(int)
        return np.round(np.interp(pos, centres, smooth) + offset[b]).astype(int)

    return lag


def _ref_frames_db(ref: np.ndarray, starts: np.ndarray, lags: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Band dB of reference frames cut at starts + lags (zeros outside the reference)."""
    out = np.empty((len(starts), N_BANDS))
    k = np.arange(N_FFT)
    for i in range(0, len(starts), chunk):
        idx = (starts[i : i + chunk] + lags[i : i + chunk])[:, None] + k
        ok = (idx >= 0) & (idx < len(ref))
        out[i : i + chunk] = _frames_db(np.where(ok, ref[np.clip(idx, 0, len(ref) - 1)], 0.0))
    return out


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def compare(
    ref: Audio,
    dump: Audio,
    lsd_threshold: float = 2.5,
    margin_s: float = 0.1,
) -> Result:
    res = Result(score=0.0, detected=False, ref_duration=ref.duration, dump_duration=dump.duration)

    rx = resample(ref.samples, ref.rate, ANALYSIS_RATE)
    dx = resample(dump.samples, dump.rate, ANALYSIS_RATE)
    rx, dx = match_channels(rx, dx)
    n = len(dx)
    if n < N_FFT:
        res.reason = "dump is empty"
        return res

    hop_s = HOP / ANALYSIS_RATE

    # Active region: first to last non-silent sample of the dump (original rate).
    # Leading/trailing silence (late join, partial send) is outside and not scored.
    peak = np.abs(dump.samples).max(axis=1)
    live = np.flatnonzero(peak > 10 ** (SILENCE_DB / 20))
    if len(live) == 0:
        res.reason = "no transmission detected: dump is silent"
        return res
    to16 = ANALYSIS_RATE / dump.rate
    a0, a1 = int(live[0] * to16), int((live[-1] + 1) * to16)
    f_lo = -(-a0 // HOP)  # first frame fully inside the active region
    f_hi = (a1 - N_FFT) // HOP + 1
    if f_hi <= f_lo:
        res.reason = "no transmission detected: dump is too short"
        return res

    mono = dx.mean(axis=1)
    al = align(
        rx.mean(axis=1)[::MATCH_DECIM],
        mono[::MATCH_DECIM],
        ANALYSIS_RATE // MATCH_DECIM,
    )
    if not al.ok or not np.isfinite(al.lags).any():
        res.reason = "no transmission detected: dump does not match the reference"
        return res
    lags = fill_lags(al.lags) * MATCH_DECIM
    block = al.block * MATCH_DECIM
    res.jumps = al.jumps

    n_frames = (n - N_FFT) // HOP + 1
    starts = np.arange(n_frames) * HOP
    lag_fn = _make_lag_fn(lags, block, int(JUMP_TOL_S * ANALYSIS_RATE))
    frame_lags = lag_fn(starts + N_FFT // 2)
    lsd_ch = []
    for c in range(dx.shape[1]):
        d_db = _frames_db(_frames(dx[:, c], N_FFT, HOP))
        r_db = _ref_frames_db(rx[:, c], starts, frame_lags)
        lsd_ch.append(np.mean(np.abs(d_db - r_db), axis=1))
    lsd = np.max(lsd_ch, axis=0)

    f_hi = min(f_hi, n_frames)
    bad = np.zeros(n_frames, dtype=bool)
    bad[f_lo:f_hi] = lsd[f_lo:f_hi] > lsd_threshold

    # Short digital-silence runs where the reference is audible: packet loss.
    # Too brief to move a 32 ms spectral frame much, but clearly audible as clicks.
    dropout = np.zeros(n_frames, dtype=bool)
    ref_mono_abs = np.abs(rx).max(axis=1)
    for s, e in _zero_runs(dump.samples, int(MIN_DROPOUT_S * dump.rate)):
        s16, e16 = int(s * to16), int(e * to16)
        if s16 < a0 or e16 > a1:
            continue
        lag = int(lag_fn(np.array([(s16 + e16) // 2]))[0])
        seg = ref_mono_abs[max(s16 + lag, 0) : max(e16 + lag, 0)]
        if seg.max(initial=0.0) < REF_AUDIBLE:
            continue
        dropout[max(s16 // HOP - 1, 0) : min(e16 // HOP + 1, n_frames)] = True
    dropout[:f_lo] = dropout[f_hi:] = False
    bad |= dropout

    # Each glitch ruins more listening time than its literal length: dilate by margin.
    n_act = f_hi - f_lo
    m = int(round(margin_s / hop_s))
    bad_d = np.zeros(n_act, dtype=bool)
    events = _runs(bad[f_lo:f_hi])
    for s, e in events:
        bad_d[max(s - m, 0) : min(e + m, n_act)] = True

    for s, e in events:
        a, b = f_lo + s, f_lo + e
        res.glitches.append(
            Glitch(
                start=a * hop_s,
                end=b * hop_s,
                ref_start=(a * HOP + frame_lags[a]) / ANALYSIS_RATE,
                kind="dropout" if dropout[a:b].any() else "distortion",
                lsd=float(lsd[a:b].max()),
            )
        )

    res.detected = True
    res.bad_fraction = float(bad_d.mean())
    res.score = 100.0 * (1.0 - res.bad_fraction)
    res.glitch_count = len(res.glitches)
    res.active_start, res.active_end = a0 / ANALYSIS_RATE, a1 / ANALYSIS_RATE
    res.ref_offset = float(lags[0]) / ANALYSIS_RATE
    res.median_lsd = float(np.median(lsd[f_lo:f_hi]))
    res.drift_ms = float((lags.max() - lags.min()) / ANALYSIS_RATE * 1000)
    good_dur = float((~bad_d).sum() * hop_s)
    res.coverage = min(good_dur / ref.duration, 1.0) if ref.duration else 0.0
    return res


def _zero_runs(x: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """Runs of digital silence (all channels) at least `min_len` samples long."""
    z = np.abs(x).max(axis=1) < 1e-5
    return [(s, e) for s, e in _runs(z) if e - s >= max(min_len, 1)]
