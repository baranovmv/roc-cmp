"""Audio loading and normalisation."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


@dataclass
class Audio:
    """Float samples in [-1, 1], shape (n_samples, n_channels)."""

    samples: np.ndarray
    rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.rate

    @property
    def channels(self) -> int:
        return self.samples.shape[1]


def _to_float(x: np.ndarray) -> np.ndarray:
    # scipy returns 24-bit PCM left-justified in int32, so integer scaling is uniform.
    if x.dtype == np.uint8:
        return (x.astype(np.float32) - 128.0) / 128.0
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.float32) / float(-np.iinfo(x.dtype).min)
    return x.astype(np.float32)


def load_wav(path: str | Path) -> Audio:
    rate, x = wavfile.read(path)
    if x.ndim == 1:
        x = x[:, None]
    return Audio(_to_float(x), int(rate))


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples
    g = gcd(src_rate, dst_rate)
    out = resample_poly(samples, dst_rate // g, src_rate // g, axis=0)
    return out.astype(np.float32)


def match_channels(ref: np.ndarray, dump: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make channel counts equal by downmixing both to mono if they differ."""
    if ref.shape[1] == dump.shape[1]:
        return ref, dump
    return ref.mean(axis=1, keepdims=True), dump.mean(axis=1, keepdims=True)
