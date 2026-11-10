"""Audio loading and normalisation."""

import os
import struct
from math import gcd
from typing import Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

_WAVE_FORMAT_PCM = 1
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


class Audio:
    """Float samples in [-1, 1], shape (n_samples, n_channels), and the sample rate."""

    def __init__(self, samples: np.ndarray, rate: int):
        self.samples = samples
        self.rate = rate

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


def _read_pcm24(path):
    """(rate, int32 samples left-justified) of a 24-bit integer PCM WAV, None for other files.

    scipy reads 24-bit WAVs only since 1.6 (Python 3.7+), but this is what roc-recv writes.
    """
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return None
        fmt = None
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                return None
            chunk_id, size = struct.unpack("<4sI", chunk)
            if chunk_id == b"fmt " and size >= 16:
                fmt = struct.unpack("<HHIIHH", f.read(16))
                f.seek(size - 16 + (size & 1), os.SEEK_CUR)
            elif chunk_id == b"data":
                if fmt is None:
                    return None
                size = min(size, os.fstat(f.fileno()).st_size - f.tell())  # streamed files
                raw = f.read(size)
                break
            else:
                f.seek(size + (size & 1), os.SEEK_CUR)

    tag, channels, rate, _, _, bits = fmt
    if tag not in (_WAVE_FORMAT_PCM, _WAVE_FORMAT_EXTENSIBLE) or bits != 24 or channels < 1:
        return None
    n = len(raw) // (3 * channels) * channels
    wide = np.zeros((n, 4), np.uint8)
    wide[:, 1:] = np.frombuffer(raw, np.uint8, n * 3).reshape(n, 3)
    x = wide.view("<i4").reshape(-1, channels)
    return int(rate), (x[:, 0] if channels == 1 else x)


def _read_wav(path: str):
    try:
        return wavfile.read(path)
    except ValueError:
        pcm24 = _read_pcm24(path)
        if pcm24 is None:
            raise
        return pcm24


def load_wav(path) -> Audio:
    rate, x = _read_wav(str(path))
    if x.ndim == 1:
        x = x[:, None]
    return Audio(_to_float(x), int(rate))


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples
    g = gcd(src_rate, dst_rate)
    out = resample_poly(samples, dst_rate // g, src_rate // g, axis=0)
    return out.astype(np.float32)


def match_channels(ref: np.ndarray, dump: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Make channel counts equal by downmixing both to mono if they differ."""
    if ref.shape[1] == dump.shape[1]:
        return ref, dump
    return ref.mean(axis=1, keepdims=True), dump.mean(axis=1, keepdims=True)
