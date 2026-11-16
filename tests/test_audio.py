import os
import struct
import tempfile
import wave

import numpy as np
import pytest
from scipy.io import wavfile

from roc_cmp.audio import _read_pcm24, load_wav


def chunk(chunk_id: bytes, payload: bytes) -> bytes:
    return chunk_id + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) & 1 else b"")


def pcm24_wav(samples: np.ndarray, rate: int = 44100, tag: int = 1, before_data: bytes = b"") -> bytes:
    """RIFF/WAVE bytes for (frames, channels) int32 samples holding 24-bit values."""
    channels = samples.shape[1]
    raw = samples.astype("<i4").reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * 3 * channels, 3 * channels, 24)
    if tag == 0xFFFE:  # cbSize, valid bits, channel mask, sub-format GUID (PCM)
        fmt += struct.pack("<HHI", 22, 24, 3) + bytes.fromhex("0100000000001000800000aa00389b71")
    body = b"WAVE" + chunk(b"fmt ", fmt) + before_data + chunk(b"data", raw)
    return b"RIFF" + struct.pack("<I", len(body)) + body


@pytest.fixture()
def workdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def write(workdir, name, data: bytes) -> str:
    path = os.path.join(workdir, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def random_24bit(frames: int, channels: int) -> np.ndarray:
    return np.random.RandomState(7).randint(-(2 ** 23), 2 ** 23, size=(frames, channels)).astype(np.int32)


@pytest.mark.parametrize("channels", [1, 2, 3])
@pytest.mark.parametrize("tag", [1, 0xFFFE])
def test_pcm24_reader(workdir, channels, tag):
    x = random_24bit(1000, channels)
    # an odd-sized chunk (padded) before the data must be skipped
    path = write(workdir, "a.wav", pcm24_wav(x, 48000, tag, chunk(b"LIST", b"abc")))
    rate, got = _read_pcm24(path)
    assert rate == 48000
    assert got.dtype == np.int32
    assert got.shape == ((1000,) if channels == 1 else (1000, channels))
    assert np.array_equal(got.reshape(1000, channels), x << 8)  # left-justified, like scipy >= 1.6


def test_pcm24_reader_matches_scipy_when_scipy_supports_it(workdir):
    path = write(workdir, "a.wav", pcm24_wav(random_24bit(500, 2)))
    try:
        rate, expected = wavfile.read(path)
    except ValueError:
        pytest.skip("this scipy cannot read 24-bit WAV, which is what the fallback is for")
    got_rate, got = _read_pcm24(path)
    assert got_rate == rate and got.dtype == expected.dtype and np.array_equal(got, expected)


def test_pcm24_reader_tolerates_truncated_data(workdir):
    data = pcm24_wav(random_24bit(100, 2))
    path = write(workdir, "a.wav", data[:-7])  # cut in the middle of the last frames
    assert _read_pcm24(path)[1].shape == (98, 2)


def test_load_wav_converts_24bit_to_float(workdir):
    x = random_24bit(300, 2)
    audio = load_wav(write(workdir, "a.wav", pcm24_wav(x)))
    assert audio.rate == 44100 and audio.samples.shape == (300, 2)
    assert np.allclose(audio.samples, x / 2.0 ** 23, atol=1e-7)


def test_pcm24_reader_ignores_other_formats(workdir):
    path = os.path.join(workdir, "pcm16.wav")
    w = wave.open(path, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(8000)
    w.writeframes(b"\0\1" * 10)
    w.close()
    assert _read_pcm24(path) is None
    assert load_wav(path).samples.shape == (10, 1)

    path = write(workdir, "junk.wav", b"not a wav file at all")
    assert _read_pcm24(path) is None
    with pytest.raises(ValueError):
        load_wav(path)
