from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from roc_cmp.audio import Audio, load_wav, resample
from roc_cmp.compare import compare

DATA = Path(__file__).parent.parent / "data"
pytestmark = pytest.mark.skipif(not (DATA / "loituma44.wav").exists(), reason="sample data not available")


@pytest.fixture(scope="module")
def ref() -> Audio:
    return load_wav(DATA / "loituma44.wav")


def excerpt(ref: Audio, a: float, b: float) -> np.ndarray:
    return ref.samples[int(a * ref.rate) : int(b * ref.rate)].copy()


def silence(ref: Audio, seconds: float) -> np.ndarray:
    return np.zeros((int(seconds * ref.rate), 2), np.float32)


def score(ref: Audio, samples: np.ndarray, rate: Optional[int] = None):
    return compare(ref, Audio(samples, rate or ref.rate))


def test_real_dumps_ordered_by_loss_rate(ref):
    s = {n: compare(ref, load_wav(DATA / (n + ".wav"))).score for n in ("good", "bad_15", "bad_25")}
    assert s["good"] > 99.5
    assert s["good"] > s["bad_15"] > s["bad_25"]
    assert s["bad_25"] > 50  # transmission clearly happened


def test_partial_and_padded_transmission_is_not_penalised(ref):
    part = excerpt(ref, 30, 60)
    dump = np.concatenate([silence(ref, 4), part, silence(ref, 6)])
    r = score(ref, dump)
    assert r.detected and r.score > 99.5 and r.glitch_count == 0


def test_different_sample_rate_dump(ref):
    r = score(ref, resample(excerpt(ref, 30, 50), ref.rate, 48000), 48000)
    assert r.score > 99.5


def test_short_gaps_are_detected_and_scale_with_count(ref):
    def with_gaps(step: float) -> float:
        d = excerpt(ref, 30, 60)
        for t in np.arange(1, 29, step):
            d[int(t * ref.rate) : int((t + 0.005) * ref.rate)] = 0
        return score(ref, d).score

    few, many = with_gaps(4.0), with_gaps(1.0)
    assert 100 > few > many > 50


def test_skipped_audio_is_a_glitch(ref):
    d = excerpt(ref, 30, 60)
    d = np.concatenate([d[: 10 * ref.rate], d[int(10.05 * ref.rate) :]])
    r = score(ref, d)
    assert r.glitch_count == 1 and r.score < 100


def test_noise_distortion_is_heavily_penalised(ref):
    d = excerpt(ref, 30, 50)
    d += 0.03 * np.random.RandomState(0).standard_normal(d.shape).astype(np.float32)
    assert score(ref, d).score < 70


@pytest.mark.parametrize("kind", ["silence", "noise", "reversed"])
def test_no_transmission_scores_zero(ref, kind):
    d = excerpt(ref, 30, 50)
    d = {
        "silence": np.zeros_like(d),
        "noise": 0.3 * np.random.RandomState(0).standard_normal(d.shape).astype(np.float32),
        "reversed": d[::-1].copy(),
    }[kind]
    assert score(ref, d).score == 0


def test_reference_with_near_silent_lead_in_is_aligned():
    # Exact zeros followed by ~1e-13 noise (as in an exported excerpt) made zero-padded windows
    # so empty that float32 FFT noise gave "correlations" far above 1 and alignment wandered.
    rng = np.random.RandomState(1)
    rate = 44100
    t = np.arange(28 * rate) / rate
    x = sum(
        np.sin(2 * np.pi * f * t + rng.uniform(0, 6))
        * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t + rng.uniform(0, 6)) ** 2)
        for f in rng.uniform(100, 2000, 12)
    )
    x = 0.08 * x / np.abs(x).max()
    x[: int(0.2 * rate)] = 0
    x[int(0.2 * rate) : int(0.4 * rate)] = 1e-13 * rng.standard_normal(int(0.2 * rate))
    ref = x.astype(np.float32)[:, None]
    dump = np.concatenate([np.zeros((rate, 1), np.float32), ref])
    r = compare(Audio(ref, rate), Audio(dump, rate))
    assert r.detected and r.score > 99 and r.glitch_count == 0
