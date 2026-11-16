import os

import numpy as np
import pytest
from scipy.io import wavfile

import roc_cmp
from roc_cmp.cli import _expect_range, main


@pytest.fixture()
def wavs(tmp_path):
    """Reference of a few seconds of noise bursts, and a dump that equals it or is silent."""
    rate = 16000
    x = (np.random.RandomState(1).randn(rate * 4, 2) * 3000).astype(np.int16)
    ref, good, silent = (str(tmp_path / n) for n in ("ref.wav", "good.wav", "silent.wav"))
    wavfile.write(ref, rate, x)
    wavfile.write(good, rate, x)
    wavfile.write(silent, rate, np.zeros_like(x))
    return ref, good, silent


def test_expect_range_parsing():
    assert _expect_range("97:") == (97.0, None)
    assert _expect_range(":50") == (None, 50.0)
    assert _expect_range("10:50.5") == (10.0, 50.5)
    for bad in ("", ":", "50", "a:b", "60:50"):
        with pytest.raises(Exception):
            _expect_range(bad)


def test_exit_status_for_score_bounds(wavs):
    ref, good, silent = wavs
    assert main([ref, good, "--min-score", "97"]) == 0
    assert main([ref, good, "--max-score", "50"]) == 1
    assert main([ref, silent, "--max-score", "50"]) == 0
    assert main([ref, silent, "--min-score", "50"]) == 1
    assert main([ref, good, "--expect", "97:"]) == 0
    assert main([ref, silent, "--expect", ":50"]) == 0
    assert main([ref, good, "--expect", ":50"]) == 1
    assert main([ref, good, "--expect", "10:50"]) == 1
    assert main([ref, good, "--expect", "10:", "--min-score", "99"]) == 0  # stricter bound wins


def test_score_api(wavs):
    ref, good, silent = wavs
    assert roc_cmp.score(ref, good).score > 97
    assert roc_cmp.score(ref, silent).score < 50
    with pytest.raises(OSError):
        roc_cmp.score(ref, ref + ".missing")


def test_reference_path(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--reference-path"])
    assert e.value.code == 0
    path = capsys.readouterr().out.strip()
    assert path == roc_cmp.reference_path()
    assert os.path.isfile(path)
    assert roc_cmp.score(path, path).score > 99
