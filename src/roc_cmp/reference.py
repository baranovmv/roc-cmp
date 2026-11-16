"""The reference WAV shipped with the package."""

import os

REFERENCE_NAME = "test_shot_sonata_no_14.wav"


def reference_path() -> str:
    """Path of the bundled reference file; stream it and compare the received dump against it."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", REFERENCE_NAME)
