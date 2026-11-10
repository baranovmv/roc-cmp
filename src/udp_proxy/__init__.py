"""UDP proxy with a per-stream network simulator (loss, delay, jitter without reordering).

Modelled on GStreamer's netsim element, without bandwidth limiting.
"""

from .cli import main

__all__ = ["main"]
