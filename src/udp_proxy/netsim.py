"""Network simulator: packet loss, delay and jitter without reordering (cf. GStreamer netsim)."""

import random
from typing import Optional

from .config import Stream

# Same trick as gstnetsim: keep order by making a packet ready right after the previous one.
_ORDER_EPSILON = 1e-6


class NetSim:
    """One direction of one stream: drops packets and schedules the survivors in order."""

    def __init__(self, stream: Stream, rng: random.Random):
        self._loss = stream.loss / 100
        self._delay = stream.delay_ms / 1000
        self._jitter = stream.jitter_ms / 1000
        self._rng = rng
        self._last_ready = float("-inf")

    def schedule(self, now: float) -> Optional[float]:
        """Return the time at which a packet arriving at `now` must be sent, or None to drop it."""
        if self._loss > 0 and self._rng.random() < self._loss:
            return None
        delay = self._delay
        if self._jitter:
            delay += self._rng.uniform(-self._jitter, self._jitter)
        ready = now + max(delay, 0.0)
        if ready <= self._last_ready:
            ready = self._last_ready + _ORDER_EPSILON
        self._last_ready = ready
        return ready
