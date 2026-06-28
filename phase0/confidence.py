"""confidence.py — DSpark-inspired confidence-scheduled verification heuristics.

The spec-decode loop wastes WAN traversals when the draft diverges early. Instead of
always running at fixed depth/K, adapt based on a cheap confidence signal derived from
the RECENT acceptance rate (the EMA of how many draft tokens the target accepted per round).

Two modes:
  * DEPTH throttling (pipelined path, coordinate_pipe): when acceptance drops, reduce
    the number of in-flight verify chunks so fewer stale chunks are wasted on a bad draft
    streak. When acceptance recovers, open the pipeline back up. K stays fixed (the CUDA
    graph shape is K+1), so this is safe for the FastVerify path.
  * K adaptation (non-pipelined path, generate_spec / coordinate): when acceptance drops,
    reduce K (draft fewer tokens). The non-pipelined path re-sends each chunk, so the
    stage doesn't cache a fixed-shape graph — K can vary per round.

The confidence signal is free: it's computed from the accept/verify results the loop
already produces, so it needs no logits, no model change, and no draft-protocol change.

Self-test: python confidence.py
"""
import math


class ConfidenceScheduler:
    """Tracks a running acceptance EMA and maps it to a scheduling decision (depth or K).

    The mapping is a piecewise-linear clamp:
      acceptance >= hi  -> max_val (confident: open up)
      acceptance <= lo  -> min_val (unconfident: throttle)
      in between        -> lerp

    min/max are inclusive bounds so a caller never gets an insane value. The EMA uses
    a 0.7/0.3 smoothing (same as the existing adaptive K in generate_spec), so a single
    bad round doesn't panic the scheduler but a sustained drop throttles within ~3 rounds.
    """

    def __init__(self, min_val, max_val, lo=0.3, hi=0.7, ema_alpha=0.3):
        self.min_val = min_val
        self.max_val = max_val
        self.lo = lo
        self.hi = hi
        self.ema_alpha = ema_alpha
        self._ema = None  # None until first observation

    def observe(self, accepted, total):
        """Record one round's acceptance (n accepted out of K proposed)."""
        rate = accepted / max(total, 1)
        if self._ema is None:
            self._ema = rate
        else:
            self._ema = (1 - self.ema_alpha) * self._ema + self.ema_alpha * rate
        return self._ema

    def value(self):
        """Current scheduling value (depth or K), clamped to [min_val, max_val]."""
        if self._ema is None:
            return self.max_val  # optimistic start: open up until we have data
        e: float = self._ema
        if e >= self.hi:
            return self.max_val
        if e <= self.lo:
            return self.min_val
        t = (e - self.lo) / (self.hi - self.lo)
        return int(round(self.min_val + t * (self.max_val - self.min_val)))

    def confidence(self):
        """Raw EMA for logging / instrumentation."""
        return self._ema


def _selftest():
    # depth scheduler: depth 1..4, throttle when acceptance < 0.3
    ds = ConfidenceScheduler(min_val=1, max_val=4, lo=0.3, hi=0.7)
    assert ds.value() == 4, "should start at max (no data)"
    assert ds.confidence() is None

    # high acceptance -> stays at max depth
    for _ in range(5):
        ds.observe(8, 8)  # perfect acceptance (K=8)
    assert ds.value() == 4, f"high acceptance should be max depth, got {ds.value()}"
    assert abs(ds.confidence() - 1.0) < 0.01

    # sustained low acceptance -> throttles to min depth
    for _ in range(8):
        ds.observe(1, 8)  # 0.125 acceptance
    assert ds.value() == 1, f"sustained low acceptance should throttle to min, got {ds.value()}"

    # recovery -> opens back up
    for _ in range(8):
        ds.observe(7, 8)  # 0.875
    assert ds.value() == 4, f"recovery should reopen, got {ds.value()}"

    # K scheduler: K 1..16
    ks = ConfidenceScheduler(min_val=1, max_val=16, lo=0.3, hi=0.7)
    assert ks.value() == 16
    ks.observe(4, 8)
    v1 = ks.value()
    ks.observe(4, 8)
    v2 = ks.value()
    assert v1 >= v2, "declining acceptance should not increase K"

    print("[confidence] PASS — ConfidenceScheduler depth/K adaptation works")


if __name__ == "__main__":
    _selftest()
