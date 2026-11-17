"""The herd's lore: which strides along a trail, and how much of the trail, have led to better ground."""

import numpy as np


class Lore:
    """What the herd remembers about walking its trails: `slots` pairs of a stride and a share of the trail kept.

    A trail follower draws a slot at random, a stride from a Cauchy distribution around the slot's stride (scale 0.1,
    redrawn until positive, at most 1) and a share from a normal distribution around the slot's share (deviation 0.1,
    clipped to [0, 1]). At the end of a season the walks that reached better ground teach one slot, in turn: its
    stride and its share become their gain-weighted Lehmer means, Σ w s² / Σ w s, so the long strides and the large
    shares that paid count more than the short and small ones. This is the success-history adaptation of L-SHADE
    (Tanabe and Fukunaga 2014).
    """

    def __init__(self, slots: int, stride: float, keep: float):
        self.strides = np.full(max(1, int(slots)), float(stride))
        self.keeps = np.full(max(1, int(slots)), float(keep))
        self.turn = 0
        self._won: list[tuple[float, float, float]] = []

    def draw(self, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """`count` strides in (0, 1] and shares kept in [0, 1], each around a slot drawn at random."""
        slot = rng.integers(len(self.strides), size=count)
        stride = self.strides[slot] + 0.1 * rng.standard_cauchy(count)
        while np.any(low := stride <= 0.0):
            stride[low] = self.strides[slot[low]] + 0.1 * rng.standard_cauchy(int(low.sum()))
        keep = self.keeps[slot] + 0.1 * rng.standard_normal(count)
        return np.minimum(1.0, stride), np.clip(keep, 0.0, 1.0)

    def note(self, stride: np.ndarray, keep: np.ndarray, gain: np.ndarray) -> None:
        """Walks with these strides and shares reached ground better by `gain` each."""
        for s, k, g in zip(np.ravel(stride), np.ravel(keep), np.ravel(gain), strict=True):
            if np.isfinite(g) and g > 0:
                self._won.append((float(s), float(k), float(g)))

    def learn(self) -> None:
        """At a season's end: the walks that paid teach the next slot."""
        if not self._won:
            return
        strides, keeps, gains = np.array(self._won).T
        self._won = []
        w = gains / gains.sum()
        self.strides[self.turn] = (w @ strides**2) / (w @ strides)
        kept = w @ keeps
        self.keeps[self.turn] = (w @ keeps**2) / kept if kept > 0 else 0.0
        self.turn = (self.turn + 1) % len(self.strides)
