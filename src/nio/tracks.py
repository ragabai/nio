"""The tracks: the places ibex have left behind, which mark the herd's trails."""

import numpy as np


class Tracks:
    """An archive of at most `size` tracks: positions ibex stood on before they moved to better ground.

    In the model every ibex that moves leaves a track where it stood. A track is worse ground than where its ibex went,
    so the offset from a track to a herd-mate points the way the herd has been moving; `follow_trail` strides along
    such offsets. Once the archive is full a new track wears over a random old one, so the tracks follow the herd as
    it moves. `resize` keeps a random subset when the herd thins.
    """

    def __init__(self, size: int, dimensions: int):
        self.size = max(0, int(size))
        self.positions = np.empty((0, dimensions))

    def __len__(self) -> int:
        return len(self.positions)

    def leave(self, x: np.ndarray, rng: np.random.Generator) -> None:
        """An ibex has left `x`: keep it as a track (over a random old one once the archive is full)."""
        if self.size == 0:
            return
        if len(self.positions) < self.size:
            self.positions = np.concatenate((self.positions, x[None, :]))
        else:
            self.positions[rng.integers(self.size)] = x

    def resize(self, size: int, rng: np.random.Generator) -> None:
        """The archive now holds at most `size` tracks; a random subset of the old ones stays."""
        self.size = max(0, int(size))
        if len(self.positions) > self.size:
            self.positions = self.positions[rng.permutation(len(self.positions))[: self.size]]
