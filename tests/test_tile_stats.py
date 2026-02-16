import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from TileStats import tile_bins


def test_tile_bins_counts():
    data = [
        (0.1, 0.1),
        (0.9, 0.9),
        (1.2, 0.1),
    ]

    bins = tile_bins(data, 1, polygon_keys=False)

    assert bins[(0.5, 0.5)] == 2
    assert bins[(1.5, 0.5)] == 1
    assert sum(bins.values()) == len(data)
