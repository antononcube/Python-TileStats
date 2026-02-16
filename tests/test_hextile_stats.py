import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from TileStats import hextile_bins


def test_hextile_bins_counts():
    data = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, math.sqrt(3) / 2),
        (0.5, math.sqrt(3) / 2),
    ]

    bins = hextile_bins(data, 1, polygon_keys=False)

    assert bins[(0.0, 0.0)] == 1
    assert bins[(1.0, 0.0)] == 1
    assert bins[(0.5, math.sqrt(3) / 2)] == 2
    assert sum(bins.values()) == len(data)
