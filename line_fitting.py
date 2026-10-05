"""
Straight-line layout of a point cloud, used by the mesh linear operator.

The line runs between the two points that lie farthest apart, and that pair is found exactly. A
first guess walks to the farthest point twice. A longer pair can only join points whose distances
from the middle of that guess add up to more than its length, so a selection that is roughly a
line leaves just the few points near its ends to compare with each other; only a cloud hugging a
sphere around that middle falls back to comparing every pair.
"""

import numpy

PAIR_BLOCK_ELEMENTS = 1 << 16


def squared_distances(points, origin):
    offsets = points - origin
    return numpy.einsum("ij,ij->i", offsets, offsets)


def farthest_pair(points):
    """Indices of the two points farthest apart."""
    first = int(squared_distances(points, points[0]).argmax())
    second = int(squared_distances(points, points[first]).argmax())
    best_pair = (first, second)
    offset = points[second] - points[first]
    best_distance = float(offset @ offset)

    middle = (points[first] + points[second]) * 0.5
    reach = numpy.sqrt(squared_distances(points, middle))
    candidates = numpy.flatnonzero(reach >= numpy.sqrt(best_distance) - reach.max())
    if len(candidates) < 2:
        return best_pair

    subset = points[candidates] - middle
    lengths = numpy.einsum("ij,ij->i", subset, subset)
    count = len(candidates)
    start = 0
    while start < count:
        rows = max(1, PAIR_BLOCK_ELEMENTS // (count - start))
        distances = subset[start:start + rows] @ subset[start:].T
        distances *= -2.0
        distances += lengths[start:]
        distances += lengths[start:start + rows, None]
        row, column = numpy.unravel_index(distances.argmax(), distances.shape)
        if distances[row, column] > best_distance:
            best_distance = float(distances[row, column])
            best_pair = (int(candidates[start + row]), int(candidates[start + column]))
        start += rows
    return best_pair


def linear_calculate_verts(locations):
    """
    New locations for the points between the two farthest apart, as (index, location) pairs.

    Those two stay where they are. Every other point drops straight onto the line through them and
    keeps its own place along it, which always lies between the two.
    """
    points = numpy.asarray(locations, dtype=numpy.float64)
    start, end = farthest_pair(points)
    line = points[end] - points[start]
    fractions = (points - points[start]) @ line / (line @ line)
    moves = []
    for index, fraction in enumerate(fractions):
        if index != start and index != end:
            moves.append((index, points[start] + line * fraction))
    return moves
