"""
Spline fitting and point-distribution maths shared by the curve and UV operators.

Everything here is pure maths over plain sequences of numbers, so the same code drives
3D control points, curve tilt/radius and 2D UV coordinates. Knots may be scalars or
sequences; every dimension is solved independently.
"""

import mathutils


def calculate_cubic_splines(tknots, knots):
    """Natural cubic splines through all given knots, for an arbitrary number of dimensions."""
    count = len(knots)
    if count < 2:
        return False

    if isinstance(knots[0], (float, int)):
        dimensions = 1
        locations = [[knot] for knot in knots]
    else:
        dimensions = len(knots[0])
        locations = [list(knot) for knot in knots]

    x = tknots[:]
    result = []

    for dimension in range(dimensions):
        a = []
        for location in locations:
            a.append(location[dimension])
        h = []
        for i in range(count - 1):
            value = x[i + 1] - x[i]
            if value == 0:
                h.append(1e-8)
            else:
                h.append(value)

        q = [0.0]
        for i in range(1, count - 1):
            q.append(3 / h[i] * (a[i + 1] - a[i]) - 3 / h[i - 1] * (a[i] - a[i - 1]))

        l = [1.0]
        u = [0.0]
        z = [0.0]

        for i in range(1, count - 1):
            value = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * u[i - 1]
            if value == 0:
                value = 1e-8
            l.append(value)
            u.append(h[i] / l[i])
            z.append((q[i] - h[i - 1] * z[i - 1]) / l[i])

        l.append(1.0)
        z.append(0.0)

        b = [0.0 for _ in range(count - 1)]
        c = [0.0 for _ in range(count)]
        d = [0.0 for _ in range(count - 1)]

        c[count - 1] = 0.0
        for i in range(count - 2, -1, -1):
            c[i] = z[i] - u[i] * c[i + 1]
            b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
            d[i] = (c[i + 1] - c[i]) / (3 * h[i])

        for i in range(count - 1):
            if len(result) <= i:
                result.append([])
            result[i].append([a[i], b[i], c[i], d[i], x[i]])

    splines = []
    for i in range(count - 1):
        splines.append(result[i])

    return splines


def calculate_linear_splines(tknots, knots):
    """Straight segments between consecutive knots, in the same shape as the cubic result."""
    splines = []
    if isinstance(knots[0], (float, int)):
        dimensions = 1
        as_list = lambda value: [value]
    else:
        dimensions = len(knots[0])
        as_list = lambda value: list(value)

    for i in range(len(knots) - 1):
        a = as_list(knots[i])
        b = as_list(knots[i + 1])
        delta = [b[k] - a[k] for k in range(dimensions)]
        t = tknots[i]
        u = tknots[i + 1] - t
        segment = []
        for k in range(dimensions):
            segment.append([a[k], delta[k], t, u])
        splines.append(segment)

    return splines


def evaluate_spline(interpolation, segment, distance):
    """Evaluates one spline segment at the given arc-length position, returning every dimension."""
    values = []
    if interpolation == 'cubic':
        for a, b, c, d, start in segment:
            delta = distance - start
            values.append(a + b * delta + c * (delta ** 2) + d * (delta ** 3))
    else:
        for a, delta, start, span in segment:
            if span == 0:
                span = 1e-8
            values.append(((distance - start) / span) * delta + a)
    return values


def relax_calculate_knots(points_len, circular):
    """Splits a run of points into two interleaved knot/point sets, the way LoopTools relaxes."""
    knots = [[], []]
    points = [[], []]
    run = list(range(points_len))

    if circular:
        if len(run) % 2 == 1:
            extend = [False, True, 0, 1, 0, 1]
        else:
            extend = [True, False, 0, 1, 1, 2]
    else:
        extend = [False, False, 0, 1, 1, 2]

    for j in range(2):
        working = run[:]
        if extend[j]:
            working = [run[-1]] + run + [run[0]]
        knot_indices = []
        for i in range(extend[2 + 2 * j], len(working), 2):
            knot_indices.append(working[i])
        knots[j] = knot_indices
        point_indices = []
        for i in range(extend[3 + 2 * j], len(working), 2):
            index = working[i]
            if index == run[-1] and not circular:
                continue
            if len(point_indices) == 0 or index != point_indices[0]:
                point_indices.append(index)
        points[j] = point_indices
        if circular and knots[j][0] != knots[j][-1]:
            knots[j].append(knots[j][0])

    if len(points[1]) == 0:
        knots.pop(1)
        points.pop(1)

    return knots, points


def relax_calculate_t(points_co, knots, points_indices, regular):
    """Arc-length parameters for the knot and point sets produced by relax_calculate_knots."""
    all_tknots = []
    all_tpoints = []
    for i in range(len(knots)):
        knot_list, point_list = knots[i], points_indices[i]
        mixed = []
        knot_count, point_count = len(knot_list), len(point_list)
        for j in range(max(knot_count, point_count)):
            if j < knot_count:
                mixed.append((True, knot_list[j]))
            if j < point_count:
                mixed.append((False, point_list[j]))
        length_total, previous, tknots, tpoints = 0, None, [], []
        for is_knot, index in mixed:
            location = mathutils.Vector(points_co[index])
            if previous is None:
                previous = location
            length_total += (location - previous).length
            if is_knot:
                tknots.append(length_total)
            else:
                tpoints.append(length_total)
            previous = location
        if regular:
            regular_tpoints = []
            for point_index in range(len(tpoints)):
                if point_index + 1 < len(tknots):
                    regular_tpoints.append((tknots[point_index] + tknots[point_index + 1]) / 2.0)
                else:
                    regular_tpoints.append(tpoints[point_index])
            tpoints = regular_tpoints
        all_tknots.append(tknots)
        all_tpoints.append(tpoints)
    return all_tknots, all_tpoints


def relax_calculate_verts(interpolation, tknots, knots, tpoints, points_indices, splines):
    """New values for every relaxed point, as (index_in_run, values) pairs."""
    moves = []
    for i in range(len(knots)):
        point_list, tknot_list, tpoint_list, run_splines = points_indices[i], tknots[i], tpoints[i], splines[i]
        for j, point_index in enumerate(point_list):
            if j >= len(tpoint_list):
                continue
            distance, segment_index = tpoint_list[j], -1
            if distance in tknot_list:
                segment_index = tknot_list.index(distance)
            else:
                for knot_index in range(len(tknot_list)):
                    if tknot_list[knot_index] > distance:
                        segment_index = knot_index - 1
                        break
                if segment_index == -1:
                    segment_index = len(tknot_list) - 1
            segment_index = max(0, min(segment_index, len(run_splines) - 1))
            moves.append((point_index, evaluate_spline(interpolation, run_splines[segment_index], distance)))
    return moves


def space_calculate_t(points_co):
    """Measured arc-length parameters, plus the evenly spaced ones they should become."""
    tknots, previous, length_total = [], None, 0
    for location in points_co:
        location = mathutils.Vector(location)
        if previous is None:
            previous = location
        length_total += (location - previous).length
        tknots.append(length_total)
        previous = location
    amount = len(points_co)
    if amount < 2:
        return tknots, tknots
    per_segment = length_total / (amount - 1)
    tpoints = [i * per_segment for i in range(amount)]
    return tknots, tpoints


def space_calculate_verts(interpolation, tknots, tpoints, splines):
    """New values for every evenly spaced point, as (index_in_run, values) pairs."""
    moves = []
    for i, distance in enumerate(tpoints):
        segment_index = -1
        for knot_index in range(len(tknots) - 1):
            if tknots[knot_index] <= distance <= tknots[knot_index + 1]:
                segment_index = knot_index
                break
        if segment_index == -1:
            if distance <= tknots[0]:
                segment_index = 0
            elif distance >= tknots[-1]:
                segment_index = len(tknots) - 2
        segment_index = max(0, min(segment_index, len(splines) - 1))
        moves.append((i, evaluate_spline(interpolation, splines[segment_index], distance)))
    return moves
