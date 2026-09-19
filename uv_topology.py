"""
Reading the UV selection and turning it into ordered runs of UV vertices.

Blender 5.x keeps the UV selection on the mesh loops themselves -- ``BMLoop.uv_select_vert``,
``BMLoop.uv_select_edge`` and ``BMFace.uv_select`` -- and those flags are only authoritative
while ``BMesh.uv_select_sync_valid`` is set. The three-way dispatch below is the same one
Blender uses in ``scripts/startup/bl_operators/uvcalc_transform.py``
(``is_face_uv_selected_fn_from_context`` / ``is_loop_edge_uv_selected_fn_from_context``):

    sync on,  flags valid   -> read the UV flags
    sync on,  flags invalid -> read the mesh selection
    sync off                -> the face must be mesh-selected to be visible, then read the UV flags

A UV vertex is every loop that sits on the same mesh vertex at the same UV coordinate, which is
what Blender calls a shared location. Runs are built out of UV edges rather than loose vertices,
so selecting a border gives exactly that border and nothing fans out into the interior. An edge
counts as selected when its own UV edge flag is set or when both of its UV corners are selected --
selecting vertices without the editor flushing that down to the edges must not leave the tools
with nothing to work on.
"""

import bisect
import math
import mathutils


def uv_vertex_selected_test(scene, bm):
    """Predicate telling whether the UV corner at a loop is selected."""
    if scene.tool_settings.use_uv_select_sync:
        if bm.uv_select_sync_valid:
            return lambda loop: not loop.face.hide and loop.uv_select_vert
        return lambda loop: not loop.face.hide and loop.vert.select
    return lambda loop: not loop.face.hide and loop.face.select and loop.uv_select_vert


def uv_edge_selected_test(scene, bm):
    """Predicate telling whether the UV edge running from a loop to the next one is selected."""
    if scene.tool_settings.use_uv_select_sync:
        if bm.uv_select_sync_valid:
            return lambda loop: not loop.face.hide and loop.uv_select_edge
        return lambda loop: not loop.face.hide and loop.edge.select
    return lambda loop: not loop.face.hide and loop.face.select and loop.uv_select_edge


class UVNode:
    """One UV vertex: every loop sharing a mesh vertex and a UV coordinate moves as one."""

    __slots__ = ("key", "uv_layer", "loops")

    def __init__(self, key, uv_layer):
        self.key = key
        self.uv_layer = uv_layer
        self.loops = []

    @property
    def uv(self):
        return self.loops[0][self.uv_layer].uv.copy()

    def write(self, uv):
        for loop in self.loops:
            loop[self.uv_layer].uv = uv


class UVRun:
    """An ordered chain or ring of UV vertices connected by selected UV edges."""

    __slots__ = ("nodes", "circular")

    def __init__(self, nodes, circular):
        self.nodes = nodes
        self.circular = circular

    def __len__(self):
        return len(self.nodes)

    @property
    def positions(self):
        return [node.uv for node in self.nodes]

    @property
    def center(self):
        positions = self.positions
        total = mathutils.Vector((0.0, 0.0))
        for position in positions:
            total += position
        return total / len(positions)


def node_key(loop, uv_layer):
    uv = loop[uv_layer].uv
    return (loop.vert.index, uv.x, uv.y)


def order_edge_keys_into_runs(edge_keys):
    """
    Sorts unordered node-key pairs into ordered runs, flagging the ones that close on themselves.

    This is the UV-space port of LoopTools' ``get_connected_selections``; branching selections
    are consumed greedily and come back as several runs, exactly like the mesh version.
    """
    node_neighbours = {}
    for first, second in edge_keys:
        node_neighbours.setdefault(first, []).append(second)
        node_neighbours.setdefault(second, []).append(first)

    runs = []
    while node_neighbours:
        start = next(iter(node_neighbours))
        run = [start]
        visited = {start}
        growing = True
        flipped = False

        while growing:
            tail = run[-1]
            if tail not in node_neighbours:
                if not flipped:
                    run.reverse()
                    flipped = True
                else:
                    growing = False
                continue

            extended = False
            for index, candidate in enumerate(node_neighbours[tail]):
                if candidate in visited:
                    continue
                node_neighbours[tail].pop(index)
                if not node_neighbours[tail]:
                    del node_neighbours[tail]
                if candidate in node_neighbours:
                    if len(node_neighbours[candidate]) == 1:
                        del node_neighbours[candidate]
                    else:
                        node_neighbours[candidate].remove(tail)
                run.append(candidate)
                visited.add(candidate)
                extended = True
                break

            if not extended:
                if not flipped:
                    run.reverse()
                    flipped = True
                else:
                    growing = False

        circular = False
        head, tail = run[0], run[-1]
        if len(run) > 2 and head in node_neighbours and tail in node_neighbours[head]:
            circular = True
            if len(node_neighbours[head]) == 1:
                del node_neighbours[head]
            else:
                node_neighbours[head].remove(tail)
            if len(node_neighbours[tail]) == 1:
                del node_neighbours[tail]
            else:
                node_neighbours[tail].remove(head)

        runs.append((run, circular))

    return runs


def collect_uv_runs(scene, bm, uv_layer):
    """Every ordered run of selected UV vertices in this bmesh, ready to be reshaped."""
    vertex_selected = uv_vertex_selected_test(scene, bm)
    edge_selected = uv_edge_selected_test(scene, bm)
    bm.verts.index_update()

    def edge_in_selection(loop):
        if edge_selected(loop):
            return True
        return vertex_selected(loop) and vertex_selected(loop.link_loop_next)

    edge_keys = set()
    for face in bm.faces:
        if face.hide:
            continue
        for loop in face.loops:
            if not edge_in_selection(loop):
                continue
            first = node_key(loop, uv_layer)
            second = node_key(loop.link_loop_next, uv_layer)
            if first == second:
                continue
            edge_keys.add((first, second) if first < second else (second, first))

    if not edge_keys:
        return []

    wanted = set()
    for first, second in edge_keys:
        wanted.add(first)
        wanted.add(second)

    nodes = {}
    for face in bm.faces:
        if face.hide:
            continue
        for loop in face.loops:
            key = node_key(loop, uv_layer)
            if key not in wanted:
                continue
            if not (vertex_selected(loop)
                    or edge_in_selection(loop)
                    or edge_in_selection(loop.link_loop_prev)):
                continue
            node = nodes.get(key)
            if node is None:
                node = UVNode(key, uv_layer)
                nodes[key] = node
            node.loops.append(loop)

    runs = []
    for keys, circular in order_edge_keys_into_runs(edge_keys):
        runs.append(UVRun([nodes[key] for key in keys], circular))

    runs.sort(key=run_sort_key)
    return runs


def run_sort_key(run):
    """
    The order runs are presented in: left to right, bottom to top, tighter first, fewer first.

    UVs are 32 bit floats, so two runs that look equally centered can differ in the last digit.
    Rounding to a ten-thousandth of the UV square keeps that noise from deciding which run an
    operator treats as the first one, while still separating anything visibly apart.
    """
    center = run.center
    positions = run.positions
    spread = sum((position - center).length for position in positions) / len(positions)
    return (round(center.x, 4), round(center.y, 4), round(spread, 4), len(positions))


def signed_area(positions):
    """Twice-signed polygon area, whose sign is the winding direction in UV space."""
    total = 0.0
    count = len(positions)
    for index in range(count):
        current = positions[index]
        following = positions[(index + 1) % count]
        total += current.x * following.y - following.x * current.y
    return total * 0.5


def arc_length_table(positions, circular):
    """Cumulative distance along the run, with the closing segment included when it is a ring."""
    count = len(positions)
    spans = count if circular else count - 1
    table = [0.0]
    total = 0.0
    for index in range(spans):
        total += (positions[(index + 1) % count] - positions[index]).length
        table.append(total)
    return table, total


def sample_polyline(positions, table, total, circular, distance):
    """The point that lies the given distance along the run."""
    count = len(positions)
    if total <= 0.0:
        return positions[0].copy()
    if circular:
        distance = distance % total
    else:
        distance = min(max(distance, 0.0), total)
    index = bisect.bisect_right(table, distance) - 1
    index = max(0, min(index, len(table) - 2))
    span = table[index + 1] - table[index]
    factor = 0.0 if span <= 0.0 else (distance - table[index]) / span
    return positions[index % count].lerp(positions[(index + 1) % count], factor)


def pair_uv_runs(runs):
    """
    Groups runs into the index pairs that should be matched together.

    Ported from LoopTools' ``bridge_match_loops``: candidates are ranked by the distance between
    run centers, then the closest ones are preferred by how similar their vertex counts are.
    """
    count = len(runs)
    centers = [run.center for run in runs]
    candidates = {index: [] for index in range(count)}
    for first in range(count):
        for second in range(first + 1, count):
            distance = (centers[first] - centers[second]).length
            candidates[first].append((distance, second))
            candidates[second].append((distance, first))
    for entries in candidates.values():
        entries.sort()

    pairs = []
    taken = set()
    for index in range(count):
        if index in taken:
            continue
        available = [entry for entry in candidates[index] if entry[1] not in taken]
        if not available:
            continue
        limit = available[0][0] * 1.1
        ranked = [(abs(len(runs[index]) - len(runs[other])), distance, other)
                  for distance, other in available if distance <= limit]
        ranked.sort()
        if not ranked:
            continue
        partner = ranked[0][2]
        taken.add(index)
        taken.add(partner)
        pairs.append((index, partner))

    return pairs


def align_run_to_run(source, target, twist, reverse):
    """
    Reorders the source nodes so that node i corresponds to target node i.

    Rings are matched by winding direction and then rotated onto the target's start vertex by the
    smallest angle around their own center, the way ``bridge_calculate_lines`` does it in 3D.
    Chains are matched end to end. ``twist`` and ``reverse`` are the manual overrides.
    """
    source_nodes = list(source.nodes)
    source_positions = [node.uv for node in source_nodes]
    target_positions = target.positions
    source_center = source.center
    target_center = target.center

    if source.circular and target.circular:
        if signed_area(source_positions) * signed_area(target_positions) < 0.0:
            source_nodes.reverse()
            source_positions = [node.uv for node in source_nodes]
        anchor = target_positions[0] - target_center
        best_index = 0
        best_angle = None
        for index, position in enumerate(source_positions):
            angle = (position - source_center).angle(anchor, math.pi)
            if best_angle is None or angle < best_angle:
                best_angle = angle
                best_index = index
        source_nodes = source_nodes[best_index:] + source_nodes[:best_index]
    else:
        if source.circular:
            anchor = target_positions[0] - target_center
            distances = [((position - source_center) - anchor).length for position in source_positions]
            best_index = distances.index(min(distances))
            source_nodes = source_nodes[best_index:] + source_nodes[:best_index]
        source_positions = [node.uv for node in source_nodes]
        head = source_positions[0] - source_center
        tail = source_positions[-1] - source_center
        target_head = target_positions[0] - target_center
        target_tail = target_positions[-1] - target_center
        straight = (head - target_head).length + (tail - target_tail).length
        crossed = (head - target_tail).length + (tail - target_head).length
        if crossed < straight:
            source_nodes.reverse()

    if twist and source.circular:
        offset = twist % len(source_nodes)
        source_nodes = source_nodes[offset:] + source_nodes[:offset]

    if reverse:
        source_nodes.reverse()
        if source.circular:
            source_nodes = [source_nodes[-1]] + source_nodes[:-1]

    return source_nodes


def match_targets(source_run, target_run):
    """
    Where every node of the source run has to land to sit on the target run.

    Equal vertex counts pair up one to one. Different counts are matched by normalized arc length,
    so the shorter run is resampled along the longer one instead of bunching up at its end.
    Both runs must already be aligned by ``align_run_to_run``.
    """
    if len(source_run) == len(target_run):
        return target_run.positions

    source_positions = source_run.positions
    target_positions = target_run.positions
    source_table, source_total = arc_length_table(source_positions, source_run.circular)
    target_table, target_total = arc_length_table(target_positions, target_run.circular)

    targets = []
    for index in range(len(source_positions)):
        factor = 0.0 if source_total <= 0.0 else source_table[index] / source_total
        targets.append(sample_polyline(target_positions, target_table, target_total,
                                       target_run.circular, factor * target_total))
    return targets
