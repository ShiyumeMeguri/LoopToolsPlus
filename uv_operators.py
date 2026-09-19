"""
LoopTools-style reshaping of UV selections.

Every operator works on the ordered runs of UV vertices that :mod:`uv_topology` builds out of the
selected UV edges, so all of them behave the same way under UV sync selection, without it, and
with several meshes in edit mode at once.
"""

import bmesh
import math
import mathutils
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from bpy.types import Operator

from .interpolation import (
    calculate_cubic_splines,
    calculate_linear_splines,
    relax_calculate_knots,
    relax_calculate_t,
    relax_calculate_verts,
    space_calculate_t,
    space_calculate_verts,
)
from .uv_topology import (
    UVRun,
    align_run_to_run,
    collect_uv_runs,
    match_targets,
    pair_uv_runs,
    run_sort_key,
)

NO_SELECTION = "No UV edges selected: select a UV loop or border, not loose vertices"

INTERPOLATION_ITEMS = (
    ("cubic", "Cubic", "Natural cubic spline through the run"),
    ("linear", "Linear", "Straight segments between the run's vertices"),
)


def gather_uv_runs(context):
    """
    (object, bmesh, runs) for every mesh in edit mode that has a UV selection.

    The bmesh wrapper travels with the runs on purpose: bmesh invalidates every element it handed
    out as soon as the owning wrapper is collected, so dropping it here would leave the nodes
    pointing at dead loops.
    """
    scene = context.scene
    gathered = []
    for obj in context.objects_in_mode:
        if obj.type != 'MESH':
            continue
        bmesh_data = bmesh.from_edit_mesh(obj.data)
        uv_layer = bmesh_data.loops.layers.uv.active
        if uv_layer is None:
            continue
        runs = collect_uv_runs(scene, bmesh_data, uv_layer)
        if runs:
            gathered.append((obj, bmesh_data, runs))
    return gathered


def flush_uv_changes(obj):
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)


def build_splines(interpolation, tknots, knots):
    if interpolation == 'cubic':
        return calculate_cubic_splines(tknots, knots)
    return calculate_linear_splines(tknots, knots)


class UVRunOperator:
    """Shared plumbing: gather the runs, reshape the long enough ones, report why nothing moved."""

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and obj.mode == 'EDIT'

    def reshape_runs(self, context, minimum_nodes, reshape):
        found = 0
        reshaped = 0
        for obj, _, runs in gather_uv_runs(context):
            found += len(runs)
            changed = False
            for run in runs:
                if len(run) < minimum_nodes:
                    continue
                reshape(run)
                reshaped += 1
                changed = True
            if changed:
                flush_uv_changes(obj)

        if reshaped:
            return {'FINISHED'}
        if found:
            self.report({'WARNING'},
                        "Selected UV runs are too short, need at least %d vertices" % minimum_nodes)
        else:
            self.report({'WARNING'}, NO_SELECTION)
        return {'CANCELLED'}


class LOOPTOOLSPLUS_OT_uv_space(Operator, UVRunOperator):
    bl_idname = "looptools_plus.uv_space"
    bl_label = "Space (UV)"
    bl_description = "Space the selected UV vertices evenly along their own run"
    bl_options = {'REGISTER', 'UNDO'}

    interpolation: EnumProperty(name="Interpolation", items=INTERPOLATION_ITEMS, default='cubic')
    influence: FloatProperty(name="Influence", default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE')

    def execute(self, context):
        interpolation = self.interpolation
        influence = self.influence / 100.0

        def reshape(run):
            positions = run.positions
            if run.circular:
                positions = positions + [positions[0]]
            knots = [[position.x, position.y] for position in positions]
            tknots, tpoints = space_calculate_t(positions)
            splines = build_splines(interpolation, tknots, knots)
            for index, values in space_calculate_verts(interpolation, tknots, tpoints, splines):
                if index >= len(run.nodes):
                    continue
                node = run.nodes[index]
                node.write(node.uv.lerp(mathutils.Vector(values), influence))

        return self.reshape_runs(context, 2, reshape)


class LOOPTOOLSPLUS_OT_uv_relax(Operator, UVRunOperator):
    bl_idname = "looptools_plus.uv_relax"
    bl_label = "Relax (UV)"
    bl_description = "Smooth the selected UV run without pulling it off its own shape"
    bl_options = {'REGISTER', 'UNDO'}

    interpolation: EnumProperty(name="Interpolation", items=INTERPOLATION_ITEMS, default='cubic')
    iterations: IntProperty(name="Iterations", default=1, min=1, max=50)
    regular: BoolProperty(name="Regular", default=True, description="Distribute the points evenly")

    def execute(self, context):
        interpolation = self.interpolation
        iterations = self.iterations
        regular = self.regular

        def reshape(run):
            for _ in range(iterations):
                positions = run.positions
                knots = [[position.x, position.y] for position in positions]
                knot_indices, point_indices = relax_calculate_knots(len(knots), run.circular)
                tknots, tpoints = relax_calculate_t(positions, knot_indices, point_indices, regular)
                splines = []
                for part in range(len(knot_indices)):
                    part_knots = [knots[index] for index in knot_indices[part]]
                    splines.append(build_splines(interpolation, tknots[part], part_knots))
                moves = relax_calculate_verts(interpolation, tknots, knot_indices,
                                              tpoints, point_indices, splines)
                for index, values in moves:
                    node = run.nodes[index]
                    node.write((node.uv + mathutils.Vector(values)) / 2.0)

        return self.reshape_runs(context, 3, reshape)


class LOOPTOOLSPLUS_OT_uv_circle(Operator, UVRunOperator):
    bl_idname = "looptools_plus.uv_circle"
    bl_label = "Circle (UV)"
    bl_description = "Arrange the selected UV run on a circle"
    bl_options = {'REGISTER', 'UNDO'}

    fit: EnumProperty(
        name="Fit",
        items=(("circle", "Circle", "Distribute over a full turn"),
               ("arc", "Arc", "Keep the run's own angular span")),
        default='circle')
    regular: BoolProperty(name="Regular", default=True, description="Distribute the points evenly")
    influence: FloatProperty(name="Influence", default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE')

    def execute(self, context):
        fit = self.fit
        regular = self.regular
        influence = self.influence / 100.0

        def reshape(run):
            positions = run.positions
            minimum_u = min(position.x for position in positions)
            maximum_u = max(position.x for position in positions)
            minimum_v = min(position.y for position in positions)
            maximum_v = max(position.y for position in positions)
            center = mathutils.Vector(((minimum_u + maximum_u) / 2.0, (minimum_v + maximum_v) / 2.0))

            radius = sum((position - center).length for position in positions) / len(positions)
            if radius < 1e-7:
                return

            if regular:
                angles = []
                for position in positions:
                    offset = position - center
                    angles.append(math.atan2(offset.y, offset.x))
                for index in range(1, len(angles)):
                    while angles[index] - angles[index - 1] > math.pi:
                        angles[index] -= 2 * math.pi
                    while angles[index] - angles[index - 1] < -math.pi:
                        angles[index] += 2 * math.pi

                start_angle = angles[0]
                end_angle = angles[-1]
                for index, node in enumerate(run.nodes):
                    if run.circular or fit == 'circle':
                        divisions = len(run.nodes) if run.circular else (len(run.nodes) - 1)
                        angle = start_angle + index * (2.0 * math.pi / divisions)
                    else:
                        angle = start_angle + (end_angle - start_angle) * (index / (len(run.nodes) - 1))
                    target = center + mathutils.Vector((math.cos(angle), math.sin(angle))) * radius
                    node.write(node.uv.lerp(target, influence))
            else:
                for node in run.nodes:
                    offset = node.uv - center
                    if offset.length < 1e-7:
                        continue
                    target = center + offset.normalized() * radius
                    node.write(node.uv.lerp(target, influence))

        return self.reshape_runs(context, 3, reshape)


class LOOPTOOLSPLUS_OT_uv_flatten(Operator, UVRunOperator):
    bl_idname = "looptools_plus.uv_flatten"
    bl_label = "Flatten (UV)"
    bl_description = "Project the selected UV run onto the straight line through its ends"
    bl_options = {'REGISTER', 'UNDO'}

    influence: FloatProperty(name="Influence", default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE')

    def execute(self, context):
        influence = self.influence / 100.0

        def reshape(run):
            start = run.nodes[0].uv
            line = run.nodes[-1].uv - start
            if line.length < 1e-7:
                return
            direction = line.normalized()
            for node in run.nodes:
                current = node.uv
                target = start + direction * (current - start).dot(direction)
                node.write(current.lerp(target, influence))

        return self.reshape_runs(context, 2, reshape)


class LOOPTOOLSPLUS_OT_uv_match(Operator, UVRunOperator):
    bl_idname = "looptools_plus.uv_match"
    bl_label = "Match Edges (UV)"
    bl_description = "Snap one selected UV run onto another one, vertex for vertex, without building geometry"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(
        name="Direction",
        items=(("FORWARD", "First onto Second",
                "Move the first run of each pair; runs are ordered left to right, "
                "bottom to top, then tighter first"),
               ("BACKWARD", "Second onto First", "Move the second run of each pair instead"),
               ("MIDDLE", "Meet Halfway", "Move both runs onto the shape between them")),
        default='FORWARD')
    reverse: BoolProperty(name="Reverse", default=False,
                          description="Walk the moving run the other way around")
    twist: IntProperty(name="Twist", default=0,
                       description="Shift which vertex matches which, for closed runs")
    influence: FloatProperty(name="Influence", default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE')

    def execute(self, context):
        influence = self.influence / 100.0

        gathered = gather_uv_runs(context)
        entries = []
        for obj, _, runs in gathered:
            for run in runs:
                entries.append((obj, run))

        if len(entries) < 2:
            self.report({'WARNING'},
                        NO_SELECTION if not entries else "Select two UV runs to match, only one found")
            return {'CANCELLED'}

        entries.sort(key=lambda entry: run_sort_key(entry[1]))
        runs = [run for _, run in entries]

        touched = set()
        matched = 0
        for first, second in pair_uv_runs(runs):
            source_run = runs[first]
            target_run = runs[second]
            aligned = UVRun(align_run_to_run(source_run, target_run, self.twist, self.reverse),
                            source_run.circular)
            source_targets = match_targets(aligned, target_run)
            target_targets = match_targets(target_run, aligned)

            if self.direction == 'FORWARD':
                self.move_nodes(aligned.nodes, source_targets, influence)
            elif self.direction == 'BACKWARD':
                self.move_nodes(target_run.nodes, target_targets, influence)
            else:
                self.move_nodes(aligned.nodes, source_targets, influence * 0.5)
                self.move_nodes(target_run.nodes, target_targets, influence * 0.5)

            touched.add(entries[first][0])
            touched.add(entries[second][0])
            matched += 1

        if not matched:
            self.report({'WARNING'}, "Could not pair the selected UV runs")
            return {'CANCELLED'}

        for obj in touched:
            flush_uv_changes(obj)
        return {'FINISHED'}

    @staticmethod
    def move_nodes(nodes, targets, factor):
        for node, target in zip(nodes, targets):
            node.write(node.uv.lerp(target, factor))


classes = (
    LOOPTOOLSPLUS_OT_uv_space,
    LOOPTOOLSPLUS_OT_uv_relax,
    LOOPTOOLSPLUS_OT_uv_circle,
    LOOPTOOLSPLUS_OT_uv_flatten,
    LOOPTOOLSPLUS_OT_uv_match,
)
