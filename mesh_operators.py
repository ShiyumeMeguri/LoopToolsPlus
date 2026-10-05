"""
LoopTools-style reshaping of mesh vertex selections.

Every mesh in edit mode takes part at once, measured in world space, so the line is the one the
viewport shows even under non-uniform object scale. Vertices sitting at exactly the same world
position, like the two sides of a split seam, are one point of the line and move together.
"""

import bmesh
from bpy.types import Operator
from mathutils import Vector

from .line_fitting import linear_calculate_verts


class LOOPTOOLSPLUS_OT_mesh_linear(Operator):
    bl_idname = "looptools_plus.mesh_linear"
    bl_label = "Set Linear"
    bl_description = ("Line the selected vertices up between the two that lie farthest apart, "
                      "evenly spaced in the order they already follow along that line")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and obj.mode == 'EDIT'

    def execute(self, context):
        meshes = []
        points = {}
        singular = 0
        for obj in context.objects_in_mode_unique_data:
            to_world = obj.matrix_world
            try:
                to_local = to_world.inverted()
            except ValueError:
                singular += 1
                continue
            bmesh_data = bmesh.from_edit_mesh(obj.data)
            selected = [vert for vert in bmesh_data.verts if vert.select]
            if not selected:
                continue
            meshes.append((obj, bmesh_data))
            for vert in selected:
                points.setdefault((to_world @ vert.co).to_tuple(), []).append((vert, to_local))

        if singular:
            self.report({'WARNING'}, "Skipped %d object(s) whose transform cannot be inverted" % singular)
        if len(points) < 3:
            self.report({'WARNING'}, "Select at least three vertices at different positions")
            return {'CANCELLED'}

        locations = list(points)
        for index, location in linear_calculate_verts(locations):
            target = Vector(location)
            for vert, to_local in points[locations[index]]:
                vert.co = to_local @ target

        for obj, bmesh_data in meshes:
            bmesh_data.normal_update()
            bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
        return {'FINISHED'}


classes = (
    LOOPTOOLSPLUS_OT_mesh_linear,
)
