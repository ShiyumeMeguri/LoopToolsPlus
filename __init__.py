bl_info = {
    "name": "LoopTools Plus",
    "author": "ShiyumeMeguri",
    "version": (0, 4, 1),
    "blender": (5, 3, 0),
    "location": "Edit Mode Context Menu (W) / UV Editor Context Menu (W)",
    "description": "LoopTools functionality for Curves, UVs and mesh vertices",
    "warning": "",
    "doc_url": "",
    "category": "User",
}

import bpy

from . import curve_operators
from . import mesh_operators
from . import uv_operators


class LOOPTOOLSPLUS_MT_menu(bpy.types.Menu):
    bl_label = "LoopTools Plus"
    bl_idname = "LOOPTOOLSPLUS_MT_menu"

    def draw(self, context):
        space = context.space_data
        if space and space.type == 'IMAGE_EDITOR':
            self.draw_uv(context)
            return
        if context.mode == 'EDIT_MESH':
            self.draw_mesh(context)
            return
        if context.active_object and context.active_object.type == 'CURVE':
            self.draw_curve(context)

    def draw_mesh(self, context):
        self.layout.operator("looptools_plus.mesh_linear", text="Set Linear")

    def draw_uv(self, context):
        layout = self.layout
        layout.operator("looptools_plus.uv_circle", text="Circle")
        layout.operator("looptools_plus.uv_flatten", text="Flatten")
        layout.separator()
        layout.operator("looptools_plus.uv_relax", text="Relax")
        layout.operator("looptools_plus.uv_space", text="Space")
        layout.separator()
        layout.operator("looptools_plus.uv_match", text="Match Edges")

    def draw_curve(self, context):
        layout = self.layout
        layout.operator("looptools_plus.curve_circle", text="Circle")
        layout.operator("looptools_plus.curve_flatten", text="Flatten")
        layout.separator()

        relax = layout.operator("looptools_plus.curve_relax", text="Relax")
        relax.relax_position = True
        relax.relax_tilt = False
        relax.relax_radius = False
        relax.opt_lock_length = False
        relax.opt_lock_tilt = False
        relax.opt_lock_radius = False
        relax.regular = True

        tilt = layout.operator("looptools_plus.curve_relax", text="Relax Tilt")
        tilt.relax_position = False
        tilt.relax_tilt = True
        tilt.relax_radius = False
        tilt.opt_lock_length = False
        tilt.opt_lock_tilt = True
        tilt.opt_lock_radius = False

        radius = layout.operator("looptools_plus.curve_relax", text="Relax Radius")
        radius.relax_position = False
        radius.relax_radius = True
        radius.relax_tilt = False
        radius.opt_lock_length = False
        radius.opt_lock_tilt = False
        radius.opt_lock_radius = True

        layout.operator("looptools_plus.curve_space", text="Space")
        layout.separator()
        layout.operator("looptools_plus.curve_linear", text="Linear")
        layout.operator("looptools_plus.curve_radius", text="Uniform Size")


def menu_func(self, context):
    self.layout.menu("LOOPTOOLSPLUS_MT_menu")


def menu_func_mesh(self, context):
    if context.tool_settings.mesh_select_mode[0]:
        menu_func(self, context)


def registered_classes():
    return curve_operators.classes + mesh_operators.classes + uv_operators.classes + (LOOPTOOLSPLUS_MT_menu,)


def register():
    for cls in registered_classes():
        bpy.utils.register_class(cls)
    if hasattr(bpy.types, "VIEW3D_MT_edit_curve_context_menu"):
        bpy.types.VIEW3D_MT_edit_curve_context_menu.prepend(menu_func)
    if hasattr(bpy.types, "VIEW3D_MT_edit_mesh_context_menu"):
        bpy.types.VIEW3D_MT_edit_mesh_context_menu.prepend(menu_func_mesh)
    if hasattr(bpy.types, "IMAGE_MT_uvs_context_menu"):
        bpy.types.IMAGE_MT_uvs_context_menu.prepend(menu_func)


def unregister():
    if hasattr(bpy.types, "IMAGE_MT_uvs_context_menu"):
        bpy.types.IMAGE_MT_uvs_context_menu.remove(menu_func)
    if hasattr(bpy.types, "VIEW3D_MT_edit_mesh_context_menu"):
        bpy.types.VIEW3D_MT_edit_mesh_context_menu.remove(menu_func_mesh)
    if hasattr(bpy.types, "VIEW3D_MT_edit_curve_context_menu"):
        bpy.types.VIEW3D_MT_edit_curve_context_menu.remove(menu_func)
    for cls in reversed(registered_classes()):
        bpy.utils.unregister_class(cls)
