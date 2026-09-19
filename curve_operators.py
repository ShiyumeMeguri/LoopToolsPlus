"""
LoopTools-style reshaping of curve control points.

Curve data can only be written outside edit mode, so every operator collects its jobs first and
then applies them inside :meth:`CurveLoopToolsBase.maintain_curve_mode`.
"""

import bpy
import contextlib
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
from .plane_fitting import calculate_plane


class CurveLoopToolsBase:
    def get_segments(self, spline):
        points = spline.bezier_points if spline.type == 'BEZIER' else spline.points
        num_points = len(points)
        if num_points < 2: return []
        sel_mask = [p.select_control_point if spline.type == 'BEZIER' else p.select for p in points]
        if not any(sel_mask): return []
        
        segments = []
        if all(sel_mask): segments.append(list(range(num_points)))
        else:
            curr = []
            for i, s in enumerate(sel_mask):
                if s: curr.append(i)
                else:
                    if curr: segments.append(curr); curr = []
            if curr:
                if spline.use_cyclic_u and sel_mask[0] and segments and segments[0][0] == 0:
                    segments[0] = curr + segments[0]
                else: segments.append(curr)
        return segments
    
    @contextlib.contextmanager
    def maintain_curve_mode(self, context):
        is_edit_mode = (context.object.mode == 'EDIT')
        if is_edit_mode:
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            yield
        finally:
            if is_edit_mode:
                bpy.ops.object.mode_set(mode='EDIT')

# ########################################
# ##### Curve Operators ##################
# ########################################

class LOOPTOOLSPLUS_OT_curve_relax(Operator, CurveLoopToolsBase):
    bl_idname = "looptools_plus.curve_relax"
    bl_label = "Relax"
    bl_description = "Relax the curve, smoothing it out"
    bl_options = {'REGISTER', 'UNDO'}

    relax_position: BoolProperty(name="Relax Position", default=True)
    relax_tilt: BoolProperty(name="Relax Tilt", default=False)
    relax_radius: BoolProperty(name="Relax Radius", default=False)
    opt_lock_length: BoolProperty(name="Lock Length", default=False) 
    opt_lock_tilt: BoolProperty(name="Lock Tilt", default=False)
    opt_lock_radius: BoolProperty(name="Lock Radius", default=False)
    regular: BoolProperty(name="Regular", default=True, description="Distribute points evenly")
    interpolation: EnumProperty(
        name="Interpolation",
        items=(("cubic", "Cubic", "Natural cubic spline"),
               ("linear", "Linear", "Simple linear interpolation")),
        default='cubic'
    )
    iterations: IntProperty(name="Iterations", default=1, min=1, max=50)

    def execute(self, context):
        jobs = []
        for obj in context.selected_objects:
            if obj.type != 'CURVE': continue
            for i, spline in enumerate(obj.data.splines):
                segments = self.get_segments(spline)
                if not segments: continue
                for seg in segments:
                    if len(seg) < 3: continue
                    # Store (object, spline_index, segment_indices)
                    jobs.append((obj, i, seg))

        if not jobs: return {'CANCELLED'}

        with self.maintain_curve_mode(context):
            for obj, spline_idx, seg in jobs:
                spline = obj.data.splines[spline_idx]
                points = spline.bezier_points if spline.type == 'BEZIER' else spline.points
                num_points = len(points)
                
                attrs = []
                if self.relax_position: attrs.append('position')
                if self.relax_tilt: attrs.append('tilt')
                if self.relax_radius: attrs.append('radius')
                if not attrs: continue
                
                for _ in range(self.iterations):
                    pts_co = [points[i].co.to_3d() for i in seg]
                    seg_data = []
                    for i in seg:
                        d, p = [], points[i]
                        if self.relax_position: d.extend([p.co.x, p.co.y, p.co.z])
                        if self.relax_tilt: d.append(p.tilt)
                        if self.relax_radius: d.append(p.radius)
                        seg_data.append(d)
                    circ = (spline.use_cyclic_u and len(seg) == num_points)
                    ki, pi = relax_calculate_knots(len(seg_data), circ)
                    tk, tp = relax_calculate_t(pts_co, ki, pi, self.regular)
                    spls = []
                    for kp in range(len(ki)):
                        kd, t = [seg_data[k] for k in ki[kp]], tk[kp]
                        s = calculate_cubic_splines(t, kd) if self.interpolation == 'cubic' else calculate_linear_splines(t, kd)
                        spls.append(s)
                    moves = relax_calculate_verts(self.interpolation, tk, ki, tp, pi, spls)
                    for l_idx, n_vals in moves:
                        p = points[seg[l_idx]]; offset = 0
                        if self.relax_position:
                            o_co = p.co.to_3d(); n_co = (o_co + mathutils.Vector(n_vals[0:3])) / 2
                            if spline.type == 'BEZIER':
                                delta = n_co - o_co; p.handle_left += delta; p.handle_right += delta; p.co = n_co
                            else:
                                w = p.co[3]; p.co = n_co.to_4d(); p.co[3] = w
                            offset += 3
                        if self.relax_tilt: p.tilt = (p.tilt + n_vals[offset]) / 2; offset += 1
                        if self.relax_radius: p.radius = (p.radius + n_vals[offset]) / 2; offset += 1
        return {'FINISHED'}

class LOOPTOOLSPLUS_OT_curve_space(Operator, CurveLoopToolsBase):
    bl_idname = "looptools_plus.curve_space"
    bl_label = "Space"
    bl_description = "Space points evenly along the curve"
    bl_options = {'REGISTER', 'UNDO'}
    interpolation: EnumProperty(name="Interpolation", items=(("cubic", "Cubic", ""), ("linear", "Linear", "")), default='cubic')
    influence: FloatProperty(name="Influence", default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE')
    lock_x: BoolProperty(name="Lock X", default=False); lock_y: BoolProperty(name="Lock Y", default=False); lock_z: BoolProperty(name="Lock Z", default=False)

    def execute(self, context):
        jobs = []
        for obj in context.selected_objects:
            if obj.type != 'CURVE': continue
            for i, spline in enumerate(obj.data.splines):
                segments = self.get_segments(spline)
                if not segments: continue
                for seg in segments:
                    if len(seg) < 2: continue
                    jobs.append((obj, i, seg))
                    
        if not jobs: return {'CANCELLED'}

        with self.maintain_curve_mode(context):
            for obj, spline_idx, seg in jobs:
                spline = obj.data.splines[spline_idx]
                points = spline.bezier_points if spline.type == 'BEZIER' else spline.points
                
                pts_co = [points[i].co.to_3d() for i in seg]
                seg_data = []
                for i in seg:
                    p = points[i]; seg_data.append([p.co.x, p.co.y, p.co.z, p.tilt, p.radius])
                tk, tp = space_calculate_t(pts_co)
                spls = calculate_cubic_splines(tk, seg_data) if self.interpolation == 'cubic' else calculate_linear_splines(tk, seg_data)
                moves = space_calculate_verts(self.interpolation, tk, tp, spls)
                infl = self.influence / 100.0
                for l_idx, n_vals in moves:
                    p = points[seg[l_idx]]
                    o_co = p.co.to_3d(); t_co = mathutils.Vector(n_vals[0:3])
                    if self.lock_x: t_co.x = o_co.x
                    if self.lock_y: t_co.y = o_co.y
                    if self.lock_z: t_co.z = o_co.z
                    f_co = o_co.lerp(t_co, infl)
                    if spline.type == 'BEZIER':
                        delta = f_co - o_co; p.handle_left += delta; p.handle_right += delta; p.co = f_co
                    else:
                        w = p.co[3]; p.co = f_co.to_4d(); p.co[3] = w
                    p.tilt = p.tilt + (n_vals[3] - p.tilt) * infl
                    p.radius = p.radius + (n_vals[4] - p.radius) * infl
        return {'FINISHED'}

class LOOPTOOLSPLUS_OT_curve_linear(Operator, CurveLoopToolsBase):
    bl_idname = "looptools_plus.curve_linear"
    bl_label = "Linear"
    bl_description = "Linearly interpolate points between start and end, distributing them evenly"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        jobs = []
        for obj in context.selected_objects:
            if obj.type != 'CURVE': continue
            for i, spline in enumerate(obj.data.splines):
                segments = self.get_segments(spline)
                if not segments: continue
                for seg in segments:
                    if len(seg) < 3: continue
                    jobs.append((obj, i, seg))

        if not jobs: return {'CANCELLED'}

        with self.maintain_curve_mode(context):
            for obj, spline_idx, seg in jobs:
                spline = obj.data.splines[spline_idx]
                points = spline.bezier_points if spline.type == 'BEZIER' else spline.points
                
                p_start = points[seg[0]].co.to_3d()
                p_end = points[seg[-1]].co.to_3d()
                
                diff = p_end - p_start
                dist_total = diff.length
                direction = diff.normalized() if dist_total > 0 else mathutils.Vector((0,0,0))
                
                for i, idx in enumerate(seg):
                    # Calculate target position
                    factor = i / (len(seg) - 1)
                    new_co = p_start + diff * factor
                    
                    p = points[idx]
                    if spline.type == 'BEZIER':
                        p.co = new_co
                        # Align handles to the line
                        # Calculate handle length (approximate)
                        h_len = dist_total / (len(seg) - 1) * 0.39 # 0.39 is a common handle factor for smooth circle, usage varies
                        
                        p.handle_left = new_co - direction * h_len
                        p.handle_right = new_co + direction * h_len
                        p.handle_left_type = 'ALIGNED'
                        p.handle_right_type = 'ALIGNED'
                    else:
                        w = p.co[3]
                        p.co = new_co.to_4d()
                        p.co[3] = w

        return {'FINISHED'}

class LOOPTOOLSPLUS_OT_curve_radius(Operator, CurveLoopToolsBase):
    bl_idname = "looptools_plus.curve_radius"
    bl_label = "Uniform Size"
    bl_description = "Set radius of selected points to the average radius"
    bl_options = {'REGISTER', 'UNDO'}
    
    mode: EnumProperty(
        name="Mode",
        items=(("average", "Average", "Average radius of selection"), 
               ("set", "Set", "Set to specific value")),
        default='average'
    )
    radius: FloatProperty(name="Radius", default=1.0, min=0.0)

    def execute(self, context):
        jobs = []
        for obj in context.selected_objects:
            if obj.type != 'CURVE': continue
            for i, spline in enumerate(obj.data.splines):
                segments = self.get_segments(spline)
                if not segments: continue
                # We need points to calculate average during collection phase if desired, 
                # or we can collect and calculate later. 
                # Calculating later inside Object mode is safer.
                jobs.append((obj, i, segments))
        
        if not jobs: return {'CANCELLED'}

        with self.maintain_curve_mode(context):
            for obj, spline_idx, segments in jobs:
                spline = obj.data.splines[spline_idx]
                points = spline.bezier_points if spline.type == 'BEZIER' else spline.points
                
                # First pass: Calculate average if needed
                target_radius = self.radius
                if self.mode == 'average':
                    total = 0.0
                    count = 0
                    for seg in segments:
                        for idx in seg:
                            total += points[idx].radius
                            count += 1
                    if count > 0:
                        target_radius = total / count
                
                # Second pass: Apply
                for seg in segments:
                    for idx in seg:
                        points[idx].radius = target_radius
                        
        return {'FINISHED'}

class LOOPTOOLSPLUS_OT_curve_flatten(Operator, CurveLoopToolsBase):
    bl_idname = "looptools_plus.curve_flatten"
    bl_label = "Flatten"
    bl_description = "Project curve points onto a best-fit plane"
    bl_options = {'REGISTER', 'UNDO'}
    
    influence: FloatProperty(name="Influence", default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE')
    alignment: EnumProperty(
        name="Alignment",
        items=(("best", "Best Fit", ""), ("view", "View", ""), ("x", "World X", ""), ("y", "World Y", ""), ("z", "World Z", "")),
        default='best'
    )

    def execute(self, context):
        view_mat = None
        if self.alignment == 'view':
            rv3d = context.region_data
            if rv3d: view_mat = rv3d.view_matrix.copy()

        jobs = []
        for obj in context.selected_objects:
            if obj.type != 'CURVE': continue
            for i, spline in enumerate(obj.data.splines):
                segments = self.get_segments(spline)
                if not segments: continue
                for seg in segments:
                    if len(seg) < 3: continue
                    jobs.append((obj, i, seg))

        if not jobs: return {'CANCELLED'}

        infl = self.influence / 100.0
        with self.maintain_curve_mode(context):
            for obj, spline_idx, seg in jobs:
                spline = obj.data.splines[spline_idx]
                points = spline.bezier_points if spline.type == 'BEZIER' else spline.points
                
                pts = [points[idx].co.to_3d() for idx in seg]
                
                # 1. Determine Plane (Center and Normal)
                if self.alignment in {'best', 'view'}:
                    center, normal = calculate_plane(pts, self.alignment, view_mat)
                else: # X, Y, Z
                    center = sum(pts, mathutils.Vector()) / len(pts)
                    normal = mathutils.Vector((0, 0, 0))
                    if self.alignment == 'x': normal.x = 1
                    elif self.alignment == 'y': normal.y = 1
                    elif self.alignment == 'z': normal.z = 1
                
                # 2. Project
                for idx in seg:
                    p = points[idx]
                    orig_co = p.co.to_3d()
                    
                    # Project onto plane defined by center and normal
                    # proj = p - n * dot(p - center, n)
                    proj = orig_co - normal * (orig_co - center).dot(normal)
                    
                    target = orig_co.lerp(proj, infl)
                    if spline.type == 'BEZIER':
                        delta = target - orig_co
                        p.handle_left += delta
                        p.handle_right += delta
                        p.co = target
                    else:
                        w = p.co[3]
                        p.co = target.to_4d()
                        p.co[3] = w
        return {'FINISHED'}

class LOOPTOOLSPLUS_OT_curve_circle(Operator, CurveLoopToolsBase):
    bl_idname = "looptools_plus.curve_circle"
    bl_label = "Circle"
    bl_description = "Arrange curve points into a circle"
    bl_options = {'REGISTER', 'UNDO'}
    
    fit: EnumProperty(name="Fit", items=(("circle", "Circle", "Full 360 circle"), ("arc", "Arc", "Partial arc")), default='circle')
    alignment: EnumProperty(
        name="Alignment",
        items=(("best", "Best Fit", ""), ("view", "View", ""), ("x", "World X", ""), ("y", "World Y", ""), ("z", "World Z", "")),
        default='best'
    )
    regular: BoolProperty(name="Regular", default=True)
    influence: FloatProperty(name="Influence", default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE')

    def execute(self, context):
        view_mat = None
        if self.alignment == 'view':
            rv3d = context.region_data
            if rv3d: view_mat = rv3d.view_matrix.copy()
            else: self.alignment = 'best'

        jobs = []
        for obj in context.selected_objects:
            if obj.type != 'CURVE': continue
            for i, spline in enumerate(obj.data.splines):
                points = spline.bezier_points if spline.type == 'BEZIER' else spline.points
                num_points = len(points)
                segments = self.get_segments(spline)
                if not segments: continue
                # Pass num_points too
                for seg in segments:
                    if len(seg) < 3: continue
                    jobs.append((obj, i, seg, num_points))

        if not jobs: return {'CANCELLED'}

        infl = self.influence / 100.0
        with self.maintain_curve_mode(context):
            for obj, spline_idx, seg, num_points in jobs:
                spline = obj.data.splines[spline_idx]
                points = spline.bezier_points if spline.type == 'BEZIER' else spline.points
                
                pts = [points[idx].co.to_3d() for idx in seg]
                center = sum(pts, mathutils.Vector()) / len(seg)
                
                # 1. Determine Normal/Plane
                if self.alignment == 'best':
                    normal = mathutils.Vector()
                    for i in range(len(seg) - 2):
                        v1 = pts[i+1] - pts[i]; v2 = pts[i+2] - pts[i]
                        normal += v1.cross(v2)
                    if normal.length < 1e-7: normal = mathutils.Vector((0, 0, 1))
                    else: normal.normalize()
                elif self.alignment == 'view':
                    normal = view_mat.to_3x3().inverted().transposed() @ mathutils.Vector((0, 0, 1))
                else:
                    normal = mathutils.Vector((0,0,0))
                    if self.alignment == 'x': normal.x = 1
                    elif self.alignment == 'y': normal.y = 1
                    elif self.alignment == 'z': normal.z = 1
                
                # 2. Project and Radius
                proj_pts = []
                radius = 0.0
                for p_co in pts:
                    proj = p_co - normal * (p_co - center).dot(normal)
                    proj_pts.append(proj)
                    radius += (proj - center).length
                radius /= len(seg)
                if radius < 1e-7: continue
                
                # 3. Axes
                axis_x = (proj_pts[0] - center).normalized()
                axis_y = normal.cross(axis_x).normalized()
                
                cyclic = (spline.use_cyclic_u and len(seg) == num_points)
                
                if self.regular:
                    angles = [math.atan2((p - center).dot(axis_y), (p - center).dot(axis_x)) for p in proj_pts]
                    for i in range(1, len(angles)):
                        while angles[i] - angles[i-1] > math.pi: angles[i] -= 2*math.pi
                        while angles[i] - angles[i-1] < -math.pi: angles[i] += 2*math.pi
                    
                    start_angle = angles[0]
                    end_angle = angles[-1]
                    
                    for i, idx in enumerate(seg):
                        if cyclic or self.fit == 'circle':
                            # Loop over full 360. If open, overlap endpoints by default or distribute N?
                            # To overlap: 1.0 / (len - 1), to gap: 1.0 / len
                            # User says "perfectly closed", so overlap endpoints if open.
                            div = len(seg) if cyclic else (len(seg) - 1)
                            angle = start_angle + i * (2 * math.pi / div)
                        else:
                            angle = start_angle + (end_angle - start_angle) * (i / (len(seg) - 1))
                        
                        target = center + axis_x * math.cos(angle) * radius + axis_y * math.sin(angle) * radius
                        orig_co = points[idx].co.to_3d()
                        res = orig_co.lerp(target, infl)
                        p = points[idx]
                        if spline.type == 'BEZIER':
                            delta = res - orig_co; p.handle_left += delta; p.handle_right += delta; p.co = res
                        else:
                            w = p.co[3]; p.co = res.to_4d(); p.co[3] = w
                else:
                    for i, idx in enumerate(seg):
                        vec = proj_pts[i] - center
                        if vec.length > 1e-7:
                            target = center + vec.normalized() * radius
                            orig_co = points[idx].co.to_3d()
                            res = orig_co.lerp(target, infl)
                            p = points[idx]
                            if spline.type == 'BEZIER':
                                delta = res - orig_co; p.handle_left += delta; p.handle_right += delta; p.co = res
                            else:
                                w = p.co[3]; p.co = res.to_4d(); p.co[3] = w
        return {'FINISHED'}

classes = (
    LOOPTOOLSPLUS_OT_curve_relax,
    LOOPTOOLSPLUS_OT_curve_space,
    LOOPTOOLSPLUS_OT_curve_linear,
    LOOPTOOLSPLUS_OT_curve_radius,
    LOOPTOOLSPLUS_OT_curve_flatten,
    LOOPTOOLSPLUS_OT_curve_circle,
)
