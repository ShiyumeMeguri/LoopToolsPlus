import bmesh
import bpy
import mathutils
import math
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from bpy.types import Operator
import contextlib


# ########################################
# ##### General Math functions ###########
# ########################################

def calculate_cubic_splines(tknots, knots):
    """
    Calculates natural cubic splines through all given knots.
    Adapted from LoopTools to support N-dimensions.
    """
    n = len(knots)
    if n < 2:
        return False
    
    # Check dimension
    if isinstance(knots[0], (float, int)):
        dim = 1
        locs = [[k] for k in knots]
    else:
        dim = len(knots[0])
        locs = [list(k) for k in knots]
        
    x = tknots[:]
    result = []
    
    # Solve for each dimension independently
    for j in range(dim):
        a = []
        for i in locs:
            a.append(i[j])
        h = []
        for i in range(n - 1):
            val = x[i + 1] - x[i]
            if val == 0:
                h.append(1e-8)
            else:
                h.append(val)
        
        q = [0.0]
        for i in range(1, n - 1):
            term = 3 / h[i] * (a[i + 1] - a[i]) - 3 / h[i - 1] * (a[i] - a[i - 1])
            q.append(term)
            
        l = [1.0]
        u = [0.0]
        z = [0.0]
        
        for i in range(1, n - 1):
            val = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * u[i - 1]
            if val == 0:
                val = 1e-8
            l.append(val)
            u.append(h[i] / l[i])
            z.append((q[i] - h[i - 1] * z[i - 1]) / l[i])
            
        l.append(1.0)
        z.append(0.0)
        
        b = [0.0 for _ in range(n - 1)]
        c = [0.0 for _ in range(n)]
        d = [0.0 for _ in range(n - 1)]
        
        c[n - 1] = 0.0
        for i in range(n - 2, -1, -1):
            c[i] = z[i] - u[i] * c[i + 1]
            b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
            d[i] = (c[i + 1] - c[i]) / (3 * h[i])
            
        for i in range(n - 1):
            if len(result) <= i:
                result.append([])
            result[i].append([a[i], b[i], c[i], d[i], x[i]])

    splines = []
    for i in range(n - 1):
        splines.append(result[i])
        
    return splines

def calculate_linear_splines(tknots, knots):
    """
    Calculates linear splines.
    """
    splines = []
    if isinstance(knots[0], (float, int)):
        dim = 1
        conversion = lambda x: [x]
    else:
        dim = len(knots[0])
        conversion = lambda x: list(x)
        
    for i in range(len(knots) - 1):
        a = conversion(knots[i])
        b = conversion(knots[i + 1])
        d = [b[k] - a[k] for k in range(dim)]
        t = tknots[i]
        u = tknots[i + 1] - t
        segment_splines = []
        for k in range(dim):
            segment_splines.append([a[k], d[k], t, u])
        splines.append(segment_splines)
        
    return splines

# ########################################
# ##### Relax logic ######################
# ########################################

def relax_calculate_knots(points_len, circular):
    knots = [[], []]
    points = [[], []]
    loop = list(range(points_len))
    
    if circular:
        if len(loop) % 2 == 1: extend = [False, True, 0, 1, 0, 1]
        else: extend = [True, False, 0, 1, 1, 2]
    else:
        extend = [False, False, 0, 1, 1, 2]
             
    for j in range(2):
        temp_loop = loop[:]
        if extend[j]: temp_loop = [loop[-1]] + loop + [loop[0]]
        k_indices = []
        for i in range(extend[2 + 2 * j], len(temp_loop), 2):
            k_indices.append(temp_loop[i])
        knots[j] = k_indices
        p_indices = []
        for i in range(extend[3 + 2 * j], len(temp_loop), 2):
            idx = temp_loop[i]
            if idx == loop[-1] and not circular: continue
            if len(p_indices) == 0 or idx != p_indices[0]:
                p_indices.append(idx)
        points[j] = p_indices
        if circular and knots[j][0] != knots[j][-1]:
            knots[j].append(knots[j][0])
            
    if len(points[1]) == 0:
        knots.pop(1)
        points.pop(1)
        
    return knots, points

def relax_calculate_t(points_co, knots, points_indices, regular):
    all_tknots = []
    all_tpoints = []
    for i in range(len(knots)):
        k_list, p_list = knots[i], points_indices[i]
        mix = []
        nk, np = len(k_list), len(p_list)
        max_len = max(nk, np)
        for j in range(max_len):
            if j < nk: mix.append((True, k_list[j]))
            if j < np: mix.append((False, p_list[j]))
        len_total, loc_prev, tknots, tpoints = 0, None, [], []
        for is_knot, idx in mix:
            loc = mathutils.Vector(points_co[idx])
            if loc_prev is None: loc_prev = loc
            len_total += (loc - loc_prev).length
            if is_knot: tknots.append(len_total)
            else: tpoints.append(len_total)
            loc_prev = loc
        if regular:
            new_tpoints = []
            for p_idx in range(len(tpoints)):
                if p_idx + 1 < len(tknots):
                    new_tpoints.append((tknots[p_idx] + tknots[p_idx+1]) / 2.0)
                else:
                    new_tpoints.append(tpoints[p_idx])
            tpoints = new_tpoints
        all_tknots.append(tknots)
        all_tpoints.append(tpoints)
    return all_tknots, all_tpoints

def relax_calculate_verts(interpolation, tknots, knots, tpoints, points_indices, splines):
    moves = []
    for i in range(len(knots)):
        p_list, tk, tp, seg_splines = points_indices[i], tknots[i], tpoints[i], splines[i]
        for j, p_idx in enumerate(p_list):
            if j >= len(tp): continue
            m, n = tp[j], -1
            if m in tk: n = tk.index(m)
            else:
                for k_idx in range(len(tk)):
                    if tk[k_idx] > m:
                        n = k_idx - 1
                        break
                if n == -1: n = len(tk) - 1
            n = max(0, min(n, len(seg_splines) - 1))
            new_vals = []
            if interpolation == 'cubic':
                dims = seg_splines[n]
                for d_idx in range(len(dims)):
                    a, b, c, d_coeff, tx = dims[d_idx]
                    dt = m - tx
                    new_vals.append(a + b*dt + c*(dt**2) + d_coeff*(dt**3))
            else:
                dims = seg_splines[n]
                for d_idx in range(len(dims)):
                    a, d_val, t, u = dims[d_idx]
                    if u == 0: u = 1e-8
                    new_vals.append(((m - t) / u) * d_val + a)
            moves.append((p_idx, new_vals))
    return moves

# ########################################
# ##### Space logic ######################
# ########################################

def space_calculate_t(points_co):
    tknots, loc_prev, len_total = [], None, 0
    for loc in points_co:
        loc = mathutils.Vector(loc)
        if loc_prev is None: loc_prev = loc
        len_total += (loc - loc_prev).length
        tknots.append(len_total)
        loc_prev = loc
    amount = len(points_co)
    if amount < 2: return tknots, tknots
    t_per_segment = len_total / (amount - 1)
    tpoints = [i * t_per_segment for i in range(amount)]
    return tknots, tpoints

def space_calculate_verts(interpolation, tknots, tpoints, splines):
    moves = []
    for i, m in enumerate(tpoints):
        n = -1
        for k_idx in range(len(tknots)-1):
            if tknots[k_idx] <= m <= tknots[k_idx+1]:
                n = k_idx
                break
        if n == -1:
            if m <= tknots[0]: n = 0
            elif m >= tknots[-1]: n = len(tknots) - 2
        n = max(0, min(n, len(splines) - 1))
        new_vals = []
        if interpolation == 'cubic':
            dims = splines[n]
            for d_idx in range(len(dims)):
                a, b, c, d_coeff, tx = dims[d_idx]
                dt = m - tx
                new_vals.append(a + b*dt + c*(dt**2) + d_coeff*(dt**3))
        else:
            dims = splines[n]
            for d_idx in range(len(dims)):
                a, d_val, t, u = dims[d_idx]
                if u == 0: u = 1e-8
                new_vals.append(((m - t) / u) * d_val + a)
        moves.append((i, new_vals))
    return moves

# ########################################
# ##### Curve Helper #####################
# ########################################

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

# ########################################
# ##### Plane Calculation Helper #########
# ########################################

def matrix_determinant(m):
    determinant = m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] \
        + m[0][2] * m[1][0] * m[2][1] - m[0][2] * m[1][1] * m[2][0] \
        - m[0][1] * m[1][0] * m[2][2] - m[0][0] * m[1][2] * m[2][1]
    return determinant

def matrix_invert(m):
    det = matrix_determinant(m)
    if det == 0: return None
    r = mathutils.Matrix((
        (m[1][1] * m[2][2] - m[1][2] * m[2][1], m[0][2] * m[2][1] - m[0][1] * m[2][2],
         m[0][1] * m[1][2] - m[0][2] * m[1][1]),
        (m[1][2] * m[2][0] - m[1][0] * m[2][2], m[0][0] * m[2][2] - m[0][2] * m[2][0],
         m[0][2] * m[1][0] - m[0][0] * m[1][2]),
        (m[1][0] * m[2][1] - m[1][1] * m[2][0], m[0][1] * m[2][0] - m[0][0] * m[2][1],
         m[0][0] * m[1][1] - m[0][1] * m[1][0])))
    return (r * (1 / det))

def calculate_plane(locs, method="best_fit", view_mat=None):
    # calculating the center of mass
    com = mathutils.Vector()
    for loc in locs:
        com += loc
    com /= len(locs)
    x, y, z = com

    if method == 'best':
        # creating the covariance matrix
        mat = mathutils.Matrix(((0.0, 0.0, 0.0),
                                (0.0, 0.0, 0.0),
                                (0.0, 0.0, 0.0),
                                ))
        for loc in locs:
            mat[0][0] += (loc[0] - x) ** 2
            mat[1][0] += (loc[0] - x) * (loc[1] - y)
            mat[2][0] += (loc[0] - x) * (loc[2] - z)
            mat[0][1] += (loc[1] - y) * (loc[0] - x)
            mat[1][1] += (loc[1] - y) ** 2
            mat[2][1] += (loc[1] - y) * (loc[2] - z)
            mat[0][2] += (loc[2] - z) * (loc[0] - x)
            mat[1][2] += (loc[2] - z) * (loc[1] - y)
            mat[2][2] += (loc[2] - z) ** 2

        # calculating the normal to the plane
        normal = False
        try:
            mat_inv = matrix_invert(mat)
            if mat_inv: mat = mat_inv
        except:
            pass
            
        # If inversion failed or we want to find eigenvector, LoopTools does this iterative approach
        # on the INVERTED matrix? Wait, checking the original code...
        # yes, mat = matrix_invert(mat) is called.
        
        if not normal:
             # simple axis guess if matrix is singular or just as starting point
            ax = 2
            if math.fabs(sum(mat[0])) < math.fabs(sum(mat[1])):
                if math.fabs(sum(mat[0])) < math.fabs(sum(mat[2])):
                    ax = 0
            elif math.fabs(sum(mat[1])) < math.fabs(sum(mat[2])):
                ax = 1
            if ax == 0:
                normal = mathutils.Vector((1.0, 0.0, 0.0))
            elif ax == 1:
                normal = mathutils.Vector((0.0, 1.0, 0.0))
            else:
                normal = mathutils.Vector((0.0, 0.0, 1.0))

            # warning! this is different from .normalize()
            # This logic basically finds the eigenvector corresponding to largest eigenvalue of the INVERTED matrix
            # which corresponds to smallest eigenvalue of covariance matrix => normal direction
            itermax = 500
            vec2 = mathutils.Vector((1.0, 1.0, 1.0))
            for i in range(itermax):
                vec = vec2
                vec2 = mat @ vec
                # Calculate length with double precision to avoid problems with `inf`
                vec2_length = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2)
                if vec2_length != 0:
                    vec2 /= vec2_length
                if vec2 == vec:
                    break
            if vec2.length == 0:
                vec2 = mathutils.Vector((1.0, 1.0, 1.0))
            normal = vec2

    elif method == 'view':
        # calculate view normal
        if view_mat:
            rotation = view_mat.to_3x3().inverted()
            normal = rotation @ mathutils.Vector((0.0, 0.0, 1.0))
        else:
            normal = mathutils.Vector((0.0, 0.0, 1.0))
    else:
        normal = mathutils.Vector((0.0, 0.0, 1.0))
            
    return(com, normal)

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

# ########################################
# ##### UV Operators #####################
# ########################################

class UVLoopToolsBase:
    def get_uv_paths(self, bm, uv_layer):
        """
        Groups selected BMLoops into ordered paths (chains or cycles).
        A 'UV vertex' is a group of loops at the same mesh-vert that share the same UV coordinate.
        """
        # 1. Identify all selected loops and group them by (mesh_vert, uv_coord)
        # We use a small epsilon for UV coordinate matching
        def uv_key(uv):
            return (round(uv.x, 6), round(uv.y, 6))

        # Blender 3.5+ moved UV vertex selection out of BMLoopUV into a hidden boolean
        # loop attribute named ".vs.<uvmap>"; the old BMLoopUV.select was removed in 5.x.
        # When "UV Sync Selection" is on, the UV editor follows the mesh selection instead.
        if bpy.context.scene.tool_settings.use_uv_select_sync:
            def is_uv_selected(loop):
                return loop.vert.select
        else:
            uv_select_layer = bm.loops.layers.bool.get(".vs." + uv_layer.name)
            if uv_select_layer is None:
                return []
            def is_uv_selected(loop):
                return loop[uv_select_layer]

        uv_nodes = {} # (vert, uv_key) -> [loops]
        for face in bm.faces:
            for l in face.loops:
                if is_uv_selected(l):
                    key = (l.vert, uv_key(l[uv_layer].uv))
                    if key not in uv_nodes:
                        uv_nodes[key] = []
                    uv_nodes[key].append(l)

        if not uv_nodes:
            return []

        # 2. Build adjacency graph between UV nodes
        # Two nodes are adjacent if they share a mesh edge AND that edge is part of a selected UV edge
        adj = {key: set() for key in uv_nodes}
        for face in bm.faces:
            for l in face.loops:
                l_next = l.link_loop_next
                key_curr = (l.vert, uv_key(l[uv_layer].uv))
                key_next = (l_next.vert, uv_key(l_next[uv_layer].uv))
                
                if key_curr in uv_nodes and key_next in uv_nodes:
                    # They might be connected!
                    adj[key_curr].add(key_next)
                    adj[key_next].add(key_curr)

        # 3. Traverse graph to find paths
        paths = []
        visited = set()
        
        # Keys sorted to ensure deterministic behavior
        all_keys = list(uv_nodes.keys())
        
        # Start with nodes that have only 1 neighbor (endpoints of chains)
        endpoints = [k for k in all_keys if len(adj[k]) == 1]
        for k in endpoints + [k for k in all_keys if k not in visited]:
            if k in visited:
                continue
            
            path = []
            curr = k
            while curr and curr not in visited:
                visited.add(curr)
                path.append(curr)
                # Find next neighbor not visited
                next_node = None
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        next_node = neighbor
                        break
                curr = next_node
            
            # Check for cycle if it's not a chain
            if path:
                first = path[0]
                last = path[-1]
                is_cyclic = first in adj[last] and len(path) > 2
                
                # Convert keys back to loop groups for processing
                paths.append({'nodes': [uv_nodes[node_key] for node_key in path], 'cyclic': is_cyclic})
                
        return paths

class LOOPTOOLSPLUS_OT_uv_relax(Operator, UVLoopToolsBase):
    bl_idname = "looptools_plus.uv_relax"
    bl_label = "Relax (UV)"
    bl_description = "Relax selected UV vertices"
    bl_options = {'REGISTER', 'UNDO'}
    interpolation: EnumProperty(name="Interpolation", items=(("cubic", "Cubic", ""), ("linear", "Linear", "")), default='cubic')
    iterations: IntProperty(name="Iterations", default=1, min=1, max=50)
    regular: BoolProperty(name="Regular", default=True)

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH' or obj.mode != 'EDIT': return {'CANCELLED'}
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer: return {'CANCELLED'}

        uv_paths = self.get_uv_paths(bm, uv_layer)
        if not uv_paths: return {'CANCELLED'}

        for path_data in uv_paths:
            uv_run_nodes = path_data['nodes']
            is_circular = path_data['cyclic']
            if len(uv_run_nodes) < 3: continue
            
            for _ in range(self.iterations):
                # We use the UV of the first loop in each node as representative
                pts_co = [node[0][uv_layer].uv.to_3d() for node in uv_run_nodes]
                seg_data = [[node[0][uv_layer].uv.x, node[0][uv_layer].uv.y] for node in uv_run_nodes]
                
                ki, pi = relax_calculate_knots(len(seg_data), is_circular)
                tk, tp = relax_calculate_t(pts_co, ki, pi, self.regular)
                spls = []
                for kp in range(len(ki)):
                    kd, t = [seg_data[k] for k in ki[kp]], tk[kp]
                    s = calculate_cubic_splines(t, kd) if self.interpolation == 'cubic' else calculate_linear_splines(t, kd)
                    spls.append(s)
                
                moves = relax_calculate_verts(self.interpolation, tk, ki, tp, pi, spls)
                for l_idx, n_vals in moves:
                    # Apply to ALL loops in this node
                    new_uv = mathutils.Vector(n_vals)
                    for l in uv_run_nodes[l_idx]:
                        l[uv_layer].uv = (l[uv_layer].uv + new_uv) / 2
        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}

class LOOPTOOLSPLUS_OT_uv_space(Operator, UVLoopToolsBase):
    bl_idname = "looptools_plus.uv_space"
    bl_label = "Space (UV)"
    bl_description = "Space selected UV vertices evenly"
    bl_options = {'REGISTER', 'UNDO'}
    interpolation: EnumProperty(name="Interpolation", items=(("cubic", "Cubic", ""), ("linear", "Linear", "")), default='cubic')
    influence: FloatProperty(name="Influence", default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE')

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH' or obj.mode != 'EDIT': return {'CANCELLED'}
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer: return {'CANCELLED'}

        uv_paths = self.get_uv_paths(bm, uv_layer)
        if not uv_paths: return {'CANCELLED'}

        infl = self.influence / 100.0
        for path_data in uv_paths:
            uv_run_nodes = path_data['nodes']
            if len(uv_run_nodes) < 2: continue
            
            pts_co = [node[0][uv_layer].uv.to_3d() for node in uv_run_nodes]
            seg_data = [[node[0][uv_layer].uv.x, node[0][uv_layer].uv.y] for node in uv_run_nodes]
            
            tk, tp = space_calculate_t(pts_co)
            spls = calculate_cubic_splines(tk, seg_data) if self.interpolation == 'cubic' else calculate_linear_splines(tk, seg_data)
            moves = space_calculate_verts(self.interpolation, tk, tp, spls)
            
            for l_idx, n_vals in moves:
                target = mathutils.Vector(n_vals)
                for l in uv_run_nodes[l_idx]:
                    l[uv_layer].uv = l[uv_layer].uv.lerp(target, infl)
        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}

class LOOPTOOLSPLUS_OT_uv_circle(Operator, UVLoopToolsBase):
    bl_idname = "looptools_plus.uv_circle"
    bl_label = "Circle (UV)"
    bl_options = {'REGISTER', 'UNDO'}
    
    fit: EnumProperty(name="Fit", items=(("circle", "Circle", "Full 360 circle"), ("arc", "Arc", "Partial arc")), default='circle')
    regular: BoolProperty(name="Regular", default=True, description="Distribute points evenly")
    influence: FloatProperty(name="Influence", default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE')

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH' or obj.mode != 'EDIT': return {'CANCELLED'}
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer: return {'CANCELLED'}

        uv_paths = self.get_uv_paths(bm, uv_layer)
        if not uv_paths: return {'CANCELLED'}

        infl = self.influence / 100.0

        for path_data in uv_paths:
            nodes = path_data['nodes']
            cyclic = path_data['cyclic']
            if len(nodes) < 3: continue
            
            # 1. Improved center: use bounding box center
            uvs = [node[0][uv_layer].uv for node in nodes]
            min_u = min(uv.x for uv in uvs); max_u = max(uv.x for uv in uvs)
            min_v = min(uv.y for uv in uvs); max_v = max(uv.y for uv in uvs)
            center = mathutils.Vector(((min_u + max_u)/2, (min_v + max_v)/2))
            
            # 2. Radius
            radius = sum((node[0][uv_layer].uv - center).length for node in nodes) / len(nodes)
            if radius < 1e-7: continue
            
            if self.regular:
                angles = []
                for node in nodes:
                    vec = node[0][uv_layer].uv - center
                    angles.append(math.atan2(vec.y, vec.x))
                
                for i in range(1, len(angles)):
                    while angles[i] - angles[i-1] > math.pi: angles[i] -= 2*math.pi
                    while angles[i] - angles[i-1] < -math.pi: angles[i] += 2*math.pi
                
                start_angle = angles[0]
                end_angle = angles[-1]
                
                for i, node in enumerate(nodes):
                    if cyclic or self.fit == 'circle':
                        # Distribute over full 360. If open, overlap endpoints to close.
                        div = len(nodes) if cyclic else (len(nodes) - 1)
                        angle = start_angle + i * (2.0 * math.pi / div)
                    else:
                        angle = start_angle + (end_angle - start_angle) * (i / (len(nodes) - 1))
                    
                    target = center + mathutils.Vector((math.cos(angle), math.sin(angle))) * radius
                    for l in node:
                        l[uv_layer].uv = l[uv_layer].uv.lerp(target, infl)
            else:
                for node in nodes:
                    for l in node:
                        vec = (l[uv_layer].uv - center)
                        if vec.length > 1e-7:
                            target = center + vec.normalized() * radius
                            l[uv_layer].uv = l[uv_layer].uv.lerp(target, infl)
                            
        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}

class LOOPTOOLSPLUS_OT_uv_flatten(Operator, UVLoopToolsBase):
    bl_idname = "looptools_plus.uv_flatten"
    bl_label = "Flatten (UV)"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH' or obj.mode != 'EDIT': return {'CANCELLED'}
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer: return {'CANCELLED'}

        uv_paths = self.get_uv_paths(bm, uv_layer)
        if not uv_paths: return {'CANCELLED'}

        for path_data in uv_paths:
            uv_run_nodes = path_data['nodes']
            if len(uv_run_nodes) < 2: continue
            
            p1 = uv_run_nodes[0][0][uv_layer].uv
            p2 = uv_run_nodes[-1][0][uv_layer].uv
            line = p2 - p1
            if line.length > 0:
                line_norm = line.normalized()
                for node in uv_run_nodes:
                    for l in node:
                        rel = l[uv_layer].uv - p1
                        l[uv_layer].uv = p1 + line_norm * rel.dot(line_norm)
        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}

def register():
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_curve_relax)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_curve_space)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_curve_linear)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_curve_radius)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_curve_flatten)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_curve_circle)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_uv_relax)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_uv_space)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_uv_circle)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_uv_flatten)

def unregister():
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_uv_flatten)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_uv_circle)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_uv_space)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_uv_relax)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_curve_circle)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_curve_flatten)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_curve_radius)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_curve_linear)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_curve_space)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_curve_relax)
