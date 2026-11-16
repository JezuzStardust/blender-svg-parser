from __future__ import annotations
import math
try:
    from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple, List, Dict, Self
except ImportError:
    from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple, List, Dict
    try:
        from typing_extensions import Self
    except Exception:
        Self = Any
import mathutils
import math
import bpy
import itertools
from typing import Optional, Iterator, Iterable, overload
from collections.abc import Sequence
from . import solvers
from .gauss_legendre import GAUSS_LEGENDRE_COEFFS_32
from .curve import Curve
from .matrix import Matrix
from .spline import Spline
from .vector import Vector

class Bezier(CurveObject):
    """Cubic bezier curve. p0, p1, p2, p3 are mathutils.Vector
    t0 and t1 are the parameter time at the start and the end of the 
    original curve for instances created as splits of larger curves.
    """
    __slots__ = ('points', 't0', 't1', 'start_handle_left', 'end_handle_right', 'is_closed')

    def __init__(self, p0: Vector=Vector(), p1: Vector=Vector(), p2: Vector=Vector(), p3: Vector=Vector(), start_handle_left: Vector=Vector(), end_handle_right: Vector=Vector(), t0: float=0.0, t1: float=1.0, is_closed: bool=False, name='Bezier', location: Vector=Vector(), scale: float=Vector(1.0, 1.0, 1.0), rotation=Vector()) -> None:
        """ 
        Initializes the cubic Bezier and sets its points and degree. 
        The points should be mathutils.Vectors of some fixed dimension.
        The number of points should be 3 or higher. 
        """
        super().__init__(name, location, scale, rotation)
        self.points: list[Vector] = [p0, p1, p2, p3]
        self.start_handle_left: Vector = start_handle_left
        self.end_handle_right: Vector = end_handle_right
        self.is_closed: bool = is_closed
        self.t0: float = t0
        self.t1: float = t1

    @classmethod
    def from_blender(cls, name: str) -> Any:
        """Alternative constructor to read and import a Bezier curve from Blender.
        This assumes that the named object is only a simple bezier curve, 
        if the Blender object is a spline, only the first part of the curve will
        be imported. Use Spline.from_Blender() instead in those cases."""
        cu = bpy.data.collections['Collection'].objects[name]
        spline = cu.data.splines[0]
        is_closed = spline.use_cyclic_u
        bezier_points = spline.bezier_points
        start_handle_left = Vector.from_mu_vector(bezier_points[0].handle_left)
        p0 = Vector.from_mu_vector(bezier_points[0].co)
        p1 = Vector.from_mu_vector(bezier_points[0].handle_right)
        p2 = Vector.from_mu_vector(bezier_points[1].handle_left)
        p3 = Vector.from_mu_vector(bezier_points[1].co)
        end_handle_right = Vector.from_mu_vector(bezier_points[1].handle_right)
        loc = Vector.from_mu_vector(cu.location)
        sca = Vector.from_mu_vector(cu.scale)
        rot: Vector = Vector(*cu.rotation_euler)
        return cls(p0, p1, p2, p3, name=name, location=loc, scale=sca, rotation=rot, start_handle_left=start_handle_left, end_handle_right=end_handle_right, is_closed=is_closed)

    def __repr__(self) -> str:
        """Prints the name of the together with all the points. """
        p = self.points
        return f'Bezier(\nname={self.name}, \np0={str(p[0])}, \np1={str(p[1])}, \np2={str(p[2])}, \np3={str(p[3])}, \nleft={str(self.start_handle_left)}, \nright={str(self.end_handle_right)}, \nt0={self.t0}, \nt1={self.t1}\n)'

    def __call__(self, t: float, world_space: bool=False, global_t: bool=False) -> Vector:
        """Returns the value at parameter t. 
        If world_space = False, the position is calculated relative to the origin of the Bezier.
        If global_t = True, the position is evaluated at the parameter t of the original curve (if the curve is split). 
        """
        if global_t:
            denom = self.t1 - self.t0
            if denom == 0.0:
                t = 0.0
            else:
                t = (t - self.t0) / denom
        p = self.points
        pos = p[0] * (1 - t) ** 3 + 3 * p[1] * (1 - t) ** 2 * t + 3 * p[2] * (1 - t) * t ** 2 + p[3] * t ** 3
        if world_space:
            if self.rotation.z != 0.0:
                rot = Matrix.rotation_xyz(*self.rotation)
                pos = rot @ pos + self.location
                return pos
            else:
                return pos + self.location
        else:
            return pos

    def reverse(self) -> None:
        """Reverses the direction of the curve."""
        self.points = list(reversed(self.points))
        self.start_handle_left, self.end_handle_right = (self.end_handle_right, self.start_handle_left)

    def set_point(self, point: Vector, i: int) -> None:
        """Sets the point with index i of the Bezier."""
        self.points[i] = point

    def translate_origin(self, vector: Vector) -> None:
        """Translates the origin of the Bezier to the position given by 
        vector without changing the world position of the curve. 
        """
        dist = self.location - vector
        self.points = [p + dist for p in self.points]
        self.start_handle_left = self.start_handle_left + dist
        self.end_handle_right = self.end_handle_right + dist
        self.location = vector

    def eval_derivative(self, t: float, global_t: bool=False) -> Vector:
        """Evaluate derivative at parameter t. If global_t and self was
        split from a larger curve, then the derivative is evaluated at the 
        parameter that corresponds to t of the original curve."""
        if global_t:
            denom = self.t1 - self.t0
            if denom < 1e-12:
                t = 0.0
            else:
                t = (t - self.t0) / denom
        p = self.points
        return 3 * (p[1] - p[0]) * (1 - t) ** 2 + 6 * (p[2] - p[1]) * (1 - t) * t + 3 * (p[3] - p[2]) * t ** 2

    def eval_second_derivative(self, t: float):
        """Evaluate the second derivative at parameter t."""
        p = self.points
        return 6 * (p[0] - 2 * p[1] + p[2]) * (1 - t) + 6 * (p[1] - 2 * p[2] + p[3]) * t

    def derivative(self):
        """Returns the derivative (quadratic Bezier) curve of self."""
        p = self.points
        return QuadraticBezier(3 * (p[1] - p[0]), 3 * (p[2] - p[1]), 3 * (p[3] - p[2]))

    def tangent(self, t: float):
        """Calculate the unit tangent at parameter t."""
        derivative = self.eval_derivative(t)
        if derivative.length() > 0.0:
            return derivative / derivative.length()
        else:
            return derivative

    def normal(self, t: float):
        """Return the tangent rotated 90 degrees anti-clockwise."""
        tangent = self.tangent(t)
        return mathutils.Vector((-tangent.y, tangent.x, 0.0))

    def curvature(self, t: float) -> float:
        """Returns the curvature at parameter t."""
        d = self.eval_derivative(t)
        sd = self.eval_second_derivative(t)
        denom = math.pow(d.length(), 3 / 2)
        return (d[0] * sd[1] - sd[0] * d[1]) / denom

    def x_moment(self):
        """Calculate the x_moment of a curve that starts at the origin
        and ends on the x-axis."""
        p = self.points
        x1 = p[1][0]
        y1 = p[1][1]
        x2 = p[2][0]
        y2 = p[2][1]
        x3 = p[3][0]
        moment = -9 * x1 ** 2 * y2 / 280 + 9 * x1 * x2 * y1 / 280 - 9 * x1 * x2 * y2 / 280 + 3 * x1 * x3 * y1 / 140 + 9 * x2 ** 2 * y1 / 280 + 3 * x2 * x3 * y1 / 56 + 3 * x2 * x3 * y2 / 56 + x3 ** 2 * y1 / 28 + x3 ** 2 * y2 / 8
        return moment

    def eval_offset(self, t: float, d: float):
        """Calculates the vector from self(t) to the offset at parameter t."""
        n = None
        dp = self.eval_derivative(t)
        if dp.length() > 0.0:
            s = d / dp.length()
        elif t == 0.0:
            p1 = self(0.0001)
            dp = p1 - self.points[0]
            s = d / dp.length
        elif t == 1.0:
            p2 = self(0.9999)
            dp = self.points[3] - p2
            s = d / dp.length
        else:
            dp = mathutils.Vector((0, 0, 0))
            s = 0
        return mathutils.Vector((-s * dp[1], s * dp[0], 0))

    def aligned(self):
        """Returns the points of the corresponding aligned curve. 
        Aligned means: start point in origin, end point on x-axis. 
        """
        m = mathutils.Matrix.Translation(-self.points[0])
        end = m @ self.points[3]
        if end[0] != 0.0:
            angle = -math.atan2(end[1], end[0])
        else:
            angle = 0.0
        k = mathutils.Matrix.Rotation(angle, 4, 'Z') @ m
        aligned_points = []
        for p in self.points:
            aligned_points.append(k @ p)
        return aligned_points

    def bounding_box(self, world_space=False):
        """Calculate the bounding box of the curve.
        Returns dict('min_x', 'max_x', 'min_y', 'max_y')"""
        extrema = self.extrema()
        x_values = [self(t, world_space=world_space)[0] for t in extrema]
        y_values = [self(t, world_space=world_space)[1] for t in extrema]
        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)
        area = (max_x - min_x) * (max_y - min_y)
        return {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y, 'area': area}

    def extrema(self):
        """
        Returns the parameter values for the minimum and maximum of the curve
        in the x and y coordinate. 
        """
        p0, p1, p2, p3 = self.points
        a = 3 * (-p0 + 3 * p1 - 3 * p2 + p3)
        b = 6 * (p0 - 2 * p1 + p2)
        c = 3 * (p1 - p0)
        tx_roots = solvers.solve_quadratic(a.x, b.x, c.x)
        ty_roots = solvers.solve_quadratic(a.y, b.y, c.y)
        roots = [0.0, 1.0]
        roots.extend([t for t in tx_roots if isinstance(t, float) and 0.0 <= t <= 1.0])
        roots.extend([t for t in ty_roots if isinstance(t, float) and 0.0 <= t <= 1.0])
        return roots

    def subsegment(self, t0: float, t1: float):
        """Split out the subsegment between t0 and t1.
        If the curves have been previously split, we can insert the parameter
        of the original whole curve where we want to split the curve."""

        def map_split_to_whole(t: float):
            """Returns the parameter value of the original curve corresponding
            to the parameter value t of a split.
            """
            return self.t0 + (self.t1 - self.t0) * t
        p0 = self(t0)
        p3 = self(t1)
        factor = (t1 - t0) / 3
        d1 = self.eval_derivative(t0)
        p1 = p0 + factor * d1
        d2 = self.eval_derivative(t1)
        p2 = p3 - factor * d2
        t0new = map_split_to_whole(t0)
        t1new = map_split_to_whole(t1)
        name = self.name + '(' + str(t0new) + ',' + str(t1new) + ')'
        loc = self.location
        sca = self.scale
        rot = self.rotation
        return Bezier(p0, p1, p2, p3, t0=t0new, t1=t1new, name=name, location=loc, scale=sca, rotation=rot)

    def split2(self, t: float):
        """CHANGED: added docstring and type hints

split2

Args:
    t (float): 

Returns:
    Any
"""
        p0, p1, p2, p3 = self.points
        p01: Vector = p0.lerp(p1, t)
        p12: Vector = p1.lerp(p2, t)
        p23: Vector = p2.lerp(p3, t)
        p012: Vector = p01.lerp(p12, t)
        p123: Vector = p12.lerp(p23, t)
        p0123: Vector = p012.lerp(p123, t)
        mid_t = self.t0 + (self.t1 - self.t0) * t
        left = Bezier(p0, p01, p012, p0123, t0=self.t0, t1=mid_t)
        right = Bezier(p0123, p123, p23, p3, t0=mid_t, t1=self.t1)
        return (left, right)

    def split2_o(self, *parameters: float):
        """Split the curve at parameters. Returns a list of the split segments."""
        ts = sorted([t for t in parameters])
        if ts[0] != 0.0:
            ts.insert(0, 0.0)
        if ts[-1] != 1.0:
            ts.append(1.0)
        sub_curves: list[Bezier] = []
        for i in range(1, len(ts)):
            sub_curves.append(self.subsegment(ts[i - 1], ts[i]))
        return sub_curves

    def _create_blender_curve(self):
        """Creates a new curve object in Blender."""
        cu = bpy.data.curves.new(self.name, 'CURVE')
        ob = bpy.data.objects.new(self.name, cu)
        ob.location = self.location
        ob.scale = self.scale
        ob.rotation_euler = self.rotation
        bpy.data.collections['Collection'].objects.link(ob)
        ob.data.resolution_u = 64
        cu.splines.new('BEZIER')
        return cu

    def add_to_blender(self, blender_curve_object=None, stringed=False):
        """Adds the Bezier curve to Blender as splines.
        blender_curve_object: an existing curve in Blender.
        stringed: is a bool which is true in case the Bezier curve is part of a Spline.
        Both the parameters are used to make it possible to reuse this in the case where
        the Bezier curve is part of a Spline.
        """
        p = self.points
        cu = blender_curve_object or self._create_blender_curve()
        spline = cu.splines[-1]
        spline.use_cyclic_u = self.is_closed
        bezier_points = spline.bezier_points
        bezier_points[-1].handle_right = p[1]
        bezier_points.add(1)
        bezier_points[-1].handle_left = p[2]
        bezier_points[-1].co = p[3]
        if not stringed:
            bezier_points[-2].co = p[0]
            bezier_points[-2].handle_left = self.start_handle_left or p[0]
            bezier_points[-1].handle_right = self.end_handle_right or p[3]

    def overlaps(self, bezier: Any):
        """Check if the bounding box of self and Bezier overlaps."""
        bb1 = self.bounding_box(world_space=True)
        bb2 = bezier.bounding_box(world_space=True)
        return not (bb1['min_x'] > bb2['max_x'] or bb2['min_x'] > bb1['max_x'] or bb1['min_y'] > bb2['max_y'] or (bb2['min_y'] > bb1['max_y']))

    def transform(self, angle: float=0.0, translation=mathutils.Vector((0, 0, 0)), world_space=False):
        """Rotates the curve an angle around the z axis and then translates it.
        If world_space, then the curve object is transformed instead of the local coordinates."""
        if world_space:
            print('NOT YET IMPLEMENTED: Will do local tranform instead.')
        m = mathutils.Matrix.Rotation(angle, 3, 'Z')
        p = self.points
        q0: mathutils.Vector = m @ p[0] + translation
        q1: mathutils.Vector = m @ p[1] + translation
        q2: mathutils.Vector = m @ p[2] + translation
        q3: mathutils.Vector = m @ p[3] + translation
        self.points = [q0, q1, q2, q3]

    def intersect_ray(self, sp: mathutils.Vector, d: mathutils.Vector):
        """Find intersection of the cubic with a ray from point sp in the direction d.
        Return the parameter value t where the intersection occurs (if any)."""
        points = self.points
        p0 = points[0]
        p1 = points[1]
        p2 = points[2]
        p3 = points[3]
        c0 = d.x * (p0.y - sp.y) + d.y * (sp.x - p0.x)
        c1 = 3 * d.x * (p1.y - p0.y) + 3 * d.y * (p0.x - p1.x)
        c2 = 3 * d.x * (p0.y - 2 * p1.y + p2.y) - 3 * d.y * (p0.x - 2 * p1.x + p2.x)
        c3 = d.x * (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) + d.y * (p0.x - 3 * p1.x + 3 * p2.x - p3.x)
        qs = solvers.solve_cubic(c3, c2, c1, c0)
        sols: list[float] = [s for s in qs if isinstance(s, float) and 0.0 < s < 1.0]
        return sols

    def find_offset_cusps(self, d: float):
        """Find the parameters values where the curvature is equal to the inverse of the distance d."""
        results = []
        n = 200
        q = self.derivative()
        ya = 0.0
        last_t = 0.0
        t0 = 0.0
        for i in range(0, n + 1):
            t = i / n
            ds = q(t).length()
            k = (q.eval_derivative(t).x * q(t).y - q.eval_derivative(t).y * q(t).x) / ds ** 3
            yb = k * d + 1
            if i != 0:
                if ya * yb < 0:
                    tx = (yb * last_t - ya * t) / (yb - ya)
                    iv = {'t0': t0, 't1': tx, 'sign': math.copysign(1, ya)}
                    results.append(iv)
                    t0 = tx
            ya = yb
            last_t = t
        last_iv = {'t0': t0, 't1': 1.0, 'sign': math.copysign(1, ya)}
        results.append(last_iv)
        return results

    def offset(self, d: float, double_side: bool=False):
        """CHANGED: added docstring and type hints

offset

Args:
    d (float): 
    double_side (bool): 

Returns:
    Any
"""
        cusps = self.find_offset_cusps(d)
        loffsets: list[Bezier] = []
        roffsets: list[Bezier] = []
        for cusp in cusps:
            curve = self.subsegment(cusp['t0'], cusp['t1'])
            off = OffsetBezier(curve, d, cusp['sign'])
            offset = off.find_cubic_approximation()
            if offset is not None:
                for bez in offset:
                    bez.location = self.location
                    bez.scale = self.scale
                    bez.rotation = self.rotation
                loffsets.extend(offset)
        if double_side:
            cusps = self.find_offset_cusps(-d)
            for cusp in cusps:
                curve = self.subsegment(cusp['t0'], cusp['t1'])
                off = OffsetBezier(curve, -d, cusp['sign'])
                offset = off.find_cubic_approximation()
                if offset is not None:
                    for bez in offset:
                        bez.location = self.location
                        bez.scale = self.scale
                        bez.rotation = self.rotation
                roffsets.extend(offset)
        return [loffsets, roffsets]

    def stroke(self, d: float):
        """CHANGED: added docstring and type hints

stroke

Args:
    d (float): 

Returns:
    Any
"""
        offsets = self.offset(d, double_side=True)
        left_offset = Spline(*offsets[0], location=self.location, rotation=self.rotation)
        right_offset = Spline(*offsets[1], location=self.location, rotation=self.rotation)
        left_offset.name = 'Left'
        right_offset.name = 'Right'
        left_offset.add_to_blender()
        right_offset.add_to_blender()

    def curve_intersections(self, c2: 'Bezier', threshold=INTERSECTION_THRESHOLD):
        """Recursive method used for finding the intersection between the two Bezier curves self and c2.
        """
        threshold2 = threshold * threshold
        bez0 = self
        results: list[tuple[float, float]] = []
        if bez0.bounding_box()['area'] < threshold2 and c2.bounding_box()['area'] < threshold2:
            if (bez0.t0 < TUPLE_FILTER_THRESHOLD or bez0.t1 > 1.0 - TUPLE_FILTER_THRESHOLD) and (c2.t0 < TUPLE_FILTER_THRESHOLD or c2.t1 > 1.0 - TUPLE_FILTER_THRESHOLD):
                return []
            else:
                return [(bez0.t0, bez0.t1, c2.t0, c2.t1)]
        cc1 = bez0.split2(0.5)
        cc2 = c2.split2(0.5)
        pairs = itertools.product(cc1, cc2)
        pairs = list(filter(lambda x: x[0].overlaps(x[1]), pairs))
        if len(pairs) == 0:
            return results
        for pair in pairs:
            results += pair[0].curve_intersections(pair[1], threshold)
        results = filter_duplicates(results)
        return results

    def find_self_intersection(self):
        """Finds the self intersection of the curve (if any).
        Returns three parameter values.
        Ref: https://comp.graphics.algorithms.narkive.com/tqLNEZqM/cubic-bezier-self-intersections
        """
        p0 = self.points[0]
        p1 = self.points[1]
        p2 = self.points[2]
        p3 = self.points[3]
        h1 = -3 * p0 + 3 * p1
        h2 = 3 * p0 - 6 * p1 + 3 * p2
        h3 = -p0 + 3 * p1 - 3 * p2 + p3
        if h3 == mathutils.Vector((0, 0, 0)):
            return None
        a = h2.x / h3.x
        b = h1.x / h3.x
        p = h2.y / h3.y
        q = h1.y / h3.y
        if a == p or q == b:
            return None
        k = (q - b) / (a - p)
        r0 = (-k ** 3 - a * k ** 2 - b * k) / 2
        r1 = (3 * k ** 2 + 2 * k * a + 2 * b) / 2
        r2 = -3 * k / 2
        sols = solvers.solve_cubic(1.0, r2, r1, r0)
        if sols:
            solutions: list[float] = []
            for s in sols:
                if isinstance(s, float) and (s >= 0.0 and s <= 1.0):
                    solutions.append(s)
                solutions.sort()
            if len(solutions) == 3:
                return [solutions[0], solutions[2]]
        return None

    def tolerances_for_curve(self, *, k: int=32, coord_min=1e-08, coord_max=0.001, t_min=1e-07, t_max=0.0005):
        """
        Pick robust float32-friendly tolerances based on curve scale.
        Returns: dict with coord_tol, t_tol, det_tol, tangent_cross_tol
        - k: multiplier for eps32; increase if you need looser geometry tests.
        - coord_min/max: clamps for coordinate tolerance (absolute units).
        - t_min/max: clamps for t-space tolerance.
        """

        def norm2(p: Vector):
            """CHANGED: added docstring and type hints

norm2

Args:
    p (Vector): 

Returns:
    Any
"""
            return math.hypot(p[0], p[1])

        def bbox_span(ctrls):
            """CHANGED: added docstring and type hints

bbox_span

Args:
    ctrls (Any): 

Returns:
    Any
"""
            xs = [p[0] for p in ctrls]
            ys = [p[1] for p in ctrls]
            return max(max(xs) - min(xs), max(ys) - min(ys))

        def control_poly_len(ctrls):
            """CHANGED: added docstring and type hints

control_poly_len

Args:
    ctrls (Any): 

Returns:
    Any
"""
            return norm2((ctrls[1][0] - ctrls[0][0], ctrls[1][1] - ctrls[0][1])) + norm2((ctrls[2][0] - ctrls[1][0], ctrls[2][1] - ctrls[1][1])) + norm2((ctrls[3][0] - ctrls[2][0], ctrls[3][1] - ctrls[2][1]))
        eps32 = 1.1920929e-07
        p = self.points
        span = bbox_span(p)
        cplen = control_poly_len(p)
        scale = max(span, cplen, 1.0)
        coord_tol = max(coord_min, min(coord_max, k * eps32 * scale))
        avg_speed = max(cplen, 1e-08)
        t_tol = max(t_min, min(t_max, coord_tol / avg_speed))
        det_tol = 10.0 * eps32 * scale ** 2
        tangent_cross_tol = 10.0 * eps32 * avg_speed ** 2
        return dict(coord_tol=coord_tol, t_tol=t_tol, det_tol=det_tol, tangent_cross_tol=tangent_cross_tol)

    def ints(self, other: 'Bezier'):
        """CHANGED: added docstring and type hints

ints

Args:
    other ('Bezier'): 

Returns:
    Any
"""

        def fatline(c: 'Bezier'):
            """CHANGED: added docstring and type hints

fatline

Args:
    c ('Bezier'): 

Returns:
    Any
"""
            p0, p1, p2, p3 = c.points
            chord = p3 - p0
            normal = chord.perpendicular().normalize()
            d1 = (p1 - p0).dot(normal)
            d2 = (p2 - p0).dot(normal)
            if d1 * d2 > 0:
                dmin = 0.75 * min(0, d1, d2)
                dmax = 0.75 * max(0, d1, d2)
            else:
                dmin = 4 / 9 * min(0, d1, d2)
                dmax = 4 / 9 * max(0, d1, d2)
            return (p0, p3, normal, dmin, dmax)

        def clipping_region(c: 'Bezier', q0: Vector, q3: Vector, normal: Vector, dmin: float, dmax: float):
            """CHANGED: added docstring and type hints

clipping_region

Args:
    c ('Bezier'): 
    q0 (Vector): 
    q3 (Vector): 
    normal (Vector): 
    dmin (float): 
    dmax (float): 

Returns:
    Any
"""
            p0, p1, p2, p3 = c.points
            d0 = (p0 - q0).dot(normal)
            d1 = (p1 - q0).dot(normal)
            d2 = (p2 - q0).dot(normal)
            d3 = (p3 - q0).dot(normal)
            t = []
            if abs(d1 - d0) > 1e-12:
                t01m = 1 / 3 * (dmin - d0) / (d1 - d0)
                t01_m = 1 / 3 * (dmax - d0) / (d1 - d0)
                if 0 <= t01m <= 1 / 3:
                    t.append(t01m)
                if 0 <= t01_m <= 1 / 3:
                    t.append(t01_m)
        p0, p3, normal, dmin, dmax = fatline(self)
        clipping_region(other, p0, p3, normal, dmin, dmax)

class QuadraticBezier:
    """Class to handle some functions of a quadratic Bezier curve.
    Used mainly for handling derivatives, etc, of a cubic Bezier."""
    __slots__ = 'points'

    def __init__(self, p0: Vector=Vector(), p1: Vector=Vector(), p2: Vector=Vector()) -> None:
        """CHANGED: added docstring and type hints

__init__

Args:
    p0 (Vector): 
    p1 (Vector): 
    p2 (Vector): 

Returns:
    None
"""
        self.points = [p0, p1, p2]

    def __call__(self, t: float) -> Vector:
        """CHANGED: added docstring and type hints

__call__

Args:
    t (float): 

Returns:
    Vector
"""
        p = self.points
        return p[0] * (1 - t) ** 2 + 2 * p[1] * (1 - t) * t + p[2] * t ** 2

    def eval_derivative(self, t: float):
        """Evaluates the derivative of the curve at parameter t."""
        p = self.points
        return -2 * p[0] * (1 - t) - 2 * p[1] * t + 2 * p[1] * (1 - t) + 2 * p[2] * t

class OffsetBezier:
    """CHANGED: added class docstring

OffsetBezier

Auto-generated documentation. Please refine details as needed.
"""
    __slots__ = ('d', 'is_linear', 'rot_curve', 'angle', 'translation', 'endpoint', 'angles', 'metrics', 'original_curve', 'sign', 'tolerance')
    'Calculates an cubic approximation of the offset of a cubic Bezier curve.'

    def __init__(self, bez: Bezier, d: float, sign=1, tolerance: float=OFFSET_TOLERANCE) -> None:
        """CHANGED: added docstring and type hints

__init__

Args:
    bez (Bezier): 
    d (float): 
    sign (Any): 
    tolerance (float): 

Returns:
    None
"""
        self.d = d
        self.sign = sign
        self.is_linear = False
        self.original_curve = bez
        self.rot_curve, self.angle, self.translation, self.endpoint = self._rotate_original_to_x()
        if self.is_linear:
            self.angles = []
            self.metrics = dict()
        else:
            self.angles: list[float] = self._calculate_angles()
            self.metrics: dict[str, float] = self._calculate_metrics()
        self.tolerance = tolerance

    def eval_offset(self, t: float):
        """Evaluates the offset curve position at parameter t."""
        cur = self.rot_curve
        dp = cur.eval_derivative(t)
        if dp.length() > 0.0:
            s = self.d / dp.length()
        else:
            s = 0.0
        return Vector(-s * dp[1], s * dp[0], 0)

    def eval_offset_derivative(self, t: float):
        """Evaluates the derivative of the offset at parameter t."""
        cur: Bezier = self.rot_curve
        der = cur.derivative()
        dp = der(t)
        dpp = der.eval_derivative(t)
        k = 1.0 + (dpp.x * dp.y - dpp.y * dp.x) * self.d / dp.length() ** 3
        return k * dp

    def _rotate_original_to_x(self):
        """Calculates a new, transformed (rotated and translated) curve, such that the offset 
        of that will end up starting at (0,0) and ending on the x-axis (at endp).
        Returns the new Bezier, the rotation angle, the translation, and the endpoint..
        """
        d = self.d
        bez = self.original_curve
        p0 = bez.points[0].to_mu_vector()
        p1 = bez.points[1].to_mu_vector()
        p2 = bez.points[2].to_mu_vector()
        p3 = bez.points[3].to_mu_vector()
        l_linear = False
        r_linear = False
        dotp0123 = abs((p0 - p1).x * (p3 - p2).x + (p0 - p1).y * (p3 - p2).y)
        l0123 = math.hypot((p0 - p1).x, (p0 - p1).y) * math.hypot((p3 - p2).x, (p3 - p2).y)
        dotp023 = abs((p0 - p2).x * (p2 - p3).x + (p0 - p2).y * (p2 - p3).y)
        l023 = math.hypot((p0 - p2).x, (p0 - p2).y) * math.hypot((p2 - p3).x, (p2 - p3).y)
        dotp013 = abs((p0 - p1).x * (p1 - p3).x + (p0 - p1).y * (p1 - p3).y)
        l013 = math.hypot((p0 - p1).x, (p0 - p1).y) * math.hypot((p1 - p3).x, (p1 - p3).y)
        if p1 == p0 and p2 == p3 or dotp0123 == l0123:
            self.is_linear = True
        if p0 == p1:
            l_linear = True
            if self.is_linear:
                p1 = bez(0.25).to_mu_vector()
            else:
                p1 = bez(0.0001).to_mu_vector()
        if p2 == p3:
            r_linear = True
            if self.is_linear:
                p2 = bez(0.75).to_mu_vector()
            else:
                p2 = bez(0.9999).to_mu_vector()
        if l_linear:
            h0 = (p1 - p0) / (p1 - p0).length
            r0 = p0 + d * mathutils.Vector((-h0.y, h0.x, 0))
        else:
            r0 = p0 + d * bez.normal(0.0)
        if r_linear:
            h1 = (p3 - p2) / (p3 - p2).length
            r1 = p3 + d * mathutils.Vector((-h1.y, h1.x, 0))
        else:
            r1 = p3 + d * bez.normal(1.0)
        th = math.atan2((r1 - r0)[1], (r1 - r0)[0])
        rot = mathutils.Matrix.Rotation(-th, 3, 'Z')
        q0 = rot @ (p0 - r0)
        q1 = rot @ (p1 - r0) if l_linear else rot @ (bez.points[1].to_mu_vector() - r0)
        q2 = rot @ (p2 - r0) if r_linear else rot @ (bez.points[2].to_mu_vector() - r0)
        q3 = rot @ (p3 - r0)
        b = Bezier(Vector.from_mu_vector(q0), Vector.from_mu_vector(q1), Vector.from_mu_vector(q2), Vector.from_mu_vector(q3))
        endp = d * b.normal(1.0) + b(1.0).to_mu_vector()
        return (b, th, r0, endp)

    def _calculate_angles(self):
        """Calculate the angles of the handles that the offset should have."""
        rp = self.rot_curve.points
        q0 = rp[1] - rp[0]
        q1 = rp[2] - rp[3]
        theta0 = math.atan2(q0.y, q0.x) - math.pi * (1 - self.sign) / 2
        theta1 = math.atan2(q1.y, q1.x) - math.pi * (1 - self.sign) / 2
        return [theta0, theta1]

    def _sample_points(self, n: int):
        """Sample the offset curve at n points. 
        Returns a list with dicts containing:
            arclen: the arclength up to the point
            p: the coordinate of the sampled point of the offset curve
            d: the offset vector (from the rotated curve to the point p
        """
        samples = []
        arclen = 0.0
        co = GAUSS_LEGENDRE_COEFFS_32
        dt = 1 / (n + 1)
        for i in range(0, n):
            for j in range(0, len(co), 2):
                t = dt * (i + 0.5 + 0.5 * co[j + 1])
                arclen += co[j] * self.eval_offset_derivative(t).length()
            t = dt * (i + 1)
            delta = self.eval_offset(t)
            point = self.rot_curve(t) + delta
            samples.append({'arclen': arclen * 0.5 * dt, 'p': point, 'd': delta})
        return samples

    def _estimate_cubic_error(self, cu: Bezier, samples):
        """CHANGED: added docstring and type hints

_estimate_cubic_error

Args:
    cu (Bezier): 
    samples (Any): 

Returns:
    Any
"""
        err = 0.0
        tol2 = self.tolerance ** 2
        for sample in samples:
            best_err: float = None
            samps = cu.intersect_ray(sample['p'], sample['d'])
            if len(samps) == 0:
                samps = [0.0, 1.0]
            for t in samps:
                p_proj = cu(t)
                this_err = (sample['p'] - p_proj).length() ** 2
                if best_err is None or this_err < best_err:
                    best_err = this_err
            err = max(err, best_err)
            if err > tol2:
                break
        return math.sqrt(err)

    def _find_best_approximation(self):
        """CHANGED: added docstring and type hints

_find_best_approximation

Args:
    (none)

Returns:
    Any
"""
        if not self.is_linear:
            candidates = self._find_cubic_candidates()
            samples = self._sample_points(10)
            best_curve: Bezier = None
            best_err: float = None
            errs: list[float] = []
            for i, cand in enumerate(candidates):
                err = self._estimate_cubic_error(cand, samples)
                errs.append(err)
                if best_curve is None or err < best_err:
                    best_err = err
                    best_curve = cand
        else:
            p = self.rot_curve.points
            der = (p[3] - p[0]) / (p[3] - p[0]).length
            disp = self.d * mathutils.Vector((-der.y, der.x, 0))
            p0 = p[0] + disp
            p1 = p[1] + disp
            p2 = p[2] + disp
            p3 = p[3] + disp
            best_curve = Bezier(p0, p1, p2, p3)
            best_err = 0.0
        if best_curve:
            best_curve.transform(self.angle, self.translation)
        return {'curve': best_curve, 'error': best_err}

    def find_cubic_approximation(self) -> list[Bezier]:
        """CHANGED: added docstring and type hints

find_cubic_approximation

Args:
    (none)

Returns:
    list[Bezier]
"""
        tolerance = self.tolerance
        approx = self._find_best_approximation()
        if approx['curve'] and approx['error'] <= tolerance:
            approx['curve'].name = self.original_curve.name
            return [approx['curve']]
        else:
            b = self.original_curve.split2(0.5)
            k1 = OffsetBezier(b[0], self.d, self.sign)
            k2 = OffsetBezier(b[1], self.d, self.sign)
            kk1 = k1.find_cubic_approximation()
            kk2 = k2.find_cubic_approximation()
            kk1.extend(kk2)
            return kk1

    def _find_cubic_candidates(self):
        """Solve the quartic equation for delta0 and delta1 that give the offset which closest 
        matches the required metrics."""
        mx: float = self.metrics['x_moment']
        a: float = self.metrics['area']
        x3: float = self.endpoint.x
        th0, th1 = self._calculate_angles()
        c0 = math.cos(th0)
        s0 = math.sin(th0)
        c1 = math.cos(th1)
        s1 = math.sin(th1)
        k0 = 50 * a ** 2 * c1 * s1 * x3 / 21 + 50 * a * s1 ** 2 * x3 ** 3 / 21 - 4 * mx * s1 ** 2 * x3 ** 2
        k1 = -10 * a ** 2 * c0 * c1 * s1 / 7 + 10 * a ** 2 * c1 ** 2 * s0 / 7 - 34 * a * c0 * s1 ** 2 * x3 ** 2 / 21 + 4 * a * c1 * s0 * s1 * x3 ** 2 / 3 + 4 * c0 * mx * s1 ** 2 * x3 - 4 * c1 * mx * s0 * s1 * x3 - 8 * s0 * s1 ** 2 * x3 ** 4 / 35
        k2 = -3 * a * c0 ** 2 * s1 ** 2 * x3 / 14 + 2 * a * c0 * c1 * s0 * s1 * x3 / 7 - a * c1 ** 2 * s0 ** 2 * x3 / 14 - c0 ** 2 * mx * s1 ** 2 + 2 * c0 * c1 * mx * s0 * s1 + 3 * c0 * s0 * s1 ** 2 * x3 ** 3 / 14 - c1 ** 2 * mx * s0 ** 2 - 9 * c1 * s0 ** 2 * s1 * x3 ** 3 / 70
        k3 = 3 * a * c0 ** 3 * s1 ** 2 / 14 - 3 * a * c0 ** 2 * c1 * s0 * s1 / 7 + 3 * a * c0 * c1 ** 2 * s0 ** 2 / 14 - c0 ** 2 * s0 * s1 ** 2 * x3 ** 2 / 35 + c0 * c1 * s0 ** 2 * s1 * x3 ** 2 / 70 + c1 ** 2 * s0 ** 3 * x3 ** 2 / 70
        k4 = -3 * c0 ** 3 * s0 * s1 ** 2 * x3 / 280 + 3 * c0 ** 2 * c1 * s0 ** 2 * s1 * x3 / 140 - 3 * c0 * c1 ** 2 * s0 ** 3 * x3 / 280
        sols = solvers.solve_quartic(k4, k3, k2, k1, k0)
        good_sols = [sol for sol in sols if isinstance(sol, float)]
        good_sols.extend([sol.real for sol in sols if isinstance(sol, complex)])
        n = 0
        cubics: list[Bezier] = []
        for d0 in good_sols:
            fac = -c0 * d0 * s1 + c1 * d0 * s0 + 2 * s1 * x3
            if fac != 0.0:
                d1 = (20 * a / 3 - 2 * d0 * s0 * x3) / (-c0 * d0 * s1 + c1 * d0 * s0 + 2 * s1 * x3)
            else:
                d1 = 0.0
            if d0 < 0.0:
                d0 = 0
                d1 = s0 * x3 / (c0 * s1 - c1 * s0)
            elif d1 < 0.0:
                d1 = 0
                d0 = s1 * x3 / (c0 * s1 - c1 * s0)
            if d0 >= 0.0 and d1 >= 0.0:
                p0 = mathutils.Vector((0, 0, 0))
                p1 = mathutils.Vector((d0 * c0, d0 * s0, 0))
                p3 = mathutils.Vector((x3, 0, 0))
                p2 = p3 + mathutils.Vector((d1 * c1, d1 * s1, 0))
                b = Bezier(p0, p1, p2, p3)
                b.name = 'Bezier' + str(n)
                n += 1
                cubics.append(b)
        return cubics

    def _calculate_metrics(self):
        """Calculates the x-moment, area, and arc length that the offset should have."""
        arclen = 0.0
        area = 0.0
        x_moment = 0.0
        co = GAUSS_LEGENDRE_COEFFS_32
        for i in range(0, len(co), 2):
            t = 0.5 * (1 + co[i + 1])
            wi = co[i]
            dp = self.eval_offset_derivative(t)
            p = Vector.from_mu_vector(self.eval_offset(t)) + self.rot_curve(t)
            d_area = wi * dp[0] * p[1]
            arclen += wi * dp.length()
            area += d_area
            x_moment += p.x * d_area
        return {'area': 0.5 * area, 'length': 0.5 * arclen, 'x_moment': 0.5 * x_moment}
