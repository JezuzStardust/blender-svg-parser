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
from .bezier import Bezier
from .curve import Curve
from .matrix import Matrix
from .vector import Vector

class Spline(CurveObject):
    """A list of Bezier curves corresponds to a single spline object.
    For each Bezier, the end point coincide with the starting point of 
    the next curve."""
    __slots__ = ('beziers', 'is_closed', 'strokewidth')

    def __init__(self, *beziers: Bezier, is_closed: bool=False, strokewidth=0.01, name='Spline', location=Vector(), scale: float=Vector(1.0, 1.0, 1.0), rotation=Vector()) -> None:
        """CHANGED: added docstring and type hints

__init__

Args:
    is_closed (bool): 
    strokewidth (Any): 
    name (Any): 
    location (Any): 
    scale (float): 
    rotation (Any): 

Returns:
    None
"""
        self.beziers = list(beziers)
        prev_bez = None
        for bez in self.beziers:
            if not prev_bez:
                prev_bez = bez
                continue
            bez.points[0] = prev_bez.points[3]
            bez.start_handle_left = prev_bez.points[2]
            prev_bez = bez
        self.is_closed = is_closed
        self.strokewidth = strokewidth
        for bezier in self.beziers:
            bezier.t0 = 0.0
            bezier.t1 = 1.0
        super().__init__(name, location, scale, rotation)

    @classmethod
    def from_blender(cls, name: str) -> Any:
        """Alternative constructor where the Spline is imported from Blender."""
        cu = bpy.data.collections['Collection'].objects[name]
        beziers = []
        loc = cu.location
        sca = cu.scale
        rot = cu.rotation_euler
        spline = cu.data.splines[0]
        is_closed = spline.use_cyclic_u
        bezier_points = spline.bezier_points
        i = len(spline.bezier_points) - 1
        for j in range(0, i):
            handle_left = bezier_points[j].handle_left
            p0 = bezier_points[j].co
            p1 = bezier_points[j].handle_right
            p2 = bezier_points[j + 1].handle_left
            p3 = bezier_points[j + 1].co
            handle_right = bezier_points[j + 1].handle_right
            beziers.append(Bezier(p0, p1, p2, p3, location=loc, scale=sca, rotation=rot, start_handle_left=handle_left, end_handle_right=handle_right))
        return cls(*beziers, name=name, location=loc, scale=sca, rotation=rot, is_closed=is_closed)

    @CurveObject.location.setter
    def location(self, location: Vector):
        """Set the location in world space of the Spline and all the Bezier curves."""
        for bez in self.beziers:
            bez.location = location
        self._location = location

    @CurveObject.scale.setter
    def scale(self, scale: Vector):
        """Sets the scale of the Spline and propagate it to all bezier curves."""
        for bez in self.beziers:
            bez.scale = scale
        self._scale = scale

    @CurveObject.rotation.setter
    def rotation(self, rotation: Vector):
        """CHANGED: added docstring and type hints

rotation

Args:
    rotation (Vector): 

Returns:
    Any
"""
        for bez in self.beziers:
            bez.rotation = rotation
        self._rotation = rotation

    def reverse(self):
        """CHANGED: added docstring and type hints

reverse

Args:
    (none)

Returns:
    Any
"""
        for bez in self.beziers:
            bez.reverse()
        self.beziers = list(reversed(self.beziers))

    def append_spline(self, spline: Any):
        """Add curve to the this curve at the end.
        The start point and handles of spline will be
        moved to match with self's endpoint
        """
        a = len(self.beziers)
        last_bez = self.beziers[-1]
        loc = self.location
        sca = self.scale
        rot = self.rotation
        s_loc = spline.location
        for bez in spline.beziers:
            p0 = bez.points[0] + s_loc - loc
            p1 = bez.points[1] + s_loc - loc
            p2 = bez.points[2] + s_loc - loc
            p3 = bez.points[3] + s_loc - loc
            shl = bez.start_handle_left
            ehr = bez.end_handle_right
            self.beziers.append(Bezier(p0, p1, p2, p3, start_handle_left=shl, end_handle_right=ehr, location=loc, scale=sca, rotation=rot))
        abez = self.beziers[a]
        abez.points[0] = last_bez.points[3]
        abez.points[1] = last_bez.end_handle_right
        abez.start_handle_left = last_bez.points[2]

    def prepend_spline(self, spline: Any):
        """
        Add curve to the this curve at the beginning.
        The end point of spline will be moved to coincide with the start point of self.
        """
        a = len(spline.beziers)
        first_bez = self.beziers[0]
        loc = self.location
        s_loc = spline.location
        for bez in reversed(spline.beziers):
            p0 = bez.points[0] + s_loc - loc
            p1 = bez.points[1] + s_loc - loc
            p2 = bez.points[2] + s_loc - loc
            p3 = bez.points[3] + s_loc - loc
            shl = bez.start_handle_left
            ehr = bez.end_handle_right
            self.beziers.insert(0, Bezier(p0, p1, p2, p3, start_handle_left=shl, end_handle_right=ehr, location=loc))
        abez = self.beziers[a - 1]
        abez.points[3] = first_bez.points[0]
        abez.points[2] = first_bez.start_handle_left
        abez.end_handle_right = first_bez.points[1]

    def append_bezier(self, bezier: 'Bezier'):
        """
        Add a single Bezier curve in the end of the curve. 
        End and start points must match. 
        """
        if self.end_point(world_space=True) == bezier.start_point(world_space=True):
            bezier.translate_origin(self.location)
            ep = self.end_point()
            bezier.points[0] = ep
            self.beziers.append(bezier)
        else:
            raise Exception('Start and end points of curves must be equal.')

    def prepend_bezier(self, bezier: Any):
        """Add a single Bezier curve to the start of the curve. 
        End point of bezier must match with start point of self.
        """
        if self.start_point(world_space=True) == bezier.end_point(world_space=True):
            bezier.translate_origin(self.location)
            sp = self.start_point()
            bezier.set_point(sp, -1)
            self.beziers.insert(0, bezier)
        else:
            raise Exception('Start and end points of curves must be equal.')

    def toggle_closed(self):
        """Toggles the curve closed.
        """
        self.is_closed = not self.is_closed

    def _create_blender_curve(self):
        """CHANGED: added docstring and type hints

_create_blender_curve

Args:
    (none)

Returns:
    Any
"""
        cu = bpy.data.curves.new(self.name, 'CURVE')
        ob = bpy.data.objects.new(self.name, cu)
        ob.location = self.location
        ob.scale = self.scale
        ob.rotation_euler = self.rotation
        bpy.data.collections['Collection'].objects.link(ob)
        ob.data.resolution_u = 64
        cu.splines.new('BEZIER')
        return cu

    def add_to_blender(self, blender_curve_object=None):
        """Adds the curve to Blender as splines. 
        """
        if blender_curve_object is None:
            cu = self._create_blender_curve()
        else:
            cu = blender_curve_object
        spline = cu.splines[-1]
        spline.use_cyclic_u = self.is_closed
        bezier_points = spline.bezier_points
        sp = self.start_point(world_space=False)
        sh = self.beziers[0].start_handle_left
        if sh:
            bezier_points[0].handle_left = sh
        else:
            bezier_points[0].handle_left = sp
        bezier_points[0].co = sp
        for bez in self.beziers:
            bez.add_to_blender(cu, stringed=True)
        eh = self.beziers[-1].end_handle_right
        if eh:
            bezier_points[-1].handle_right = eh
        else:
            bezier_points[-1].handle_right = self.end_point(world_space=False)

    def self_intersections(self, threshold=INTERSECTION_THRESHOLD):
        """Find the intersections within the spline.
        The results is a dict with two different key types. 
        1. When one of the Bezier in the Spline contains a self intersection: 
        key: int = i
        value: float = t
        In this case self.beziers[i] intersects itself at parameter t.
        2. When two different curves intersect: 
        key: (int, int) = (i, j)
        value: [(ta1, tb1), (ta2, tb2), ...]
        In this case self.bezier[i](ta1) = self.bezier[j](tb1) and similar for (ta2, tb2), etc.
        """
        intersections = {}
        for i in range(len(self.beziers)):
            ints = self.beziers[i].find_self_intersection()
            if ints:
                intersections[i] = ints
        pairs = itertools.combinations(enumerate(self.beziers), 2)
        pairs = iter((pair for pair in pairs if pair[0][1].overlaps(pair[1][1])))
        for pair in pairs:
            results = pair[0][1].curve_intersections(pair[1][1], threshold)
            if results:
                intersections[pair[0][0], pair[1][0]] = results
        return intersections

    def intersections(self, other: 'Spline', threshold=INTERSECTION_THRESHOLD):
        """This should perhaps only intersect the two curves. 
        We already have self intersections via the other function."""
        'Self intersections: dict: \n        - Key (i), value: (tia, tib): the i:th Bezier intersects itself at tia and tib. \n        - Key (i, j), value: (tia, tib, tja, tjb): the i:th Bezier intersects the j:th Bezier at (tia approx tib)\n        and (tja approx tjb)\n\n        Intersectoins: dict:\n        - Key (i, j), value (tia, tib, tja, tjb): the i:th Bezier of self intersect the j:th Bezier of other.\n        '
        intersections = {}
        self_enumerate = enumerate(self.beziers)
        other_enumerate = enumerate(other.beziers)
        pairs = itertools.product(self_enumerate, other_enumerate)
        pairs = iter((pair for pair in pairs if pair[0][1].overlaps(pair[1][1])))
        for pair in pairs:
            result = pair[0][1].curve_intersections(pair[1][1], threshold)
            if result:
                intersections[pair[0][0], pair[1][0]] = result
        return intersections

    def start_point(self, world_space=False):
        """CHANGED: added docstring and type hints

start_point

Args:
    world_space (Any): 

Returns:
    Any
"""
        return self.beziers[0](0, world_space=world_space)

    def end_point(self, world_space=False):
        """CHANGED: added docstring and type hints

end_point

Args:
    world_space (Any): 

Returns:
    Any
"""
        return self.beziers[-1](1, world_space=world_space)

    def split(self, i: int, t: float):
        """Splits the Spline at parameter t of curve i."""
        first = self.beziers[0:i]
        second = self.beziers[i + 1:]
        splits = self.beziers[i].split2(t)
        first.append(splits[0])
        second.insert(0, splits[1])
        loc = self.location
        rot = self.rotation
        return [Spline(*first, location=loc, rotation=rot), Spline(*second, location=loc, rotation=rot)]

    def split_at_self_intersections(self):
        """CHANGED: added docstring and type hints

split_at_self_intersections

Args:
    (none)

Returns:
    Any
"""
        ints = self.self_intersections()
        compiled = {}
        for key in ints:
            if isinstance(key, tuple):
                if key[0] in compiled:
                    compiled[key[0]].extend(list((i[0] for i in ints[key])))
                else:
                    compiled[key[0]] = list((i[0] for i in ints[key]))
                if key[1] in compiled:
                    compiled[key[1]].extend(list((i[1] for i in ints[key])))
                else:
                    compiled[key[1]] = list((i[1] for i in ints[key]))
            elif key in compiled:
                compiled[key].extend(ints[key])
            else:
                compiled[key] = ints[key]
        bezs = self.beziers
        curves: list[Bezier] = []
        print(5 * '\n', compiled)
        keep: int = 1
        print('Splits', compiled)
        for i, bez in enumerate(bezs):
            print('Curve ', i)
            if i in compiled:
                print('Splitting')
                splits = bez.split2(*compiled[i])
                print(len(splits), 'pieces.')
                for s in splits[0:-1]:
                    if keep > 0:
                        print('Keeping', i)
                        curves.append(s)
                    else:
                        print('Discarding split.')
                    keep *= -1
                if keep > 0:
                    print('Keeping last curve in split.')
                    curves.append(splits[-1])
                else:
                    print('Discarding last curve in split.')
            elif keep > 0:
                print('Keeping unsplitted.')
                curves.append(bez)
            else:
                print('Discarding unsplitted.')
        return Spline(*curves, location=self.location, rotation=self.rotation)

    def join_spline(self, mode):
        """Join another spline to the end of self. 
        mode determines how the joins are made in case self does not have continuous derivative."""