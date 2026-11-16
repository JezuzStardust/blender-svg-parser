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
from .matrix import Matrix
from .spline import Spline
from .vector import Vector

class Curve(CurveObject):
    """Curve object, container class for splines. Mirrors the Curve Object in Blender."""
    __slots__ = 'splines'

    def __init__(self, *splines: Spline, name='Curve', location: Vector=Vector(), scale: Vector=Vector(1.0, 1.0, 1.0), rotation: Vector=Vector()) -> None:
        """CHANGED: added docstring and type hints

__init__

Args:
    name (Any): 
    location (Vector): 
    scale (Vector): 
    rotation (Vector): 

Returns:
    None
"""
        self.splines = list(splines)
        super().__init__(name=name, location=location, scale=scale, rotation=rotation)

    @classmethod
    def from_blender(cls, name: str) -> Any:
        """CHANGED: added docstring and type hints

from_blender

Args:
    name (str): 

Returns:
    Any
"""
        splines = []
        cu = bpy.data.collections['Collection'].objects[name]
        loc = cu.location
        sca = cu.scale
        rot = cu.rotation_euler
        name = cu.name
        for spline in cu.data.splines:
            beziers = []
            is_closed = spline.use_cyclic_u
            spline = cu.data.splines[0]
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
            splines.append(Spline(*beziers, is_closed=is_closed, name=name + '_Spline', location=loc, scale=sca, rotation=rot))
        return cls(*splines, name=name, location=loc, scale=sca, rotation=rot)

    @CurveObject.location.setter
    def location(self, location: Vector) -> None:
        """CHANGED: added docstring and type hints

location

Args:
    location (Vector): 

Returns:
    None
"""
        for spline in self.splines:
            spline.location = location

    def add_spline(self, spline: Any):
        """Add a new spline to the curve object.
        Since each spline defines a separate poly-Bezier,
        there is no need to check for matching start and end points."""
        self.splines.append(spline)

    def combine_curves(self, curve: Any):
        """Combine two curves into one."""
        pass

    def offset(self, d):
        """CHANGED: added docstring and type hints

offset

Args:
    d (Any): 

Returns:
    Any
"""
        for spline in self.splines:
            spline.offset_spline(d)

    def intersections(self):
        """CHANGED: added docstring and type hints

intersections

Args:
    (none)

Returns:
    Any
"""
        pass

    def stroke(self):
        """CHANGED: added docstring and type hints

stroke

Args:
    (none)

Returns:
    Any
"""
        pass

    def add_to_blender(self):
        """CHANGED: added docstring and type hints

add_to_blender

Args:
    (none)

Returns:
    Any
"""
        cu = bpy.data.collections['Collection'].objects[self.name]
        for spline in self.splines:
            spline.add_to_blender(blender_curve_object=cu)

class CurveObject:
    """Base class for all curves."""
    __slots__ = ('name', '_location', '_scale', '_rotation')

    def __init__(self, name='Curve Object', location=Vector(), scale: float=Vector(1.0, 1.0, 1.0), rotation=Vector()) -> None:
        """CHANGED: added docstring and type hints

__init__

Args:
    name (Any): 
    location (Any): 
    scale (float): 
    rotation (Any): 

Returns:
    None
"""
        self.name: str = name
        self._location: Vector = location
        self._scale: Vector = scale
        self._rotation: Vector = rotation

    @property
    def location(self):
        """CHANGED: added docstring and type hints

location

Args:
    (none)

Returns:
    Any
"""
        return self._location

    @location.setter
    def location(self, location: Vector) -> None:
        """CHANGED: added docstring and type hints

location

Args:
    location (Vector): 

Returns:
    None
"""
        self._location = location

    @property
    def scale(self) -> Vector:
        """CHANGED: added docstring and type hints

scale

Args:
    (none)

Returns:
    Vector
"""
        return self._scale

    @scale.setter
    def scale(self, scale: Vector) -> None:
        """CHANGED: added docstring and type hints

scale

Args:
    scale (Vector): 

Returns:
    None
"""
        self._scale = scale

    @property
    def rotation(self) -> Vector:
        """CHANGED: added docstring and type hints

rotation

Args:
    (none)

Returns:
    Vector
"""
        return self._rotation

    @rotation.setter
    def rotation(self, rotation: Vector) -> None:
        """CHANGED: added docstring and type hints

rotation

Args:
    rotation (Vector): 

Returns:
    None
"""
        self._rotation = rotation