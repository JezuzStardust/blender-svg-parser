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
from .spline import Spline
from .vector import Vector

def add_line(a: mathutils.Vector, b: mathutils.Vector):
    """Add a line between a and b in Blender."""
    me = bpy.data.meshes.new('Line')
    verts = [a, b]
    edges = [(0, 1)]
    faces = []
    me.from_pydata(verts, edges, faces)
    ob = bpy.data.objects.new('Line', me)
    bpy.data.collections['Collection'].objects.link(ob)

def add_bbox(curve: 'Bezier', xmin: float, xmax: float, ymin: float, ymax: float):
    """CHANGED: added docstring and type hints

add_bbox

Args:
    curve ('Bezier'): 
    xmin (float): 
    xmax (float): 
    ymin (float): 
    ymax (float): 

Returns:
    Any
"""
    o = curve.location.to_mu_vector()
    a = o + mathutils.Vector((xmin, ymin, 0))
    b = o + mathutils.Vector((xmin, ymax, 0))
    c = o + mathutils.Vector((xmax, ymax, 0))
    d = o + mathutils.Vector((xmax, ymin, 0))
    add_line(a, b)
    add_line(b, c)
    add_line(c, d)
    add_line(d, a)

def add_square(p: 'Vector', r: float=0.1):
    """Adds a square to Blender at position p with side r."""
    me = bpy.data.meshes.new('Square')
    x = Vector(1, 0, 0)
    y = Vector(0, 1, 0)
    verts = [p + r * (x + y) / 2, p + r * (x - y) / 2, p - r * (x + y) / 2, p - r * (x - y) / 2]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
    faces = []
    me.from_pydata(verts, edges, faces)
    ob = bpy.data.objects.new(me.name, me)
    bpy.data.collections['Collection'].objects.link(ob)

def filter_duplicates(tuples, threshold=TUPLE_FILTER_THRESHOLD):
    """Filter out tuples that differ less than threshold."""
    result = []
    for tup in tuples:
        if not any((tuple_is_close(tup, other, threshold) for other in result)):
            result.append(tup)
    return result

def tuple_is_close(a: tuple[float, float], b: tuple[float, float], threshold=TUPLE_FILTER_THRESHOLD):
    """Checks if two tuples a, and b, differ less then threshold. 
    (a, b) is close to (a', b') if (a - a') < threshold and abs(b - b') < threshold."""
    comparisons = all((math.isclose(*c, abs_tol=threshold) for c in zip(a, b)))
    return comparisons

def ray_intersect(p0: mathutils.Vector, d0: mathutils.Vector, p1: mathutils.Vector, d1: mathutils.Vector):
    """CHANGED: added docstring and type hints

ray_intersect

Args:
    p0 (mathutils.Vector): 
    d0 (mathutils.Vector): 
    p1 (mathutils.Vector): 
    d1 (mathutils.Vector): 

Returns:
    Any
"""
    det = d0.x * d1.y - d0.y * d1.x
    t = (d0.x * (p0.y - p1.y) - d0.y * (p0.x - p1.x)) / det
    return mathutils.Vector((p1.x + d1.x * t, p1.y + d1.y * t))