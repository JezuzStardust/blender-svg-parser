# Copyright 2023-2025 Jens Zamanian

# DISCLAIMER: 
# A big part of this file is based on the ideas found in the blogpost: 
# https://raphlinus.github.io/curves/2022/09/09/parallel-beziers.html
# and parts of the code is adapted from the in the interactive
# demo on that page (source can be found at: raphlinus/raphlinus.github.io). 
# However, all the code have() been rewritten in Python and almost
# all of it is modified so any bugs/errors are my fault (JZ).
# Hopefully it is correct to consider this a derived work and
# hence retain the Apache 2.0 licence for this file.
# END DISCLAIMER

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classes and utility functions for Bezier curves."""

# Bezier: Contains all data and operations needed to define and work with a Bezier curve.
# Spline: Contains a list of Bezier curves.
# 1. How should we init this class? Either we init by passing all the points and the class creates and stores the Bezier instances, or we can init by passing pre-fabricated Bezier instances.
# 2. Can we programme for both options? Either with wargs or kwargs.
# Curves
# Again we need to think about how to init these. 

##### CONSTANTS #####
INTERSECTION_THRESHOLD = 1e-6 # Threshold for when to stop subdividing when finding intersections.
TUPLE_FILTER_THRESHOLD = .2e-1 # Threshold for when two intersections are assumed to be the same. 
OFFSET_TOLERANCE = 1e-4
##### END: CONSTANTS #####

import mathutils
import math
import bpy
import itertools
from typing import Iterator, Iterable, overload
from collections.abc import Sequence

from . import solvers
from .gauss_legendre import GAUSS_LEGENDRE_COEFFS_32

##### UTILITY FUNCTIONS #####
def add_line(a: mathutils.Vector, b: mathutils.Vector):
    """Add a line between a and b in Blender."""
    me = bpy.data.meshes.new('Line')
    verts = [a, b]
    edges = [(0,1)]
    faces = []
    me.from_pydata(verts, edges, faces)
    ob = bpy.data.objects.new('Line', me)
    bpy.data.collections['Collection'].objects.link(ob)

def add_bbox(curve: "Bezier", xmin: float, xmax: float, ymin: float, ymax: float):
    o = curve.location.to_mu_vector()
    a = o + mathutils.Vector((xmin, ymin, 0))
    b = o + mathutils.Vector((xmin, ymax, 0))
    c = o + mathutils.Vector((xmax, ymax, 0))
    d = o + mathutils.Vector((xmax, ymin, 0))
    add_line(a, b)
    add_line(b, c)
    add_line(c, d)
    add_line(d, a)

def add_square(p: "Vector", r: float = 0.1):
    """Adds a square to Blender at position p with side r."""
    me = bpy.data.meshes.new('Square')
    x = Vector(1,0,0)
    y = Vector(0,1,0)
    verts = [p + r * (x + y) / 2, p + r * (x - y) / 2, p - r * (x + y) / 2, p - r * (x - y) / 2]
    edges = [(0,1), (1,2), (2,3), (3,0), (0,2), (1,3)]
    faces = []
    me.from_pydata(verts, edges, faces)
    ob = bpy.data.objects.new(me.name, me)
    bpy.data.collections['Collection'].objects.link(ob)

def filter_duplicates(tuples, threshold = TUPLE_FILTER_THRESHOLD):
    """Filter out tuples that differ less than threshold."""
    result = []
    for tup in tuples:
        if not any(tuple_is_close(tup, other, threshold) for other in result):
            result.append(tup)

    # excluded = [(0, 0), (1, 0), (0, 1), (1, 1)]
    # final = []
    # for tup in result: 
        # if not any(tuple_is_close(tup, other, threshold) for other in excluded):
            # final.append(tup)
    return result

def tuple_is_close(a: tuple[float, float], b: tuple[float, float], threshold = TUPLE_FILTER_THRESHOLD):
    """Checks if two tuples a, and b, differ less then threshold. 
    (a, b) is close to (a', b') if (a - a') < threshold and abs(b - b') < threshold."""
    comparisons = all(math.isclose(*c, abs_tol = threshold) for c in zip(a,b))
    return comparisons

##### END: UTILITY FUNCTIONS #####

class Vector(Sequence[float]): # Inheritance only for type checking.
    __slots__ = ("x", "y", "z")

    _index_to_attr = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        i = None

    @classmethod
    def from_mu_vector(cls, v: mathutils.Vector):
        return cls(v[0], v[1], v[2])


    def __repr__(self) -> str:
        # return f"Vector(x = {self.x:.6g}, y = {self.y:.6g}, z = {self.z:.6g})" 
        return f"Vector(x = {self.x}, y = {self.y}, z = {self.z})"

    def __len__(self) -> int: return 3

    def __iter__(self) -> Iterator[float]:
        for name in self._index_to_attr:
            yield getattr(self, name)

    # For type-checking, two versions of __getitem__ are noted.
    @overload
    def __getitem__(self, idx: int) -> float: ...
    @overload
    def __getitem__(self, idx: slice) -> tuple[float, ...]: ...
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            rng = range(*idx.indices(3))
            return tuple(getattr(self, self._index_to_attr[i]) for i in rng)
        if not isinstance(idx, int):
            raise TypeError("Indices must be int or slice.")
        if not -3 <= idx < 3:
            raise IndexError("Vector index out of range (0..2)")
        return getattr(self, self._index_to_attr[idx])

    def to_mu_vector(self) -> mathutils.Vector:
        return mathutils.Vector((self.x, self.y, self.z))

    @overload
    def __setitem__(self, idx: int, value: float) -> None: ...
    @overload
    def __setitem__(self, idx: slice, value: Iterable[float]) -> None: ...
    def __setitem__(self, idx, value) -> None:
        if isinstance(idx, slice):
            vals = list(map(float, value))
            rng = list(range(*idx.indices(3)))
            if len(vals) != len(rng):
                raise ValueError("Slice assignment missmatch.")
            for i, v in zip(rng, vals):
                setattr(self, self._index_to_attr[i], v)
            return
        if not isinstance(idx, int):
            raise TypeError("Indices must be int or slice.")
        if not -3 <= idx < 3:
            raise IndexError("Vector index out of range.")
        setattr(self, self._index_to_attr[idx], float(value))

    def length(self) -> float:
        return math.sqrt(sum(c**2 for c in self))

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, v: float) -> "Vector":
        x = v * self.x
        y = v * self.y
        z = v * self.z
        return Vector(x, y, z)

    __rmul__ = __mul__

    def __truediv__(self, v: float) -> "Vector":
        return Vector(self.x / v, self.y / v, self.z / v)
    
    def __neg__(self) -> "Vector":
        return Vector(-self.x, -self.y, -self.z)

    def normalize(self) -> "Vector":
           l = self.length()
           if l == 0:
               return Vector(0, 0)
           return Vector(self.x / l, self.y / l)

    def dot(self, other: "Vector") -> float:
        """Dot product."""
        return sum(p * q for p, q in zip(self, other))

    def cross(self, other: "Vector") -> "Vector":
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return Vector(x, y, z)

    def perpendicular(self) -> "Vector":
        """Return a vector perpendicular to this one (90° counter-clockwise)."""
        return Vector(-self.y, self.x, self.z)

    def lerp(self, other: "Vector", t: float) -> "Vector":
            """Linear interpolation between self and other at parameter t."""
            return self * (1 - t) + other * t


class Matrix():
    __slots__ = ("m00", "m01", "m02"
                 "m10", "m11", "m12",
                 "m20", "m21", "m22")

    _names = (
        ("m00", "m01", "m02"),
        ("m10", "m11", "m12"),
        ("m20", "m21", "m22"),
    )

    def __init__(self, rows: Iterable[Iterable[float]] | None = None) -> None:
        if rows is None:
            # Identity matrix
            self.m00, self.m01, self.m02 = 1.0, 0.0, 0.0
            self.m10, self.m11, self.m12 = 0.0, 1.0, 0.0
            self.m20, self.m21, self.m22 = 0.0, 0.0, 1.0
            return

        it = [tuple(map(float, r)) for r in rows]
        if len(it) != 3 or any(len(r) != 3 for r in it):
            raise ValueError("Matrix expects 3 rows of 3 floats")
        self.m00, self.m01, self.m02 = it[0][0], it[0][1], it[0][2],
        self.m10, self.m11, self.m12 = it[1][0], it[1][1], it[1][2],
        self.m20, self.m21, self.m22 = it[2][0], it[2][1], it[2][2]

    @classmethod
    def identity(cls) -> "Matrix":
        return cls()

    @classmethod
    def from_rows(cls, r0: Iterable[float], r1: Iterable[float], r2: Iterable[float]) -> "Matrix":
        return cls((r0, r1, r2))

    @classmethod
    def from_cols(cls, c0: Iterable[float], c1: Iterable[float], c2: Iterable[float]) -> "Matrix":
        c0 = tuple(map(float, c0)); c1 = tuple(map(float, c1)); c2 = tuple(map(float, c2))
        if not (len(c0) == len(c1) == len(c2) == 3):
            raise ValueError("Columns must be length 3.")
        return cls(((c0[0], c1[0], c2[0]),
                    (c0[1], c1[1], c2[1]),
                    (c0[2], c1[2], c2[2])))

    @classmethod
    def rotation_xyz(cls, rx: float, ry: float, rz: float) -> "Matrix":
        """Build rotation R = Rz(rz) @ Ry(ry) @ Rx(rx) (column-vector convention)."""
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        m00 = cz * cy
        m01 =  cz*sy*sx - sz*cx
        m02 =  cz*sy*cx + sz*sx
        m10 =  sz*cy
        m11 =  sz*sy*sx + cz*cx
        m12 =  sz*sy*cx - cz*sx
        m20 = -sy
        m21 =  cy*sx
        m22 =  cy*cx
        return cls.from_rows((m00, m01, m02), (m10, m11, m12), (m20, m21, m22))

    def copy(self) -> "Matrix":
        return Matrix.from_rows((self.m00, self.m01, self.m02),
                                (self.m10, self.m11, self.m12),
                                (self.m20, self.m21, self.m22))

    def __repr__(self) -> str:
        r0 = f"[{self.m00:.6g}, {self.m01:.6g}, {self.m02:.6g}]"
        r1 = f"[{self.m10:.6g}, {self.m11:.6g}, {self.m12:.6g}]"
        r2 = f"[{self.m20:.6g}, {self.m21:.6g}, {self.m22:.6g}]"
        return f"Mat3(\n  {r0},\n  {r1},\n  {r2}\n)"

    def __iter__(self) -> Iterator[tuple[float, float, float]]:
        yield (self.m00, self.m01, self.m02)
        yield (self.m10, self.m11, self.m12)
        yield (self.m20, self.m21, self.m22)

    def __getitem__(self, key: tuple[int, int]) -> float:
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if not isinstance(i, int) or not isinstance(j, int):
                raise TypeError("Indices must be ints.")
            if (not -3 <= i < 3 or not -3 <= j < 3):
                raise ValueError("Indices out of range (0..2)")
            name = Matrix._names[i][j]
            return getattr(self, name)
        else:
            raise TypeError("key must be tuple[int, int]")

    def __setitem__(self, key, value) -> None:
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if not isinstance(i, int) or not isinstance(j, int):
                raise TypeError("indices must be ints")
            if not -3 <= i < 3 or not -3 <= j < 3:
                raise IndexError("indices out of range (0..2)")
            name = Matrix._names[i][j]
            setattr(self, name, float(value))
            return
        if isinstance(key, int):
            # assign an entire row
            row = tuple(map(float, value))
            if len(row) != 3:
                raise ValueError("row assignment expects 3 floats")
            for j, v in enumerate(row):
                setattr(self, Matrix._names[key][j], v)
            return
        raise TypeError("Invalid key.")

    def transpose(self) -> "Matrix":
        return Matrix.from_rows((self.m00, self.m10, self.m20),
                                (self.m01, self.m11, self.m21),
                                (self.m02, self.m12, self.m22))

    T = property(transpose) # Shorthand M.T now possible

    def det(self) -> float:
        a, b, c = self.m00, self.m01, self.m02
        d, e, f = self.m10, self.m11, self.m12
        g, h, i = self.m20, self.m21, self.m22
        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

    def inverse(self) -> "Mat3":
        # Adjugate / determinant formula; fine for 3-by-3
        a, b, c = self.m00, self.m01, self.m02
        d, e, f = self.m10, self.m11, self.m12
        g, h, i = self.m20, self.m21, self.m22
        A =  (e*i - f*h);  B = -(b*i - c*h);  C =  (b*f - c*e)
        D = -(d*i - f*g);  E =  (a*i - c*g);  F = -(a*f - c*d)
        G =  (d*h - e*g);  H = -(a*h - b*g);  I =  (a*e - b*d)
        det = a*A + b*D + c*G
        if det == 0.0:
            raise ZeroDivisionError("singular matrix")
        inv_det = 1.0 / det
        return Matrix.from_rows((A*inv_det, B*inv_det, C*inv_det),
                                (D*inv_det, E*inv_det, F*inv_det),
                                (G*inv_det, H*inv_det, I*inv_det))

    def __matmul__(self, other):
        # Mat3 @ Mat3
        if isinstance(other, Matrix):
            a = self
            b = other
            return Matrix.from_rows(
                (
                    a.m00*b.m00 + a.m01*b.m10 + a.m02*b.m20,
                    a.m00*b.m01 + a.m01*b.m11 + a.m02*b.m21,
                    a.m00*b.m02 + a.m01*b.m12 + a.m02*b.m22,
                ),
                (
                    a.m10*b.m00 + a.m11*b.m10 + a.m12*b.m20,
                    a.m10*b.m01 + a.m11*b.m11 + a.m12*b.m21,
                    a.m10*b.m02 + a.m11*b.m12 + a.m12*b.m22,
                ),
                (
                    a.m20*b.m00 + a.m21*b.m10 + a.m22*b.m20,
                    a.m20*b.m01 + a.m21*b.m11 + a.m22*b.m21,
                    a.m20*b.m02 + a.m21*b.m12 + a.m22*b.m22,
                ),
            )
        # Mat3 @ vector-like -> return same type when possible
        if hasattr(other, "x") and hasattr(other, "y") and hasattr(other, "z"):
            x = self.m00*other.x + self.m01*other.y + self.m02*other.z
            y = self.m10*other.x + self.m11*other.y + self.m12*other.z
            z = self.m20*other.x + self.m21*other.y + self.m22*other.z
            try:
                return other.__class__(x, y, z)
            except Exception:
                return (x, y, z)
        return NotImplemented

    def __mul__(self, s: float) -> "Matrix":
        if not isinstance(s, (int, float)):
            return NotImplemented
        s = float(s)
        return Matrix.from_rows((self.m00*s, self.m01*s, self.m02*s),
                                (self.m10*s, self.m11*s, self.m12*s),
                                (self.m20*s, self.m21*s, self.m22*s))

    __rmul__ = __mul__

    def to_Blender(self) -> mathutils.Matrix:
        """Return a mathutils.Matrix (3×3). Requires Blender's mathutils."""
        return mathutils.Matrix(((self.m00, self.m01, self.m02),
                                 (self.m10, self.m11, self.m12),
                                 (self.m20, self.m21, self.m22)))

    @classmethod
    def from_Blender(cls, M: mathutils.Matrix) -> "Matrix":
        return cls(((float(M[0][0]), float(M[0][1]), float(M[0][2])),
                    (float(M[1][0]), float(M[1][1]), float(M[1][2])),
                    (float(M[2][0]), float(M[2][1]), float(M[2][2]))))


class CurveObject():
    """Base class for all curves."""
    __slots__ = ("name", 
                 "_location",
                 "_scale",
                 "_rotation")
    # Methods:
    # - boundary box
    # - intersections 
    # - self intersections
    # - create blender curve object

    def __init__(self,
                 name = "Curve Object",
                 location = Vector(), 
                 scale = Vector(1.0, 1.0, 1.0),
                 rotation = Vector()
                 ):
        self.name: str = name
        self._location: Vector = location
        self._scale: Vector = scale
        self._rotation: Vector = rotation

    @property # Use property for location and rotation so that subclasses can also do that.
    def location(self):
        return self._location

    @location.setter
    def location(self, location: Vector) -> None:
        self._location = location

    @property
    def scale(self) -> Vector:
        return self._scale

    @scale.setter
    def scale(self, scale: Vector) -> None:
        self._scale = scale

    @property
    def rotation(self) -> Vector:
        return self._rotation

    @rotation.setter
    def rotation(self, rotation: Vector) -> None:
        self._rotation = rotation


class QuadraticBezier():
    """Class to handle some functions of a quadratic Bezier curve.
    Used mainly for handling derivatives, etc, of a cubic Bezier."""
    __slots__ = ("points")

    def __init__(self, p0: Vector = Vector(), p1: Vector = Vector(), p2: Vector = Vector()):
        self.points = [p0, p1, p2]

    def __call__(self, t: float) -> Vector:
        p = self.points
        return p[0] * (1 - t)**2 + 2 * p[1] * (1 - t)*t + p[2] * t**2

    def eval_derivative(self, t: float):
        """Evaluates the derivative of the curve at parameter t."""
        p = self.points
        return -2 * p[0] * (1 - t) - 2 * p[1] * t + 2 * p[1] * (1 - t) + 2 * p[2] * t


class Bezier(CurveObject):
    """Cubic bezier curve. p0, p1, p2, p3 are mathutils.Vector
    t0 and t1 are the parameter time at the start and the end of the 
    original curve for instances created as splits of larger curves.
    """
    __slots__ = ("points", 
                 "t0", "t1",
                 "start_handle_left", # The extra handles attached to Blender's bezier curves.
                 "end_handle_right",
                 "is_closed" # In Blender, a single Bezier curve can be toggled closed (it is really handled as a spline).
                 )

    # WARNING: points are always the local points. Any algorithm, e.g. for calculating intersections, 
    # must be aware of the location and rotation! How can we do that?
    # Instead of just using bez.points, we should have a method that calculates the translated and rotated points.
    def __init__(self, 
                 p0: Vector = Vector(),
                 p1: Vector = Vector(),
                 p2: Vector = Vector(),
                 p3: Vector = Vector(),
                 start_handle_left: Vector = Vector(),
                 end_handle_right: Vector = Vector(),
                 t0: float = 0.0,
                 t1: float = 1.0, 
                 is_closed: bool = False,
                 name = "Bezier",
                 location: Vector = Vector(),
                 scale = Vector(1.0, 1.0, 1.0),
                 rotation = Vector(),
                 ) -> None:
        """ 
        Initializes the cubic Bezier and sets its points and degree. 
        The points should be mathutils.Vectors of some fixed dimension.
        The number of points should be 3 or higher. 
        """
        super().__init__(name, location, scale, rotation)
        self.points: list[Vector] = [p0, p1, p2, p3]

        # The dangling handles of a Bezier curve in Blender are not really part of a mathematical Bezier.
        # Instead they belong to the previous or next Bezier in case of a poly-bezier curve.
        # Since Blender uses them, it is better to keep them.
        self.start_handle_left: Vector = start_handle_left
        self.end_handle_right: Vector = end_handle_right
        self.is_closed: bool = is_closed # In Blender, single Bezier curves can be toggled closed.

        # t0 and t1 give the parameter values of the parent curve in case self is created from a split. 
        # Needed for keeping track of intersections.
        self.t0: float = t0
        self.t1: float = t1

    @classmethod
    def from_Blender(cls, name: str):
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

        return cls(p0, p1, p2, p3, 
                   name = name,
                   location = loc, 
                   scale = sca,
                   rotation = rot,
                   start_handle_left = start_handle_left,
                   end_handle_right = end_handle_right,
                   is_closed = is_closed
                   )

    def __repr__(self):
        """Prints the name of the together with all the points. """
        p = self.points
        return f"Bezier(\nname={self.name}, \np0={str(p[0])}, \np1={str(p[1])}, \np2={str(p[2])}, \np3={str(p[3])}, \nleft={str(self.start_handle_left)}, \nright={str(self.end_handle_right)}, \nt0={self.t0}, \nt1={self.t1}\n)"
        # string = '<' + self.name + '\n' 
        # string += "p0= " + str(p[0]) + '\n'
        # string += "p1= " + str(p[1]) + '\n'
        # string += "p2= " + str(p[2]) + '\n'
        # string += "p3= " + str(p[3]) + '\n'
        # string += "start_handle_left: " + str(self.start_handle_left) + '\n'
        # string += "end_handle_right: " + str(self.end_handle_right) + '>'
        # return string

    def __call__(self, t: float, world_space: bool = False, global_t: bool = False) -> Vector:
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
        pos = p[0] * (1 - t)**3 + 3 * p[1] * (1 - t)**2 * t + 3 * p[2] * (1 - t) * t**2 + p[3] * t**3
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
        self.start_handle_left, self.end_handle_right = self.end_handle_right, self.start_handle_left

    def set_point(self, point: Vector, i: int) -> None:
        """Sets the point with index i of the Bezier."""
        self.points[i] = point

    def translate_origin(self, vector: Vector) -> None:
        """Translates the origin of the Bezier to the position given by 
        vector without changing the world position of the curve. 
        """
        dist = self.location - vector
        self.points = [p + dist for p in self.points]
        self.start_handle_left = self.start_handle_left +  dist
        self.end_handle_right = self.end_handle_right + dist
        self.location = vector

    def eval_derivative(self, t: float, global_t: bool = False) -> Vector:
        """Evaluate derivative at parameter t. If global_t and self was
        split from a larger curve, then the derivative is evaluated at the 
        parameter that corresponds to t of the original curve."""
        if global_t:
            denom = self.t1 - self.t0
            # TODO: Store this in variable or make function for float comparisions.
            if denom < 1e-12: 
                t = 0.0
            else:
                t = (t - self.t0) / denom
        p = self.points
        return 3 * (p[1] - p[0]) * (1-t)**2 + 6 * (p[2] - p[1]) * (1-t) * t + 3 * (p[3] - p[2]) * t**2

    def eval_second_derivative(self, t: float):
        """Evaluate the second derivative at parameter t."""
        # TODO: Only used in curvature, which is currently not used for anything. 
        p = self.points
        return 6 * (p[0] - 2 * p[1] + p[2]) * (1-t) + 6 * (p[1] - 2 * p[2] + p[3]) * t

    def derivative(self):
        """Returns the derivative (quadratic Bezier) curve of self."""
        p = self.points
        return QuadraticBezier(3 * (p[1] - p[0]), 3 * (p[2] - p[1]), 3 * (p[3] - p[2]))

    def tangent(self, t: float): 
        """Calculate the unit tangent at parameter t."""
        # TODO: This is just a convenience function and can be removed.
        derivative = self.eval_derivative(t) 
        if derivative.length() > 0.0: 
            return derivative / derivative.length()
        else:
            return derivative # mathutils.Vector((0,0,0))

    def normal(self, t: float):
        """Return the tangent rotated 90 degrees anti-clockwise."""
        # The real normal always points in the direction the curve is turning
        # so we should probably call this something else.
        tangent = self.tangent(t)
        return mathutils.Vector((-tangent.y, tangent.x, 0.0))

    def curvature(self, t):
        """Returns the curvature at parameter t."""
        # TODO: Can probably be removed. 
        d = self.eval_derivative(t)
        sd = self.eval_second_derivative(t)
        denom = math.pow(d.length(), 3 / 2)
        return (d[0] * sd[1] - sd[0] * d[1]) / denom

    def x_moment(self):
        """Calculate the x_moment of a curve that starts at the origin
        and ends on the x-axis."""
        # TODO: Not used.
        p = self.points
        x1 = p[1][0]
        y1 = p[1][1]
        x2 = p[2][0]
        y2 = p[2][1]
        x3 = p[3][0]

        moment = -9*x1**2*y2/280 + 9*x1*x2*y1/280 - 9*x1*x2*y2/280 + 3*x1*x3*y1/140 + 9*x2**2*y1/280 + 3*x2*x3*y1/56 + 3*x2*x3*y2/56 + x3**2*y1/28 + x3**2*y2/8
        return moment

    def eval_offset(self, t: float, d: float):
        """Calculates the vector from self(t) to the offset at parameter t."""
        # Could be calculated via the normal as (t) + d * N(t), 
        # but the normal always points in the direction the curve 
        # is turning. 
        # By using a rotation we instead always get a rotation to the left
        # side of the curve and we can control which direction the offset is.
        n = None
        dp = self.eval_derivative(t) 
        if dp.length() > 0.0:
            s = d / dp.length()
        else:
            if t == 0.0:
                p1 = self(0.0001)
                dp = p1 - self.points[0] 
                s = d / dp.length
            elif t == 1.0:
                p2 = self(.9999)
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
        # TODO: If this is needed, rewrite the code so that usage of mathutils is minimized. 
        # mathutils is not exact enough.
        # Otherwise, remove.
        m = mathutils.Matrix.Translation(-self.points[0])
        end = m @ self.points[3]
        if end[0] != 0.0:
            angle = -math.atan2(end[1],end[0])
        else:
            angle = 0.0
        k = mathutils.Matrix.Rotation(angle, 4, 'Z') @ m

        aligned_points = []
        for p in self.points:
            aligned_points.append(k @ p)
        return aligned_points

    def bounding_box(self, world_space = False):
        # TODO: could be removed if a new version is included in self_intersections.
        """Calculate the bounding box of the curve.
        Returns dict('min_x', 'max_x', 'min_y', 'max_y')"""
        # TODO: Make it possible to calculate the tight bounding box if this
        # is deemed useful. 
        extrema = self.extrema()
        x_values = [self(t, world_space = world_space)[0] for t in extrema] 
        y_values = [self(t, world_space = world_space)[1] for t in extrema] 
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
        # TODO: Clean up using e.g. itertools. 
        # TODO: This must take the rotation into account when that is added.
        p0, p1, p2, p3 = self.points

        a = 3 * (-p0 + 3 * p1 - 3 * p2 + p3)
        b = 6 * (p0 - 2*p1 + p2)
        c = 3*(p1 - p0) 
        # Solve for all points where x'(t) = 0 and y'(t) = 0.
        tx_roots = solvers.solve_quadratic(a.x, b.x, c.x) 
        ty_roots = solvers.solve_quadratic(a.y, b.y, c.y) 
        roots = [0.0, 1.0] # Extrema can also occur at the endpoints.
        roots.extend([t for t in tx_roots if isinstance(t, float) and 0.0 <= t <= 1.0])
        roots.extend([t for t in ty_roots if isinstance(t, float) and 0.0 <= t <= 1.0])

        return roots

    def map_split_to_whole(self, t: float):
        """Returns the parameter value of the original curve corresponding
        to the parameter value t of a split.
        """
        # If self is the original curve (t0 = 0, t1 = 1) then it will just return t.
        # E.g. if self is a split between 0 and 0.5 of a curve then t = 0.5 correpsonds to t = 0.25 of the original curve.
        return self.t0 + (self.t1 - self.t0) * t

    def subsegment(self, t0: float, t1: float):
        """Split out the subsegment between t0 and t1.
        If the curves have been previously split, we can insert the parameter
        of the original whole curve where we want to split the curve."""


        p0 = self(t0)
        p3 = self(t1)
        factor = (t1 - t0) / 3
        d1 = self.eval_derivative(t0)
        p1 = p0 + factor * d1
        d2 = self.eval_derivative(t1)
        p2 = p3 - factor * d2

        t0new = self.map_split_to_whole(t0)
        t1new = self.map_split_to_whole(t1)

        # TODO: Update the name to something better? 
        # Remove any previous ranges in the name.
        i = self.name.find('(')
        if i > 0:
            text = self.name[:i]
        else:
            text = self.name
        name = text + '(' + str(t0new) + ',' + str(t1new) + ')'
        loc = self.location
        sca = self.scale
        rot = self.rotation
        return Bezier(p0, p1, p2, p3,
                      t0 = t0new, t1 = t1new,
                      name = name,
                      location = loc,
                      scale = sca,
                      rotation = rot)

    def split2(self, t: float):
        # De Casteljau subdivision
        p0, p1, p2, p3 = self.points
        # p0: Vector = self.points[0]
        # p1: Vector = self.points[1]
        # p2: Vector = self.points[2]
        # p3: Vector = self.points[3]

        p01: Vector = p0.lerp(p1, t)
        p12: Vector = p1.lerp(p2, t)
        p23: Vector = p2.lerp(p3, t)
        p012: Vector = p01.lerp(p12, t)
        p123: Vector = p12.lerp(p23, t)
        p0123: Vector = p012.lerp(p123, t)

        # Calculate the parameter value at the original curve if split multiple times.
        mid_t = self.map_split_to_whole(t) 

        i = self.name.find('(')
        if i > 0:
            text = self.name[:i]
        else:
            text = self.name
        name_l = text + '(' + str(self.t0) + ',' + str(mid_t) + ')'
        name_r = text + '(' + str(mid_t) + ',' + str(self.t1) + ')'

        left = Bezier(p0, p01, p012, p0123, name = name_l, t0 = self.t0, t1 = mid_t)
        right = Bezier(p0123, p123, p23, p3, name = name_r, t0 = mid_t, t1 = self.t1)

        return left, right

    def split2_o(self, *parameters: float):
        """Split the curve at parameters. Returns a list of the split segments."""
        # TODO: Rename this to split as soon as the other one is removed.
        # TODO: Should this return a Curve instead?
        ts = sorted([t for t in parameters])
        if ts[0] != 0.0: 
            ts.insert(0, 0.0)
        if ts[-1] != 1.0: 
            ts.append(1.0)
        
        sub_curves: list[Bezier] = []
        for i in range(1, len(ts)): 
            sub_curves.append(self.subsegment(ts[i-1], ts[i]))

        return sub_curves

    def _create_Blender_curve(self):
        """Creates a new curve object in Blender."""
        # TODO: Catch the name of the object created by ob.name.
        # and store this for later reference?
        # TODO: Nest this inside of add_to_Blender()
        cu = bpy.data.curves.new(self.name, 'CURVE')
        ob = bpy.data.objects.new(self.name, cu)
        ob.location = self.location
        ob.scale = self.scale
        ob.rotation_euler = self.rotation
        bpy.data.collections['Collection'].objects.link(ob)
        ob.data.resolution_u = 64
        cu.splines.new('BEZIER') # Add a first spline to Blender.
        return cu

    def add_to_Blender(self, blender_curve_object = None, stringed = False):
        """Adds the Bezier curve to Blender as splines.
        blender_curve_object: an existing curve in Blender.
        stringed: is a bool which is true in case the Bezier curve is part of a Spline.
        Both the parameters are used to make it possible to reuse this in the case where
        the Bezier curve is part of a Spline.
        """

        # Stringed = True means that the Bezier as part of a Spline.
        # The end and beginning start_handle_left and end_handle_right
        # should not be set, since these are set by the previous and the
        # next Bezier curves in the spline.
        # where the end point of one curve coincides with the start point
        # of the next. 

        p = self.points

        cu = blender_curve_object or self._create_Blender_curve()

        spline = cu.splines[-1]
        spline.use_cyclic_u = self.is_closed
        bezier_points = spline.bezier_points

        bezier_points[-1].handle_right = p[1]
        bezier_points.add(1)
        bezier_points[-1].handle_left = p[2]
        bezier_points[-1].co = p[3]
        if not stringed:
            # If not part of spline set also the first point and the dangling handles.
            bezier_points[-2].co = p[0]
            bezier_points[-2].handle_left = self.start_handle_left or p[0]
            bezier_points[-1].handle_right = self.end_handle_right or p[3]

    def overlaps(self, bezier):
        """Check if the bounding box of self and Bezier overlaps."""
        # TODO: Only used in self_intersections.
        # Remove this when new algorithm is in place!
        # 0      1      2      3
        # min_x, max_x, min_y, max_y 
        bb1 = self.bounding_box(world_space = True)
        bb2 = bezier.bounding_box(world_space = True)
        return not (bb1['min_x'] > bb2['max_x'] or bb2['min_x'] > bb1['max_x'] or bb1['min_y'] > bb2['max_y'] or bb2['min_y'] > bb1['max_y'])

    def transform(self, angle = 0.0, translation = mathutils.Vector((0,0,0)), world_space = False):
        """Rotates the curve an angle around the z axis and then translates it.
        If world_space, then the curve object is transformed instead of the local coordinates."""
        # TODO: Implement the world_space option. Use the superclass and put the rotation there.
        if world_space: 
            print("NOT YET IMPLEMENTED: Will do local tranform instead.")
        m = mathutils.Matrix.Rotation(angle, 3, 'Z')
        p = self.points
        q0: mathutils.Vector = m @ p[0] + translation # type: ignore
        q1: mathutils.Vector = m @ p[1] + translation # type: ignore
        q2: mathutils.Vector = m @ p[2] + translation # type: ignore
        q3: mathutils.Vector = m @ p[3] + translation # type: ignore
        self.points = [q0, q1, q2, q3] # type: ignore

    def intersect_ray(self, sp: mathutils.Vector, d: mathutils.Vector):
        """Find intersection of the cubic with a ray from point sp in the direction d.
        Return the parameter value t where the intersection occurs (if any)."""
        # NOTE: Part of good offset.
        # TODO: Check for improvements. Perhaps it is only necessary to check for rays in a given direction, e.g. +x.
        # Then this could be simplied a lot.
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
        # NOTE: Part of good offset.
        results = []
        n = 200 # Arbitary number? TODO: Add this as a parameter.
        q = self.derivative()
        ya = 0.0
        last_t = 0.0
        t0 = 0.0
        for i in range(0, n + 1):
            t = i / n 
            ds = q(t).length()
            # Curvature
            k = (q.eval_derivative(t).x * q(t).y - q.eval_derivative(t).y * q(t).x) / ds**3
            yb = k * d + 1
            if i != 0:
                # Sign has changed
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

    def offset(self, d: float, double_side: bool = False): 
        # NOTE: Part of good offset.
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
                        # TODO: In case we just create the splines directly,
                        # location, etc, can be set in the resulting spline
                        # only.
                        bez.location = self.location
                        bez.scale = self.scale
                        bez.rotation = self.rotation
                roffsets.extend(offset)
        return [loffsets, roffsets]

    def stroke(self, d: float): 
        # NOTE: Part of good algorithm.
        # d = self.d
        offsets = self.offset(d, double_side = True)
        left_offset = Spline(*offsets[0], location = self.location, rotation = self.rotation)
        right_offset = Spline(*offsets[1], location = self.location, rotation = self.rotation)
        left_offset.name = 'Left'
        right_offset.name = 'Right'
        left_offset.add_to_Blender()
        right_offset.add_to_Blender()

    def curve_intersections(self, c2: 'Bezier', threshold = INTERSECTION_THRESHOLD):
        """Recursive method used for finding the intersection between the two Bezier curves self and c2.
        """
        # TODO: Check for endpoint matching.
        # NOTE: Will be superseeded.
        threshold2 = threshold * threshold
        bez0 = self
        results: list[tuple[float, float]] = [] 
        # if bez0.t1 - bez0.t0 < threshold and c2.t1 - c2.t0 < threshold:
        if bez0.bounding_box()['area'] < threshold2 and c2.bounding_box()['area'] < threshold2:
            # return [((bez0.t0 + bez0.t1)/2 , (c2.t0 + c2.t1)/2)]
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
        # TODO: Correct or improve with Newton-Rhapson
        """Finds the self intersection of the curve (if any).
        Returns three parameter values.
        Ref: https://comp.graphics.algorithms.narkive.com/tqLNEZqM/cubic-bezier-self-intersections
        """
        p0 = self.points[0]
        p1 = self.points[1]
        p2 = self.points[2]
        p3 = self.points[3]

        H1 = -3 * p0 + 3 * p1
        H2 = 3 * p0 - 6 * p1 + 3 * p2
        H3 = - p0 + 3 * p1 - 3 * p2 + p3 
        if H3 == mathutils.Vector((0,0,0)):
            return None

        A = H2.x / H3.x
        B = H1.x / H3.x
        P = H2.y / H3.y
        Q = H1.y / H3.y
        
        if A == P or Q == B:
            return None

        k = (Q - B) / (A - P)

        r0 = (- k**3 - A * k**2 - B * k ) / 2
        r1 = (3 * k**2 + 2 * k * A + 2 * B) / 2
        r2 = - 3 * k / 2;
        sols = solvers.solve_cubic(1.0, r2, r1, r0)
        
        if sols: 
            solutions: list[float] = []
            for s in sols:
                if isinstance(s, float) and (s >= 0.0 and s <= 1.0):
                    solutions.append(s)
                solutions.sort()
            if len(solutions) == 3:
                # The middle solution is a rouge solution.
                # The first and last are the two parameter values where it meets itself. 
                # Only need the first.
                return [solutions[0], solutions[2]]
        return None

    def tolerances_for_curve(self, *, k=32, 
                             coord_min=1e-8, coord_max=1e-3,
                             t_min=1e-7, t_max=5e-4):
        """
        Pick robust float32-friendly tolerances based on curve scale.
        Returns: dict with coord_tol, t_tol, det_tol, tangent_cross_tol
        - k: multiplier for eps32; increase if you need looser geometry tests.
        - coord_min/max: clamps for coordinate tolerance (absolute units).
        - t_min/max: clamps for t-space tolerance.
        """
        # Helper functions
        def norm2(p): 
            return math.hypot(p[0], p[1])

        def bbox_span(ctrls):
            xs = [p[0] for p in ctrls]; ys = [p[1] for p in ctrls]
            return max(max(xs)-min(xs), max(ys)-min(ys))

        def control_poly_len(ctrls):
            return (norm2((ctrls[1][0]-ctrls[0][0], ctrls[1][1]-ctrls[0][1])) +
                    norm2((ctrls[2][0]-ctrls[1][0], ctrls[2][1]-ctrls[1][1])) +
                    norm2((ctrls[3][0]-ctrls[2][0], ctrls[3][1]-ctrls[2][1])))

        eps32 = 1.1920929e-07  # np.finfo(np.float32).eps without importing numpy

        P = self.points
        span = bbox_span(P)                  # characteristic size
        cplen = control_poly_len(P)          # ~ average speed proxy
        scale = max(span, cplen, 1.0)         # avoid degenerate 0

        # --- Geometric (coordinate) tolerance ---
        # Scale with eps and scene size; clamp to a sensible window.
        coord_tol = max(coord_min, min(coord_max, k * eps32 * scale))

        # --- Parametric tolerance ---
        # Map coord_tol back to t via a speed estimate (≈ control poly length).
        avg_speed = max(cplen, 1e-8)
        t_tol = max(t_min, min(t_max, coord_tol / avg_speed))

        # --- Line-line parallel determinant tolerance (area units) ---
        # Determinant involves length^2; scale accordingly.
        det_tol = 10.0 * eps32 * (scale ** 2)

        # --- Tangency test tolerance for cross(B'(tA), B'(tB)) ---
        # Cross has units length^2; use speed^2 scaling.
        tangent_cross_tol = 10.0 * eps32 * (avg_speed ** 2)

        return dict(coord_tol=coord_tol, t_tol=t_tol, det_tol=det_tol, tangent_cross_tol=tangent_cross_tol)

    def ints(self, other: "Bezier"):
        """Calculates the intersction between self and other.
        Follows: http://nishitalab.org/user/nis/cdrom/cad/CAGD90Curve.pdf
        Sederberg, and Nishita, Curve Intersection by Bezier Clipping, 1990
        """
        # TODO: Add all tolerances as parameters. 
        # Tolerances found: 
        # horisontal_hits, eps=1e-12 for determining if the denominator is close to zero.
        # merge_spans, eps=1e-15, for determining if spans should be joined. Might not be necessary at all. 
        # clip_against_fatline, eps=1e-12 to be passed to horisontal hits and to soften fatline slightly.
        # Main function (ints) tau = 1e-10 to determine parameter exactness, also sent to add_results to remove
        # duplicates differing by less than this. 

        def fatline(c: "Bezier"):
            """Calculates a fatline around the curve c.
            Returns start and end points p0 and p3, and the widths of
            the fatline dmin and dmax.
            """
            p0, p1, p2, p3 = c.points
            chord = p3 - p0 # TODO: Check if p3 = p0 (approx) then return None?
            normal = chord.perpendicular().normalize()
            # Distance of controls p1 and p2 t to the chord.
            d1 = (p1 - p0).dot(normal)
            d2 = (p2 - p0).dot(normal)
            if d1 * d2 > 0:
                dmin = 0.75 * min(0, d1, d2)
                dmax = 0.75 * max(0, d1, d2)
            else: 
                dmin = 4/9 * min(0, d1, d2)
                dmax = 4/9 * max(0, d1, d2)

            k1 = p0 + normal * dmin
            k2 = p3 + normal * dmin
            l1 = p0 + normal * dmax
            l2 = p3 + normal * dmax
            # add_line(k1.to_mu_vector(), k2.to_mu_vector())
            # add_line(l1.to_mu_vector(), l2.to_mu_vector())
            # TODO: p3 does not have to be returned.
            return p0, p3, normal, dmin, dmax

        def signed_distances(c: "Bezier", q0: Vector, q3: Vector, normal: Vector):
            """Calculates the signed distances of the controls of c 
            to the lne q0-q3."""
            # TODO: q3 not used.
            p0, p1, p2, p3 = c.points
            d0 = (p0 - q0).dot(normal)
            d1 = (p1 - q0).dot(normal)
            d2 = (p2 - q0).dot(normal)
            d3 = (p3 - q0).dot(normal)

            # k0 = p0 - normal * d0
            # k1 = p1 - normal * d1
            # k2 = p2 - normal * d2
            # k3 = p3 - normal * d3
            # add_line(p0.to_mu_vector(), k0.to_mu_vector())
            # add_line(p1.to_mu_vector(), k1.to_mu_vector())
            # add_line(p2.to_mu_vector(), k2.to_mu_vector())
            # add_line(p3.to_mu_vector(), k3.to_mu_vector())
            return d0, d1, d2, d3

        def cross(o: Vector, a: Vector, b: Vector):
            """Cross product of two dimensional vectors a - o and b - o. 
            Returns only the z-component."""
            v1 = a - o
            v2 = b - o
            return v1.cross(v2)[2]

        def calculate_convex_hull(d0: float, d1: float, d2: float, d3: float):
            """Calculate the upper and lower hulls of the constructed bezier curve."""
            # Create the 'non-parametric' curve D(t) = (t, d(t)) where d(t) is the signed
            # distance to the other curve. The time parameter is evenly spaced.
            # The controls points are:
            points = []
            points.append(Vector(0,   d0, 0))
            points.append(Vector(1/3, d1, 0))
            points.append(Vector(2/3, d2, 0))
            points.append(Vector(1,   d3, 0))

            lower = []
            for p in points:
                # Check if the control polygon is convex and keep only the points that 
                # are. The cross product check if we are turning to the right. 
                while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)

            upper = []
            for p in points:
                # Check if the control polygon is convex and keep only the points that 
                # are. The cross product check if we are turning to the right. 
                while len(upper) >= 2 and cross(upper[-2], upper[-1], p) >= 0:
                    upper.pop()
                upper.append(p)
            return lower, upper

        def horizontal_hits(hull: list[tuple[float, float]], d: float, eps=1e-12):
            """Find the intersection of the hull points (chain) and the 
            horizontal line d."""
            # Pair the points
            hits = []
            for (ta, da, _), (tb, db, _) in zip(hull, hull[1:]):
                denom = db - da
                if abs(denom) < eps:
                    continue
                s = (d - da) / denom
                if 0 <= s <= 1:
                    t_hit = ta + (tb - ta) * s
                    hits.append(t_hit)
            return hits

        def eval_hull(hull: list[tuple[float, float]], s: float):
            """Evaluate the hull (chain) at parameter value s."""
            for (sa, da, _), (sb, db, _) in zip(hull, hull[1:]):
                if sa <= s <= sb:
                    t = (s - sa) / (sb - sa) # NOTE: sb != sa by construction of chain.
                    return da + t * (db - da)
            # if s is exaclty 1.0 or numerical tail? 
            return hull[-1][1]

        def merge_spans(spans: list[tuple[float, float]], eps: float = 1e-15):
            """Merges the parameter intervals spans = [(t0, t1), (t2, t3), ...] 
            so that overlapping spanns are combined."""
            if not spans:
                return []
            spans.sort()
            out = [list(spans[0])]
            for a, b in spans[1:]:
                if a <= out[-1][1] + eps: # overlaps or touches
                    out[-1][1] = max(out[-1][1], b)
                else:
                    out.append([a,b])
            return [(a,b) for a, b in out] # Convert back to list of tuples.

        def clip_against_fatline(c1: "Bezier", c2: "Bezier", eps=1e-12):
            """Creates a fatline around c2 and clips c1 against it."""
            
            # TODO: What does eps do here?

            # Calculate the fatline of c2.
            q0, q3, normal, dmin, dmax = fatline(c2)

            # Determine the non-parametric curve.
            d0, d1, d2, d3 = signed_distances(c1, q0, q3, normal)
            # Find upper and lower convex hull around the non-parametric curve.
            lower, upper = calculate_convex_hull(d0, d1, d2, d3)

            # List all candidates.
            cand = {0.0, 1.0}
            cand.update(s for s,_,_ in lower)
            cand.update(s for s,_,_ in upper)

            cand.update(horizontal_hits(lower, dmin, eps))
            cand.update(horizontal_hits(lower, dmax, eps))
            cand.update(horizontal_hits(upper, dmin, eps))
            cand.update(horizontal_hits(upper, dmax, eps))

            S = sorted(cand)
            if len(S) < 2:
                return None

            feasible = []
            for a, b in zip(S, S[1:]):
                sm = 0.5 * (a + b)
                dlo = eval_hull(lower, sm)
                dup = eval_hull(upper, sm)
                if (dlo <= dmax + eps) and (dup >= dmin - eps):
                    feasible.append((a, b))

            spans = merge_spans(feasible, eps=1e-15)
            if not spans:
                return None

            sL = max(0.0, spans[0][0] - eps)
            sR = min(1.0, spans[-1][1] + eps)
            return (sL, sR)
            # return spans

        def add_result(results, t0a, t1a, t0b, t1b, tol=1e-10):
            """Add result only if it is not already present."""
            for (a,b, c, d) in results:
                if abs(a - t0a) < tol and abs(b - t1a) < tol and abs(c - t0b) < tol and abs(d - t1b) < tol:
                    return
            results.append((t0a, t1a, t0b, t1b))

        def one_clipping(c1: "Bezier", c2: "Bezier"):
            """Clips c1 agains the fatline of c2."""
            spans1 = clip_against_fatline(c1, c2)
            if spans1:
                t0, t1 = spans1
                return c1.subsegment(t0, t1)
            else:
                return None

        from collections import deque
        que = deque()
        que.append((self, other))

        results = []
        MAX_ITER = 12
        tau = 1e-12 # TODO: Use the desired t-exactness. Perhaps even better than this?
        while que:
            new1, new2 = que.popleft()
            i = 0
            while i < MAX_ITER and ((new1.t1 - new1.t0) > tau or (new2.t1 - new2.t0) > tau):
                old1 = new1
                old2 = new2
                i += 1
                new1 = one_clipping(old1, old2) # TODO: Must handle non-intersections!
                if not new1:
                    break
                new2 = one_clipping(old2, new1)
                if not new2:
                    break
                db1 = new1.t1 - new1.t0
                db2 = new2.t1 - new2.t0
                if db1 > 0.7 * (old1.t1 - old1.t0) or db2 > 0.7 * (old2.t1 - old2.t0):
                    if db1 > db2:
                        b1a, b1b = new1.split2(0.5)
                        que.append((b1a, new2))
                        que.append((b1b, new2))
                    else: # db2 > db1
                        b2a, b2b = new2.split2(0.5)
                        que.append((new1, b2a))
                        que.append((new1, b2b))
                    break
            if new1 and new2 and (new1.t1 - new1.t0) < tau and (new2.t1 - new2.t0) < tau:
                add_result(results, new1.t0, new1.t1, new2.t0, new2.t1, tau) 
                # results.append((new1.t0, new1.t1, new2.t0, new2.t1))

        results = [(0.5*(a+b), 0.5*(c+d)) for a, b, c, d in results]
        results.sort()
        return results


class Spline(CurveObject): 
    """A list of Bezier curves corresponds to a single spline object.
    For each Bezier, the end point coincide with the starting point of 
    the next curve."""
    # TODO: Handle offsets when the handles at a point are not aligned!
    # TODO: Handle endcaps.
    # TODO: Handle intersections between two splines.
    # TODO: Handle massaging of the offset curve so that all intersections are combined. 
    # Plan: 
    # 1. Write bezier-bezier intersection method.
    # 2. Write bezier-self-intersection method.
    # 3. Write bezier-line-intersection method.
    # Algorithm: 
    # 1. Create left and right offsets. 
    # 2. Flip direction of right offset.
    # 3. Add endcaps with consistent direction.
    # 4. Find all self-intersction of the stroke outline and split the curves at these points. 
    # 5. For each segment of the stroke outline:
    #   6. Imagine the line going from directly left (as defined by the tangent) and extends to infinity. 
    #   7. Intersect this line with all beziers (do broad phase first by looking at aabb and then fine phase.
    #   8. Count the number of curves with tangent with a +y tangent as +1 and all with -y tangent as -1.
    #   9. If this number is non-zero, the curve is inside and should be deleted.
    # 10. Delete all internal curves and re-attach the curve to a continuous line (the last part might be difficult?)
    #     Potentially: At each intersection point we should have (ideally) 4 beziers attached (2 ingoing and 2 outgoing).
    #     After deletion there should be only two (one incoming and one outgoing). 

    __slots__ = ('beziers', 'is_closed', 'strokewidth')

    def __init__(self, 
                 *beziers: Bezier, 
                 is_closed = False, 
                 strokewidth = 0.01,
                 name = "Spline",
                 location = Vector(),
                 scale = Vector(1.0, 1.0, 1.0),
                 rotation = Vector(),
                 ):

        self.beziers = list(beziers)
        # Ensure that the end point and handles of one point, coincides with the corresponding for the next point.
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

        # Reset all the Beziers that comes from a split curve.
        for bezier in self.beziers:
            bezier.t0 = 0.0
            bezier.t1 = 1.0

        super().__init__(name, location, scale, rotation)

    @classmethod
    def from_Blender(cls, name: str):
        """Alternative constructor where the Spline is imported from Blender."""
        cu= bpy.data.collections['Collection'].objects[name]
        beziers = []
        loc = cu.location
        sca = cu.scale
        rot = cu.rotation_euler
        spline = cu.data.splines[0]
        is_closed = spline.use_cyclic_u

        bezier_points = spline.bezier_points
        # How can we do this better? 
        i = len(spline.bezier_points) - 1
        for j in range(0, i):
            handle_left = bezier_points[j].handle_left
            p0 = bezier_points[j].co
            p1 = bezier_points[j].handle_right
            p2 = bezier_points[j + 1].handle_left
            p3 = bezier_points[j + 1].co
            handle_right = bezier_points[j + 1].handle_right
            beziers.append(Bezier(p0, p1, p2, p3, 
                                  location = loc, 
                                  scale = sca, 
                                  rotation = rot,
                                  start_handle_left = handle_left,
                                  end_handle_right = handle_right
                                  ))
            
        return cls(*beziers, 
                   name = name, 
                   location = loc, 
                   scale = sca,
                   rotation = rot,
                   is_closed = is_closed,
                  )

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
        for bez in self.beziers:
            bez.rotation = rotation
        self._rotation = rotation

    def reverse(self):
        for bez in self.beziers:
            bez.reverse()
        self.beziers = list(reversed(self.beziers))

    def append_spline(self, spline):
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
        # Make new Bezier curves, since we do not want to modify spline.
        for bez in spline.beziers: 
            p0 = bez.points[0] + s_loc - loc
            p1 = bez.points[1] + s_loc - loc
            p2 = bez.points[2] + s_loc - loc
            p3 = bez.points[3] + s_loc - loc
            shl = bez.start_handle_left
            ehr = bez.end_handle_right
            self.beziers.append(
                Bezier(p0, p1, p2, p3,
                       start_handle_left = shl, end_handle_right = ehr,
                       location = loc, scale = sca, rotation = rot))

        # Move the start point of spline so that it matches the 
        # end point of self. 
        abez = self.beziers[a]
        abez.points[0] = last_bez.points[3]
        abez.points[1] = last_bez.end_handle_right
        abez.start_handle_left = last_bez.points[2]

    def prepend_spline(self, spline):
        """
        Add curve to the this curve at the beginning.
        The end point of spline will be moved to coincide with the start point of self.
        """
        a = len(spline.beziers)
        first_bez = self.beziers[0]
        loc = self.location
        s_loc = spline.location
        # Make new Bezier curves, since we do not want to modify spline.
        for bez in reversed(spline.beziers):
            p0 = bez.points[0] + s_loc - loc
            p1 = bez.points[1] + s_loc - loc
            p2 = bez.points[2] + s_loc - loc
            p3 = bez.points[3] + s_loc - loc
            shl = bez.start_handle_left
            ehr = bez.end_handle_right
            self.beziers.insert(0, Bezier(p0, p1, p2, p3, start_handle_left = shl, end_handle_right = ehr, location = loc) )

        # Move the start point of spline so that it matches the 
        # end point of self. 
        abez = self.beziers[a-1] # The last Bezier of spline.
        abez.points[3] = first_bez.points[0]
        abez.points[2] = first_bez.start_handle_left
        abez.end_handle_right = first_bez.points[1]
    
    def append_bezier(self, bezier: "Bezier"):
        """
        Add a single Bezier curve in the end of the curve. 
        End and start points must match. 
        """
        # The check might need to be done within some precision.
        # TODO: end_point and start_point methods are missing in Bezier!
        if self.end_point(world_space=True) == bezier.start_point(world_space=True):
            bezier.translate_origin(self.location) # Make the origins coincide.
            ep = self.end_point()
            bezier.points[0] = ep
            self.beziers.append(bezier)
        else:
            raise Exception("Start and end points of curves must be equal.")

    def prepend_bezier(self, bezier):
        """Add a single Bezier curve to the start of the curve. 
        End point of bezier must match with start point of self.
        """

        if self.start_point(world_space = True) == bezier.end_point(world_space = True): 
            bezier.translate_origin(self.location)
            sp = self.start_point()
            bezier.set_point(sp, -1)
            self.beziers.insert(0,bezier)
        else:
            raise Exception("Start and end points of curves must be equal.")

    def toggle_closed(self):
        """Toggles the curve closed.
        """
        self.is_closed = not self.is_closed 

    def _create_Blender_curve(self):
        # TODO: Is this the same in all classes? Move to CurveObject
        # TODO: Nest this inside of add_to_Blender()?
        cu = bpy.data.curves.new(self.name, 'CURVE')
        ob = bpy.data.objects.new(self.name, cu)
        ob.location = self.location
        ob.scale = self.scale
        ob.rotation_euler = self.rotation
        bpy.data.collections["Collection"].objects.link(ob)
        ob.data.resolution_u = 64
        cu.splines.new('BEZIER')
        return cu

    def add_to_Blender(self, blender_curve_object = None):
        """Adds the curve to Blender as splines. 
        """
        # TODO: How should we choose which collection to add the curve to? 
        # TODO: How can we do this so that we reuse the add_to_Blender from Bezier?
        #       Answer: Probably more hassle than this.
        # TODO: Move creation of curve object, linking, etc, to superclass.

        if blender_curve_object is None: 
            cu = self._create_Blender_curve()
        else:
            cu = blender_curve_object

        spline = cu.splines[-1]
        spline.use_cyclic_u = self.is_closed
        bezier_points = spline.bezier_points

        # TODO: This sets each bezier_point.co twice! Rethink!
        # Set the first point and the left handle of the first point.
        sp = self.start_point(world_space = False)
        sh = self.beziers[0].start_handle_left
        if sh:
            bezier_points[0].handle_left = sh
        else:
            bezier_points[0].handle_left = sp
        bezier_points[0].co = sp # Set the first point. 

        for bez in self.beziers:
            bez.add_to_Blender(cu, stringed=True)

        # Set the last handle
        eh = self.beziers[-1].end_handle_right
        if eh:
            bezier_points[-1].handle_right = eh
        else:
            bezier_points[-1].handle_right = self.end_point(world_space = False)

    def self_intersections(self, threshold = INTERSECTION_THRESHOLD):
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
        # TODO: Remove the middle solution of Bezier.find_self_intersections().
        # TODO: When closed, this misses the intersections at the end Bezier (the one returning to the start point.
        # Fix that!
        intersections = {}
        for i in range(len(self.beziers)):
            ints = self.beziers[i].find_self_intersection() 
            if ints: 
                intersections[i] = ints
        # Pair the curves.
        pairs = itertools.combinations(enumerate(self.beziers), 2)
        # Remove pairs which do not have overlapping bounding boxes. 
        pairs = iter(pair for pair in pairs if pair[0][1].overlaps(pair[1][1]))
        for pair in pairs:
            results = pair[0][1].curve_intersections(pair[1][1], threshold)
            if results:
                intersections[pair[0][0], pair[1][0]] = results
        return intersections

    def intersections(self, other: 'Spline', threshold = INTERSECTION_THRESHOLD):
        """This should perhaps only intersect the two curves. 
        We already have self intersections via the other function."""

        """Self intersections: dict: 
        - Key (i), value: (tia, tib): the i:th Bezier intersects itself at tia and tib. 
        - Key (i, j), value: (tia, tib, tja, tjb): the i:th Bezier intersects the j:th Bezier at (tia approx tib)
        and (tja approx tjb)

        Intersectoins: dict:
        - Key (i, j), value (tia, tib, tja, tjb): the i:th Bezier of self intersect the j:th Bezier of other.
        """
        intersections = {}
        self_enumerate = enumerate(self.beziers)
        other_enumerate = enumerate(other.beziers)
        pairs = itertools.product(self_enumerate, other_enumerate)
        pairs = iter(pair for pair in pairs if pair[0][1].overlaps(pair[1][1])) 
        for pair in pairs: 
            result = pair[0][1].curve_intersections(pair[1][1], threshold)
            if result: 
                intersections[pair[0][0], pair[1][0]] = result
        return intersections

    def start_point(self, world_space = False):
        return self.beziers[0](0, world_space = world_space)

    def end_point(self, world_space = False):
        return self.beziers[-1](1, world_space = world_space)

    def split(self, i: int, t: float):
        """Splits the Spline at parameter t of curve i."""
        # TODO: Not needed so far. Remove if not.
        first = self.beziers[0:i]
        second = self.beziers[i+1:]
        splits = self.beziers[i].split2(t)
        first.append(splits[0])
        second.insert(0, splits[1])
        loc = self.location
        rot = self.rotation
        return [Spline(*first, location = loc, rotation = rot), Spline(*second, location = loc, rotation = rot)]

    def split_at_self_intersections(self):
        # 1. Calculate the self intersections.
        # 2. Split the curve into parts.
        # - If it is an internal self intersection at t: 
        #   we should split into two parts. 
        # - If it is an bez-bez intersection: 
        #   we should split into three parts. 
        #
        # The problem is what should be done when split the curve multiple times. 
        # The correct split points are then hard. 
        # Compile a dict(index, [t1, t2, t3...]) 
        # where ti are the intersection points of self.bezier[index]
        ints = self.self_intersections()
        compiled = {}
        for key in ints: 
            if isinstance(key, tuple): 
                if key[0] in compiled:
                    compiled[key[0]].extend(list(i[0] for i in ints[key]))
                else: 
                    compiled[key[0]] = list(i[0] for i in ints[key])
                if key[1] in compiled: 
                    compiled[key[1]].extend(list(i[1] for i in ints[key]))
                else: 
                    compiled[key[1]] = list(i[1] for i in ints[key])
            else: 
                if key in compiled: 
                    compiled[key].extend(ints[key])
                else: 
                    compiled[key] = ints[key]
        
        bezs = self.beziers
        curves: list[Bezier] = []
        print(5*'\n', compiled)

        # Keep the curves until first intersection.
        # Throw away all curves after the intersection.
        # Repeat.
        keep: int = 1
        print("Splits", compiled)
        for i, bez in enumerate(bezs):
            print("Curve ", i)
            if i in compiled:
                print("Splitting")
                splits = bez.split2(*compiled[i])
                print(len(splits), "pieces.")
                for s in splits[0:-1]: 
                    if keep > 0:
                        print("Keeping", i)
                        curves.append(s)
                    else:
                        print("Discarding split.")
                    keep *= -1
                if keep > 0:
                    print("Keeping last curve in split.")
                    curves.append(splits[-1])
                else:
                    print("Discarding last curve in split.")

            elif keep > 0:
                print("Keeping unsplitted.")
                curves.append(bez)
            else:
                print("Discarding unsplitted.")

        return Spline(*curves, location = self.location, rotation = self.rotation)

    def join_spline(self, mode):
        """Join another spline to the end of self. 
        mode determines how the joins are made in case self does not have continuous derivative."""


class Curve(CurveObject):
    """Curve object, container class for splines. Mirrors the Curve Object in Blender."""
    
    __slots__ = ("splines")

    # TODO: Add alternative constructor to import this from Blender.
    # TODO: When that is implemented, check if the location setter works properly.
    # In the end, every single Bezier within each Spline should have the same location.

    def __init__(self, *splines: Spline, 
                 name = "Curve", 
                 location: Vector = Vector(), 
                 scale: Vector = Vector(1.0, 1.0, 1.0),
                 rotation: Vector = Vector()
                 ):
        self.splines = list(splines)
        super().__init__(name = name, location = location, scale = scale, rotation = rotation)

    @classmethod
    def from_Blender(cls, name: str):
        splines = []
        cu = bpy.data.collections['Collection'].objects[name]
        loc = cu.location
        sca = cu.scale
        rot = cu.rotation_euler
        name = cu.name

        for spline in cu.data.splines:
            beziers = []
            is_closed = spline.use_cyclic_u #
            spline = cu.data.splines[0]
            bezier_points = spline.bezier_points

            # How can we do this better? 
            i = len(spline.bezier_points) - 1
            for j in range(0, i):
                handle_left = bezier_points[j].handle_left
                p0 = bezier_points[j].co
                p1 = bezier_points[j].handle_right
                p2 = bezier_points[j + 1].handle_left
                p3 = bezier_points[j + 1].co
                handle_right = bezier_points[j + 1].handle_right
                beziers.append(Bezier(p0, p1, p2, p3, 
                                      location = loc, 
                                      scale = sca, 
                                      rotation = rot,
                                      start_handle_left = handle_left, 
                                      end_handle_right = handle_right
                                      ))

            splines.append(Spline(*beziers, is_closed = is_closed, name = name + "_Spline", location = loc, scale = sca, rotation = rot))

        return cls(*splines, 
                   name = name, 
                   location = loc, 
                   scale = sca,
                   rotation = rot,
                  )

    @CurveObject.location.setter
    def location(self, location: Vector) -> None:
        # TODO: This is how it should be done, however, this creates a new Blender object for each spline! 
        # Rewrite the add_to_Blender so that we can reuse them. 
        for spline in self.splines:
            spline.location = location

    # TODO: Add CurveObject.rotation.setter 
    # TODO: Add CurveObject.scale.setter

    def add_spline(self, spline):
        """Add a new spline to the curve object.
        Since each spline defines a separate poly-Bezier,
        there is no need to check for matching start and end points."""
        self.splines.append(spline)

    def combine_curves(self, curve):
        """Combine two curves into one."""
        # TODO: Is this really needed?
        pass

    def offset(self, d):
        for spline in self.splines:
            spline.offset_spline(d)

    def intersections(self):
        pass

    def stroke(self):
        #self.endcap
        #self.bevellimit
        #...
        pass

    def add_to_Blender(self):
        cu = bpy.data.collections['Collection'].objects[self.name]
        for spline in self.splines:
            spline.add_to_Blender(blender_curve_object = cu)


class OffsetBezier():
    __slots__ = ("d",
                 "is_linear",
                 "rot_curve",
                 "angle",
                 "translation", 
                 "endpoint",
                 "angles",
                 "metrics",
                 "original_curve",
                 "sign",
                 "tolerance",
                 )
    """Calculates an cubic approximation of the offset of a cubic Bezier curve."""

    def __init__(self, bez: Bezier, d: float, sign = 1, tolerance = OFFSET_TOLERANCE):
        # TODO: Should do way less work! 
        # Just store the original curve and the distance.
        self.d = d
        self.sign = sign
        self.is_linear = False
        self.original_curve = bez
        self.rot_curve, self.angle, self.translation, self.endpoint = self._rotate_original_to_x()
        # No need to calculate the angles and the metrics for linear curves.
        if self.is_linear:
            self.angles = []
            self.metrics = dict()
        else:
            self.angles: list[float] = self._calculate_angles()
            self.metrics: dict[str, float] = self._calculate_metrics()
        self.tolerance = tolerance

    def eval_offset(self, t: float):
        """Evaluates the offset curve position at parameter t."""
        # TODO: Rename to eval_offset_curve since it evaluates the offset curve 
        # at parameter t.

        cur = self.rot_curve
        dp = cur.eval_derivative(t) 
        if dp.length() > 0.0:
            s = self.d / dp.length()
        else:
            s = 0.0
        # return cur.eval_offset(t, self.d) + cur(t)
        return Vector(-s * dp[1], s * dp[0], 0)

    def eval_offset_derivative(self, t: float):
        """Evaluates the derivative of the offset at parameter t."""
        cur: Bezier = self.rot_curve
        der = cur.derivative()
        dp = der(t)
        dpp = der.eval_derivative(t)
        k = 1.0 + (dpp.x * dp.y - dpp.y * dp.x) * self.d / dp.length()**3
        # k = 1.0 + dpp.cross(dp)[2] * self.d / (dp.length**3) 
        return k * dp

    def _rotate_original_to_x(self):
        """Calculates a new, transformed (rotated and translated) curve, such that the offset 
        of that will end up starting at (0,0) and ending on the x-axis (at endp).
        Returns the new Bezier, the rotation angle, the translation, and the endpoint..
        """
        # TODO: Rewrite this in terms of my own Vector() and Matrix() classes.
        d = self.d
        bez = self.original_curve
        p0 = bez.points[0].to_mu_vector()
        p1 = bez.points[1].to_mu_vector()
        p2 = bez.points[2].to_mu_vector()
        p3 = bez.points[3].to_mu_vector()
        # There are three problematic cases: 
        # 1. p1 = p0 and/or p2 = p3. Move the handle p1 [p2] to B(0.0001) [B(0.9999)]. Good enough.
        # 2. p0, p1, p2, and p3 are in a straight line. 
        #Handle offset directly (no need to calculate area, etc).
        l_linear = False # p1 = p0
        r_linear = False # p2 = p3
        # The dots product are used to determine if the points are collinear.
        dotp0123 = abs( (p0 - p1).x * (p3 - p2).x + (p0 - p1).y * (p3 - p2).y)
        l0123 = math.sqrt((p0 - p1).x**2 + (p0 - p1).y**2) * math.sqrt((p3 - p2).x ** 2 + (p3 - p2).y**2)
        dotp023 = abs((p0 - p2).x * (p2 - p3).x + (p0 - p2).y * (p2 - p3).y)
        l023 = math.sqrt((p0 - p2).x**2 + (p0 - p2).y**2) * math.sqrt((p2 - p3).x ** 2 + (p2 - p3).y**2)
        dotp013 = abs((p0 - p1).x * (p1 - p3).x + (p0 - p1).y * (p1 - p3).y)
        l013 = math.sqrt((p0 - p1).x**2 + (p0 - p1).y**2) * math.sqrt((p1 - p3).x ** 2 + (p1 - p3).y**2)

        # TODO: All of this has to be done via tolerances!
        if (p1 == p0 and p2 == p3) or dotp0123 == l0123: 
            self.is_linear = True
        if p0 == p1: 
            l_linear = True
            if self.is_linear:
                # Set the handle at a convenient point.
                p1 = bez(0.25).to_mu_vector()
            else:
                # Approximate handle.
                p1 = bez(.0001) .to_mu_vector()
        if p2 == p3: 
            r_linear = True
            if self.is_linear:
                # Set the handle at a convenient point.
                p2 = bez(0.75).to_mu_vector()
            else:
                # Make an approximate handle.
                p2 = bez(0.9999).to_mu_vector()
        
        if l_linear:
            h0 = (p1 - p0) / (p1 - p0).length
            r0 = p0 + d * mathutils.Vector((-h0.y, h0.x, 0))
        else: 
            r0 = p0 + d * bez.normal(0.0) # eval_offset vector of original curve.
        if r_linear: 
            h1 = (p3 - p2) / (p3 - p2).length
            r1 = p3 + d * mathutils.Vector((-h1.y, h1.x, 0))
        else:
            r1 = p3 + d * bez.normal(1.0)
        th = math.atan2( (r1 - r0)[1], (r1 - r0)[0])
        rot = mathutils.Matrix.Rotation(-th, 3, 'Z')
        q0 = rot @ (p0 - r0)
        q1 = rot @ (p1 - r0) if l_linear else rot @ (bez.points[1].to_mu_vector() - r0)
        q2 = rot @ (p2 - r0) if r_linear else rot @ (bez.points[2].to_mu_vector() - r0)
        q3 = rot @ (p3 - r0)
        b = Bezier(Vector.from_mu_vector(q0), Vector.from_mu_vector(q1), Vector.from_mu_vector(q2), Vector.from_mu_vector(q3))

        # Endpoint of the rotated offset (on the x-axis).
        endp = d * b.normal(1.0) + b(1.0).to_mu_vector()
        return b, th, r0, endp

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
        err = 0.0
        tol2 = self.tolerance**2
        # For each of the samples of the expected offset...
        for sample in samples:
            best_err: float = None # type: ignore
            # We find the corresponding point on the candidate cu. 
            samps = cu.intersect_ray(sample['p'], sample['d'])
            if len(samps) == 0:
                # No rays intersect, but be sample endpoints.
                samps = [0.0, 1.0]
            # Then we check to see the distance.
            for t in samps: 
                p_proj = cu(t)
                this_err = (sample['p'] - p_proj).length()**2
                if best_err is None or this_err < best_err:
                    best_err = this_err
            err = max(err, best_err)
            if err > tol2:
                break
        return math.sqrt(err)

    def _find_best_approximation(self):
        # TODO: Need to know rot_curve, angle, translation. 
        if not self.is_linear:
            candidates = self._find_cubic_candidates()
            samples = self._sample_points(10)
            best_curve: Bezier = None # type: ignore
            best_err: float = None # type: ignore
            errs: list[float] = []
            for i, cand in enumerate(candidates):
                err = self._estimate_cubic_error(cand, samples)
                errs.append(err)
                if best_curve is None or err < best_err:
                    best_err = err
                    best_curve = cand
        # Straight curve is easy to offset.
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
        # best_curve.add_to_Blender() 
        return {'curve': best_curve, 'error': best_err}

    def find_cubic_approximation(self) -> list[Bezier]:
        tolerance = self.tolerance
        approx = self._find_best_approximation()

        if approx['curve'] and approx['error'] <= tolerance:
            # print("Adding curve", approx['curve'])
            approx['curve'].name = self.original_curve.name
            # approx['curve'].add_to_Blender()
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
        mx: float = self.metrics["x_moment"]
        a: float = self.metrics["area"]
        x3: float = self.endpoint.x # The x-value of the endpoint.
        th0, th1 = self._calculate_angles()
        c0 = math.cos(th0)
        s0 = math.sin(th0)
        c1 = math.cos(th1)
        s1 = math.sin(th1)
        # The below is obtained by calculating the area and the x-moment in terms of d0, th0, d1, th1 and x3. 
        # We then solve for d1 from the equation for the area and plug that into the equation for the moment.
        # After some simplification we get a quartic equation for d0 with the coefficients below.
        k0 = 50*a**2*c1*s1*x3/21 + 50*a*s1**2*x3**3/21 - 4*mx*s1**2*x3**2
        k1 = -10*a**2*c0*c1*s1/7 + 10*a**2*c1**2*s0/7 - 34*a*c0*s1**2*x3**2/21 + 4*a*c1*s0*s1*x3**2/3 + 4*c0*mx*s1**2*x3 - 4*c1*mx*s0*s1*x3 - 8*s0*s1**2*x3**4/35
        k2 = -3*a*c0**2*s1**2*x3/14 + 2*a*c0*c1*s0*s1*x3/7 - a*c1**2*s0**2*x3/14 - c0**2*mx*s1**2 + 2*c0*c1*mx*s0*s1 + 3*c0*s0*s1**2*x3**3/14 - c1**2*mx*s0**2 - 9*c1*s0**2*s1*x3**3/70
        k3 = 3*a*c0**3*s1**2/14 - 3*a*c0**2*c1*s0*s1/7 + 3*a*c0*c1**2*s0**2/14 - c0**2*s0*s1**2*x3**2/35 + c0*c1*s0**2*s1*x3**2/70 + c1**2*s0**3*x3**2/70
        k4 = -3*c0**3*s0*s1**2*x3/280 + 3*c0**2*c1*s0**2*s1*x3/140 - 3*c0*c1**2*s0**3*x3/280
        # print("K", k4, k3, k2, k1, k0)
        sols = solvers.solve_quartic(k4, k3, k2, k1, k0)

        good_sols = [sol for sol in sols if isinstance(sol, float)]
        good_sols.extend([sol.real for sol in sols if isinstance(sol, complex)])
        n = 0
        cubics: list[Bezier] = []
        for d0 in good_sols:
            fac = -c0*d0*s1 + c1*d0*s0 + 2*s1*x3
            # print("fac", fac)
            if fac != 0.0:
                d1 = (20*a/3 - 2*d0*s0*x3)/(-c0*d0*s1 + c1*d0*s0 + 2*s1*x3)
            else:
                d1 = 0.0
            # In case d0 < 0, set d0 = 0 and d1 to be the intersection between the tangents. 
            # See https://raphlinus.github.io/curves/2022/09/09/parallel-beziers.html
            # for rationale (but basically experimentation leads to this choice).
            # TODO: Investigate if these curves are ever the best ones. 
            # If not we can discard them directly.
            # print("d pairs before", d0, d1)
            if d0 < 0.0:
                d0 = 0
                d1 = s0*x3/(c0*s1 - c1*s0) 
            # Same thing for d1 < 0 as above. 
            elif d1 < 0.0:
                d1 = 0
                d0 = s1*x3/(c0*s1 - c1*s0)
            # print("d pairs after", d0, d1)
            if d0 >= 0.0 and d1 >= 0.0:
                p0 = mathutils.Vector((0, 0, 0))
                p1 = mathutils.Vector((d0 * c0, d0 * s0, 0))
                p3 = mathutils.Vector((x3, 0, 0))
                p2 = p3 + mathutils.Vector((d1 * c1, d1 * s1, 0))
                b = Bezier(p0, p1, p2, p3)
                b.name = "Bezier" + str(n)
                n += 1
                cubics.append(b)
                # b.add_to_Blender() # For testing.
            # print("d0, d1", d0, d1, a, mx)
        return cubics

    def _calculate_metrics(self):
        """Calculates the x-moment, area, and arc length that the offset should have."""
        arclen = 0.0
        area = 0.0
        x_moment = 0.0
        co = GAUSS_LEGENDRE_COEFFS_32
        # Uses a Gauss-Legendre quadrature.
        # TODO: Read up on this.
        for i in range(0, len(co), 2): 
            t = 0.5 * (1 + co[i + 1])
            wi = co[i]
            dp = self.eval_offset_derivative(t)
            p = Vector.from_mu_vector(self.eval_offset(t)) + self.rot_curve(t)
            d_area = wi * dp[0] * p[1]
            arclen += wi * dp.length()
            area += d_area;
            x_moment += p.x * d_area; 

        return {"area": 0.5 * area, "length": 0.5 * arclen, "x_moment": 0.5 * x_moment}


class OffsetCurve():
    """Takes a curve and offsets each bezier curve and patches the Bezier together."""
    pass


# TODO: Move the is_linear to the bezier curve instead.

def ray_intersect(p0: mathutils.Vector, d0: mathutils.Vector, p1: mathutils.Vector, d1: mathutils.Vector): 
    # TODO: Figure out what this does exactly. 
    # Probably line-line inersection.
    det = d0.x * d1.y - d0.y * d1.x
    t = (d0.x * (p0.y - p1.y) - d0.y * (p0.x - p1.x)) / det
    return mathutils.Vector((p1.x + d1.x * t, p1.y + d1.y * t))
