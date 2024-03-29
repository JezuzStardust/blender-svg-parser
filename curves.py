# Copyright 2022 Google LLC
# Copyright 2022 Raph Levien
# Copyright 2023 Jens Zamanian

# DISCLAIMER: 
# A big part of this file is based on the ideas found in the blogpost: 
# https://raphlinus.github.io/curves/2022/09/09/parallel-beziers.html
# and parts of the code is adapted from the in the interactive
# demo on that page (source can be found at: raphlinus/raphlinus.github.io). 
# However, all the code have been rewritten in Python and almost
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
import numpy as np
from typing import Optional

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

def add_square(p: mathutils.Vector, r = 0.1):
    """Adds a square to Blender at position p with side r."""
    me = bpy.data.meshes.new('Square')
    x = mathutils.Vector((1,0,0))
    y = mathutils.Vector((0,1,0))
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
                 location = mathutils.Vector((0.0 ,0.0 ,0.0)),
                 scale = mathutils.Vector((1.0, 1.0, 1.0)),
                 rotation = mathutils.Euler((0.0, 0.0, 0.0),'XYZ')
                ):
        self.name = name
        self.location = location
        self.scale = scale
        self.rotation = rotation

    @property # Use property for location and rotation so that subclasses can do that do.
    def location(self):
        return self._location

    @location.setter
    def location(self, location: mathutils.Vector):
        self._location = location

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale: mathutils.Vector):
        self._scale = scale

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, rotation: mathutils.Euler):
        self._rotation = rotation


class QuadraticBezier():
    """Class to handle some functions of a quadratic Bezier curve.
    Used mainly for handling derivatives, etc, of a cubic Bezier."""
    __slots__ = ("points")

    def __init__(self, p0: mathutils.Vector, p1: mathutils.Vector, p2: mathutils.Vector):
        self.points = [p0, p1, p2]

    def __call__(self, t: float):
        p = self.points
        return p[0] * (1 - t)**2 + 2 * p[1] * (1 - t)*t + p[2] * t**2

    def eval_derivative(self, t: float):
        """Evaluates the derivative of the curve at parameter t."""
        p = self.points
        return -2 * p[0] * (1 - t) - 2 * p[1] * t + 2 * p[1] * (1 - t) + 2 * p[2] * t


class Bezier(CurveObject):
    """
    Bezier curve of 3rd order. p0, p1, p2, p3 are mathutils.Vector
    t0 and t1 are the parameter time at the start and the end of the 
    original curve for instances created as splits of larger curves.
    """
    __slots__ = ("points", 
                 "t0", "t1",
                 "start_handle_left",
                 "end_handle_right",
                 "is_closed" # In Blender, a single Bezier curve can be toggled closed (it is really handled as a spline).
                 )

    def __init__(self, 
                 p0: mathutils.Vector,
                 p1: mathutils.Vector,
                 p2: mathutils.Vector,
                 p3: mathutils.Vector,
                 start_handle_left: Optional[mathutils.Vector] = None,
                 end_handle_right: Optional[mathutils.Vector] = None, 
                 t0 = 0.0,
                 t1 = 1.0, 
                 is_closed = False,
                 name = "Bezier", 
                 location: mathutils.Vector = mathutils.Vector((0.0, 0.0, 0.0)),
                 scale = mathutils.Vector((1.0, 1.0, 1.0)),
                 rotation = mathutils.Euler((0.0, 0.0, 0.0), 'XYZ'),
                 ) -> None:
        """ 
        Initializes the cubic Bezier and sets its points and degree. 
        The points should be mathutils.Vectors of some fixed dimension.
        The number of points should be 3 or higher. 
        """
        super().__init__(name, location, scale, rotation)
        self.points = [p0, p1, p2, p3]
        # The dangling handles of a Bezier curve in Blender are not really part of a mathematical curve. 
        # Instead they belong to the previous or next Bezier in case of a poly-bezier curve. 
        # Since Blender uses them, it is better to keep them.
        self.start_handle_left = start_handle_left
        self.end_handle_right = end_handle_right
        self.is_closed = is_closed # In Blender, single Bezier curves can be toggled closed.
        
        # t0 and t1 give the parameter values of the parent curve in case this is created from a split. 
        # Needed for keeping track of intersections.
        # TODO: Might not need this with the new algorithm (but perhaps to find intersections).
        self.t0 = t0
        self.t1 = t1
        # self.handle_linear()

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
        start_handle_left = bezier_points[0].handle_left
        p0: mathutils.Vector = bezier_points[0].co
        p1: mathutils.Vector = bezier_points[0].handle_right
        p2: mathutils.Vector = bezier_points[1].handle_left
        p3: mathutils.Vector = bezier_points[1].co
        end_handle_right: mathutils.Vector = bezier_points[1].handle_right
        loc: mathutils.Vector = cu.location
        sca: mathutils.Vector = cu.scale
        rot: mathutils.Euler = cu.rotation_euler
        return cls(p0, p1, p2, p3, 
                   name = name,
                   location = loc, 
                   scale = sca,
                   rotation = rot,
                   start_handle_left = start_handle_left,
                   end_handle_right = end_handle_right,
                   is_closed = is_closed
                   )

    def handle_linear(self):
        """Handles the cases where either or both of the control handles
        coincides with the start or endpoints."""
        # TODO: Remove this. Handles that conicide with start/endpoints are 
        # only a problem for offsetting. 
        # Handle it there. 
        # Not used.
        p0 = self.points[0]
        p1 = self.points[1]
        p2 = self.points[2]
        p3 = self.points[3]
        if p1 == p0:
            print("Left linear")
            n = 2
            p1_new = self(1/2**n) - p0
            while p1_new.length > 0 and n < 100:
                n += 1
                p1_new = self(1/2**n) - p0
            self.points[1] = self(1/2**(n-1))
            print(n)
        if p2 == p3:
            print("Right linear")
            n = 2
            p2_new = self(1 - 1/2**n)
            while p2_new.length> 0 and n < 100:
                n += 1
                p2_new = p3 - self(1 - 1/2**n)
            print(n)
            self.points[2] = self(1 - 1/2**(n-1))

    def __repr__(self):
        """Prints the name of the together with all the points. """
        p = self.points
        string = '<' + self.name + '\n' 
        string += "p0: " + str(p[0]) + '\n'
        string += "p1: " + str(p[1]) + '\n'
        string += "p2: " + str(p[2]) + '\n'
        string += "p3: " + str(p[3]) + '\n'
        string += "start_handle_left: " + str(self.start_handle_left) + '\n'
        string += "end_handle_right: " + str(self.end_handle_right) + '>'
        return string

    def __call__(self, t: float, world_space: bool = False) -> mathutils.Vector:
        """ Returns the value at parameter t. 
        If world_space = False, the position is calculated relative 
        to the origin of the Bezier."""
        p: list[mathutils.Vector] = self.points
        pos = p[0] * (1 - t)**3 + 3 * p[1] * (1 - t)**2 * t + 3 * p[2] * (1 - t) * t**2 + p[3] * t**3
        if world_space: 
            if self.rotation.z != 0.0:
                rot = self.rotation.to_matrix() # type: ignore
                pos: mathutils.Vector = rot @ pos + self.location #type: ignore
                return pos
            else:
                return pos + self.location
        else:
            return pos
 
    def reverse(self):
        """Reverses the direction of the curve."""
        self.points = list(reversed(self.points))
        self.start_handle_left, self.end_handle_right = self.end_handle_right, self.start_handle_left

    def set_point(self, point: mathutils.Vector, i: int):
        """Sets the point with index i of the Bezier."""
        # TODO: Remove this.
        self.points[i] = point

    def translate_origin(self, vector: mathutils.Vector):
        """Translates the origin of the Bezier to the position given by 
        vector without changing the world position of the curve. 
        """
        dist = self.location - vector
        self.points = [p + dist for p in self.points]
        self.start_handle_left = self.start_handle_left +  dist
        self.end_handle_right = self.end_handle_right + dist
        self.location = vector

    def eval_derivative(self, t: float):
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
        if derivative.length > 0.0: 
            return derivative / derivative.length #mathutils.vector.length
        else:
            return derivative # mathutils.Vector((0,0,0))

    def normal(self, t: float):
        """Return the tangent rotated 90 degrees anti-clockwise."""
        # The real normal always points in the direction the curve is turning
        # so we should probably call this something else.
        der = self.eval_derivative(t) 
        unit_der = der / der.length
        return mathutils.Vector((-unit_der.y, unit_der.x, 0.0))

    def curvature(self, t):
        """Returns the curvature at parameter t."""
        # TODO: Can probably be removed. 
        d = self.eval_derivative(t)
        sd = self.eval_second_derivative(t)
        denom = math.pow(d.length, 3 / 2)
        return (d[0] * sd[1] - sd[0] * d[1]) / denom

    def area(self) -> float:
        """Returns the (signed) area of the curve."""
        p = self.points
        # Precalculated in terms of the points using Green's theorem.
        # TODO: This os not really used for anything. 
        # The area we need is the area of the sampled curve which is calculated
        # by GAUSS_LEGENDRE_COEFFS_32. REMOVE?
        x0 = p[0].x
        y0 = p[0].y
        x1 = p[1].x
        y1 = p[1].y
        x2 = p[2].x
        y2 = p[2].y
        x3 = p[3].x
        y3 = p[3].y
        
        area = 3 / 20 * (x0 * (-2 * y1 - y2 + 3 * y3) + x1 * (2 * y0 - y2 - y3) + x2 * (y0 + y1 - 2 * y3) + x3 * (-3 * y0 + y1 + 2 * y2))
        # area = 3 / 20 * (
        #                     p[0][0] * ( - 2 * p[1][1] - p[2][1] + 3 * p[3][1] ) 
        #                     + p[1][0] * ( 2 * p[0][1] - p[2][1] - p[3][1] ) 
        #                     + p[2][0] * ( p[0][1] + p[1][1] - 2 * p[3][1] )
        #                     + p[3][0] * ( - 3 * p[0][1] + p[1][1] + 2 * p[2][1] )
        #                 )
        return area

    def x_moment(self):
        """Calculate the x_moment of a curve that starts at the origin
        and ends on the x-axis."""
        # TODO: This is only relevant in this version for the offset. 
        # However, it is not used. Remove.
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
        if dp.length > 0.0:
            s = d / dp.length
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

    def split(self, t0: float, t1: Optional[float] = None):
        # TODO: Remove this
        """Splits the Bezier curve at the parameter(s) t0 (and t1). 
        In case just one parameter value is given, a list of two curves 
        is returned. 
        Else, a single curve, corresponding to the curve between 
        t0 and t1 is returned. 
        Based on: https://github.com/Pomax/BezierInfo-2 
        The code for this function is almost a straight translation
        of the JavaScript code in the ref above int Python.
        """
        loc = self.location
        sca = self.scale
        rot = self.rotation
        if t0 == 0.0 and t1 is not None: 
            return self.split(t1)[0]
        elif t1 == 1.0:
            return self.split(t0)[1]
        else: 
            p = self.points

            new1 = p[0] * (1 - t0) + p[1] * t0 
            new2 = p[1] * (1 - t0) + p[2] * t0 
            new3 = p[2] * (1 - t0) + p[3] * t0 
            new4 = new1 * (1 - t0) + new2 * t0 
            new5 = new2 * (1 - t0) + new3 * t0 
            new6 = new4 * (1 - t0) + new5 * t0

            # result[0].t0 = self.map_split_to_whole(0) 
            # result[0].t1 = self.map_split_to_whole(t0) 
            # result[1].t0 = self.map_split_to_whole(t0) 
            # result[1].t1 = self.map_split_to_whole(1) 

            t00 = self.map_split_to_whole(0) 
            t10 = self.map_split_to_whole(t0) 
            t01 = self.map_split_to_whole(t0) 
            t11 = self.map_split_to_whole(1) 


            result = [Bezier(p[0], new1, new4, new6, t0 = t00, t1 = t10, location = loc, scale = sca, rotation = rot), 
                      Bezier(new6, new5, new3, p[3], t0 = t01, t1 = t11, location = loc, scale = sca, rotation = rot)]

            # The new split curves should keep track for the original 
            # parameter values at the end points (t0 and t1).
            if t1 is None:
                return result
            else: 
                # Calculate which parameter of the split curve (result[1]) 
                # which corresponds to the point t1 on the whole curve. 
                # Then split again at that point. 
                t1p = self.map_whole_to_split(t1, t0, 1) 
        return result[1].split(t1p)[0] 

    def whole_parameter_to_split(self, t: float):
        ds = self.t0
        de = self.t1
        return (t - ds) / (de - ds)

    def subsegment(self, t0: float, t1: float):
        """Splits out the subsegment between t0 and t1. 
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

        # TODO: The below name is perhaps not so good.
        name = self.name + '(' + str(t0new) + ',' + str(t1new) + ')'
        loc = self.location
        sca = self.scale
        rot = self.rotation
        return Bezier(p0, p1, p2, p3,
                      t0 = t0new, t1 = t1new,
                      name = name,
                      location = loc,
                      scale = sca,
                      rotation = rot)

    def split2(self, *parameters: float):
        """Split the curve at parameters. Returns a list of the split segments."""
        # TODO: Rename this to split as soon as the other one is removed.
        ts = sorted([t for t in parameters])
        if ts[0] != 0.0: 
            ts.insert(0, 0.0)
        if ts[-1] != 1.0: 
            ts.append(1.0)
        
        sub_curves: list[Bezier] = []
        for i in range(1, len(ts)): 
            sub_curves.append(self.subsegment(ts[i-1], ts[i]))

        return sub_curves

    @staticmethod
    def map_whole_to_split(t, ds, de):
        """Returns the parameter in the splitted curve 
        corresponding to the parameter t of the whole (unsplitted) curve. 
        t1 is the parameter value which we want to map and 
        the split curve runs from parameter ds to de of the whole curve. 

        Ref: http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node13.html
        """
        return (t - ds) / (de - ds)
    
    def map_split_to_whole(self, t: float):
        """Returns the parameter value of the whole curve, 
        corresponding to the parameter t in the splitted curve. 
        """
        return self.t0 + (self.t1 - self.t0) * t

    def _create_Blender_curve(self):
        """Creates a new curve object in Blender."""
        # TODO: Catch the name of the object created by ob.name.
        # and store this for later reference?
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
        # 0      1      2      3
        # min_x, max_x, min_y, max_y 
        bb1 = self.bounding_box(world_space = True)
        bb2 = bezier.bounding_box(world_space = True)
        if bb1['min_x'] >= bb2['max_x'] or bb2['min_x'] >= bb1['max_x'] or bb1['min_y'] >= bb2['max_y'] or bb2['min_y'] >= bb1['max_y']:
            return False
        else:
            return True

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
        for i in range(0,n+1):
            t = i / n 
            ds = q(t).length 
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
        # d = self.d
        offsets = self.offset(d, double_side = True)
        left_offset = Spline(*offsets[0], location = self.location, rotation = self.rotation)
        right_offset = Spline(*offsets[1], location = self.location, rotation = self.rotation)
        left_offset.name = 'Left'
        right_offset.name = 'Right'
        left_offset.add_to_Blender()
        right_offset.add_to_Blender()

    def find_self_intersection(self):
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
                if (s >= 0.0 and s <= 1.0):
                    solutions.append(s)
                solutions.sort()
            if len(solutions) == 3:
                # The middle solution is a rouge solution.
                # The first and last are the two parameter values where it meets itself. 
                # Only need the first.
                return [solutions[0], solutions[2]]
        return None

    def find_intersections(self, bez):
        """Find the intersection between self and other bez.
        Returns a list of parameter value pairs (t, t') so that self(t) = bez(t').
        """
        # TODO: Not in use. use curve_intersections instead.
        treshold = 1e-12
        bb1 = self.bounding_box(world_space = True)
        bb2 = bez.bounding_box(world_space = True)
        if bb1['min_x'] >= bb2['max_x'] or bb2['min_x'] >= bb1['max_x'] or bb1['min_y'] >= bb2['max_y'] or bb2['min_y'] >= bb1['max_y']:
            return None
        else:
            if bb1['area'] * bb2['area'] < treshold:
                if (self.t0 < INTERSECTION_THRESHOLD or self.t1 > 1.0 - INTERSECTION_THRESHOLD) and (bez.t0 < INTERSECTION_THRESHOLD or bez.t1 > 1.0 - INTERSECTION_THRESHOLD):
                        return None
                return [(self.t0, self.t1, bez.t0, bez.t1)]
            else: 
                bb1s = self.split(0.5)
                bb2s = bez.split(0.5)
                inters = []
                a = bb1s[0].find_intersections(bb2s[0])
                b = bb1s[0].find_intersections(bb2s[1])
                c = bb1s[1].find_intersections(bb2s[0])
                d = bb1s[1].find_intersections(bb2s[1])
                if a:
                    inters.extend(a)
                if b:
                    inters.extend(b)
                if c:
                    inters.extend(c)
                if d:
                    inters.extend(d)

                if inters:
                    return inters
                else:
                    return None

    def curve_intersections(self, c2: 'Bezier', threshold = INTERSECTION_THRESHOLD):
        """Recursive method used for finding the intersection between two Bezier curves, c1, and c2.
        """
        # TODO: Better
        # Since the class Bezier is not defined at this point, 
        # we use 'Bezier' for the type annotation to make the typing system work properly. 

        # TODO: Check for endpoint matching. 
        # For now, we simply remove tuples where both values are sufficiently close to 0 or 1.
        # Probably we can save some time by checking this earlier. 
        # if self.points[3] == c2.points[0]:
        #     print("HOPSAN")
        #     print(self.points[3], c2.points[0])
        #     bez0 = self.split2(.9999)[0]
        #     c2 = c2.split2(.001)[1]
        # else:
        #     bez0 = self
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

### Start: Are these used?
    def fat_line(self):
        """Used for Bezier clipping: 
        T.W. Sederberg, T. Nishita, Curve intersection using Bezier clipping
        https://doi.org/10.1016/0010-4485(90)90039-F
        """
        p0 = self.points[0]
        p1 = self.points[1]
        p2 = self.points[2]
        p3 = self.points[3]
        t = p3 - p0
        # TODO: Check for zero curve?
        n = mathutils.Vector((-t.y, t.x, 0.0))
        n = n / n.length
        c = - p0.dot(n) 
        d1 = p1.dot(n) + c
        d2 = p2.dot(n) + c
        if d1 * d2 > 0: 
            k = 3 / 4
        else: 
            k = 4 / 9
        dmin = k * min(0, d1, d2)
        dmax = k * max(0, d1, d2)
        return dmin, dmax, n, c

    def fat_line_distance(self, bez: 'Bezier'):
        """Calculate the distance from the control points of self
        to the fat line of bez.
        """
        dmin, dmax, n, c = bez.fat_line()
        d0 = self.points[0].dot(n) + c 
        d1 = self.points[1].dot(n) + c 
        d2 = self.points[2].dot(n) + c 
        d3 = self.points[3].dot(n) + c 
        return d0, d1, d2, d3, dmin, dmax

    def intersect_fat_line(self, bez: 'Bezier'):
        d0, d1, d2, d3, dmin, dmax = self.fat_line_distance(bez)
        D0 = mathutils.Vector((0.0,   d0, 0.0))
        D1 = mathutils.Vector((1 / 3, d1, 0.0))
        D2 = mathutils.Vector((2 / 3, d2, 0.0))
        D3 = mathutils.Vector((1.0,   d3, 0.0))
        L0 = mathutils.Vector((0.0 ,dmin,0.0))
        L1 = mathutils.Vector((1.0 ,dmin,0.0))
        L2 = mathutils.Vector((0.0 ,dmax,0.0))
        L3 = mathutils.Vector((1.0 ,dmax,0.0))
        a = mathutils.geometry.intersect_line_line_2d(D0, D1, L0, L1)
        b = mathutils.geometry.intersect_line_line_2d(D0, D1, L2, L3)
        c = mathutils.geometry.intersect_line_line_2d(D0, D3, L0, L1)
        d = mathutils.geometry.intersect_line_line_2d(D0, D3, L2, L3)
        e = mathutils.geometry.intersect_line_line_2d(D1, D2, L0, L1)
        f = mathutils.geometry.intersect_line_line_2d(D1, D2, L2, L3)
        g = mathutils.geometry.intersect_line_line_2d(D2, D3, L0, L1)
        h = mathutils.geometry.intersect_line_line_2d(D2, D3, L2, L3)
        L = [a, b, c, d, e, f, g, h]
        L = [x[0] for x in L if x is not None]
        tmin = min(L)
        tmax = max(L)
        return tmin, tmax

    def find_intersections2(self, bez: 'Bezier'):
        """Hej"""
        # TODO: Abandon this version of bez-bez intersection? 
        # Instead we can check for bad cases, e.g. when the curve
        # is close to itself and remove these.
        ta1, tb1 = self.intersect_fat_line(bez)
        a = self.subsegment(ta1, tb1)
        ta2, tb2 = bez.intersect_fat_line(a)
        b = bez.subsegment(ta2, tb2)
        ta3, tb3 = a.intersect_fat_line(b)
        a1 = a.subsegment(ta3, tb3)
        ta4, tb4 = b.intersect_fat_line(a1)
        b1 = b.subsegment(ta4, tb4)
        print(ta4, tb4, b1.t0, b1.t1) 
        ta5, tb5 = a1.intersect_fat_line(b1)
        print(ta5, tb5)
# End: Are these used?

class Spline(CurveObject): 
    """A list of Bezier curves corresponds to a single spline object.
    For each Bezier, the end point coincide with the starting point of 
    the next curve."""
    # TODO: Handle offsets when the handles at a point are not aligned!
    # TODO: Handle endcaps.
    # TODO: Handle intersections between two splines.
    # TODO: Handle massaging of the offset curve so that all intersections are 
    # combined. 

    __slots__ = ('beziers', 'is_closed', 'strokewidth')

    def __init__(self, 
                 *beziers: Bezier, 
                 is_closed = False, 
                 strokewidth = 0.01,
                 name = "Spline",
                 location = mathutils.Vector((0.0 ,0.0, 0.0)),
                 scale = mathutils.Vector((1.0, 1.0, 1.0)),
                 rotation = mathutils.Euler((0,0,0), 'XYZ'),
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
        cu = bpy.data.collections['Collection'].objects[name]
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
    def location(self, location: mathutils.Vector):
        """Set the location in world space of the Spline and all the Bezier curves."""
        for bez in self.beziers:
            bez.location = location
        self._location = location

    @CurveObject.scale.setter
    def scale(self, scale: mathutils.Vector):
        """Sets the scale of the Spline and propagate it to all bezier curves."""
        for bez in self.beziers:
            bez.scale = scale
        self._scale = scale

    @CurveObject.rotation.setter
    def rotation(self, rot: mathutils.Euler):
        for bez in self.beziers:
            bez.rotation = rot
        self._rotation = rot

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
            self.beziers.insert(0,Bezier(p0, p1, p2, p3, loc, start_handle_left = shl, end_handle_right = ehr))
        
        # Move the start point of spline so that it matches the 
        # end point of self. 
        abez = self.beziers[a-1] # The last Bezier of spline.
        abez.points[3] = first_bez.points[0]
        abez.points[2] = first_bez.start_handle_left
        abez.end_handle_right = first_bez.points[1]
    
    def append_bezier(self, bezier):
        """
        Add a single Bezier curve in the end of the curve. 
        End and start points must match. 
        """
        # The check might need to be done within some precision.
        if self.end_point(world_space=True) == bezier.start_point(world_space=True):
            bezier.translate_origin(self.location) # Make the origins coincide.
            ep = self.end_point()
            bezier.set_point(ep, 0) # Common point is same instance of mathutils.Vector.
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
                 location = mathutils.Vector((0,0,0)), 
                 scale = mathutils.Vector((1.0, 1.0, 1.0)),
                 rotation = mathutils.Euler((0.0, 0.0, 0.0),'XYZ')
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
    def location(self, loc):
        # TODO: This is how it should be done, however, this creates a new Blender object for each spline! 
        # Rewrite the add_to_Blender so that we can reuse them. 
        for spline in self.splines:
            spline.location = loc

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
        if dp.length > 0.0:
            s = self.d / dp.length
        else:
            s = 0
        # return cur.eval_offset(t, self.d) + cur(t)
        return mathutils.Vector((-s * dp[1], s * dp[0], 0))

    def eval_offset_derivative(self, t: float):
        """Evaluates the derivative of the offset at parameter t."""
        cur: Bezier = self.rot_curve
        der = cur.derivative()
        dp = der(t)
        dpp = der.eval_derivative(t)
        k = 1.0 + (dpp.x * dp.y - dpp.y * dp.x) * self.d / dp.length**3
        # k = 1.0 + dpp.cross(dp)[2] * self.d / (dp.length**3) 
        return k * dp

    def _rotate_original_to_x(self):
        """Calculates a new, transformed (rotated and translated) curve, such that the offset 
        of that will end up starting at (0,0) and ending on the x-axis (at endp).
        Returns the new Bezier, the rotation angle, the translation, and the endpoint..
        """
        d = self.d
        bez = self.original_curve
        p0 = bez.points[0]
        p1 = bez.points[1]
        p2 = bez.points[2]
        p3 = bez.points[3]
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

        if (p1 == p0 and p2 == p3) or dotp0123 == l0123: 
            self.is_linear # TODO: What is this? Should it be self.is_linear = True?
        if p0 == p1: 
            l_linear = True
            if self.is_linear:
                # Set the handle at a convenient point.
                p1 = bez(0.25)
            else:
                # Approximate handle.
                p1 = bez(.0001) 
        if p2 == p3: 
            r_linear = True
            if self.is_linear:
                # Set the handle at a convenient point.
                p2 = bez(0.75)
            else:
                # Make an approximate handle.
                p2 = bez(0.9999)
        
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
        q0 = rot @ (bez.points[0] - r0)
        q1 = rot @ (p1 - r0) if l_linear else rot @ (bez.points[1] - r0)
        q2 = rot @ (p2 - r0) if r_linear else rot @ (bez.points[2] - r0)
        q3 = rot @ (bez.points[3] - r0)
        b = Bezier(q0, q1, q2, q3) #type: ignore
        # Endpoint of the rotated offset (on the x-axis).
        endp = d * b.normal(1.0) + b(1.0)
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
                arclen += co[j] * self.eval_offset_derivative(t).length
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
                this_err = (sample['p'] - p_proj).length**2
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
            p = self.eval_offset(t) + self.rot_curve(t)
            d_area = wi * dp[0] * p[1]
            arclen += wi * dp.length
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
