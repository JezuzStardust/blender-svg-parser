"""
Classes and utility functions for Bezier curves.  
"""
# Bezier: Contains all data and operations needed to define and work with a Bezier curve. 
# Spline: Contains a list of Bezier curves. 
# 1. How should we init this class? Either we init by passing all the points and the class creates and stores the Bezier instances, or we can init by passing pre-fabricated Bezier instances. 
# 2. Can we programme for both options? Either with wargs or kwargs. 
# Curves
# Again we need to think about how to init these. 

# TODO: Remove premature optimization. 
# import cProfile
# cProfile.run('<commands here as quoted string'>)

# Begin constants
# TODO: Which are needed?
THRESHOLD = 5e-5
TUPLE_FILTER_THRESHOLD = 1e-2
# End constants

from mathutils import Vector, Matrix
import mathutils 
import math
import bpy
import itertools
from . import solvers
from .gauss_legendre import GAUSS_LEGENDRE_COEFFS_32
from typing import Union, Optional
# import operator

##### Utility Functions #####
def add_line(a: mathutils.Vector, b: mathutils.Vector):
    """Add a line between a and b in Blender."""
    me = bpy.data.meshes.new('Line')
    ob = bpy.data.objects.new('Line', me)
    bpy.data.collections['Collection'].objects.link(ob)
    ob.data.vertices.add(2)
    ob.data.vertices[0].co = a
    ob.data.vertices[1].co = b
    ob.data.edges.add(1)
    ob.data.edges[0].vertices = (0,1) 
    ob.data.update(calc_edges_loose=True)

def add_square(p: mathutils.Vector, r = 0.1):
    me = bpy.data.meshes.new('Square')
    ob = bpy.data.objects.new('Square', me)
    bpy.data.collections['Collection'].objects.link(ob)
    x = Vector((1,0,0))
    y = Vector((0,1,0))
    ob.data.vertices.add(4)
    ob.data.vertices[0].co = p + r * (x + y) / 2
    ob.data.vertices[1].co = p + r * (x - y) / 2
    ob.data.vertices[2].co = p - r * (x + y) / 2
    ob.data.vertices[3].co = p - r * (x - y) / 2
    ob.data.edges.add(4)
    ob.data.edges[0].vertices = (0,1) 
    ob.data.edges[1].vertices = (1,2) 
    ob.data.edges[2].vertices = (2,3) 
    ob.data.edges[3].vertices = (3,0) 
    ob.data.update(calc_edges_loose=False)
"""
Numpy version of quadratic solver. 
I use my own from solvers.py
def quadratic_solve(a,b,c): 
     roots = np.roots([a,b,c])
     rot = []
     for root in roots:
         if np.isreal(root):
             rot.append(root)
     return tuple(rot)
 """

def curve_intersections(c1, c2, threshold = THRESHOLD):
    """Recursive method used for finding the intersection between 
    two Bezier curves, c1, and c2. 
    """
    # TODO: Make this a method of Bezier and/or Curve. 
    # It can have the signature _find_intersections_recursively(self, curve, threshold)
    # print('curve', c1.t0, c1.t1, c2.t0, c2.t1)
    if c1.t1 - c1.t0 < threshold and c2.t1 - c2.t0 < threshold:
        return [((c1.t0 + c1.t1)/2 , (c2.t0 + c2.t1)/2)]
        # return [(c1.t0, c2.t0)]

    cc1 = c1.split(0.5)
    cc2 = c2.split(0.5)
    pairs = itertools.product(cc1, cc2)

    # pairs = list(filter(lambda x: are_overlapping(x[0].bounding_box(world_space = True), x[1].bounding_box(world_space = True)), pairs))
    pairs = list(filter(lambda x: x[0].overlaps(x[1]), pairs))
    # print(pairs)
    results = [] 
    if len(pairs) == 0:
        return results
    
    for pair in pairs:
        results += curve_intersections(pair[0], pair[1], threshold)
    results = filter_duplicates(results)
    return results

def filter_duplicates(tuples, threshold = TUPLE_FILTER_THRESHOLD):
    """
    Filter out tuples that differ less than threshold.
    """
    result = []
    for tup in tuples:
        if not any(_is_close(tup, other, threshold) for other in result):
            result.append(tup)
    return result

def _is_close(a, b, threshold = TUPLE_FILTER_THRESHOLD):
    """Checks if two tuples a, and b, differ less then threshold."""
    comparisons = all(math.isclose(*c, abs_tol = threshold) for c in zip(a,b))
    return comparisons

def is_colinear(v1, v2, threshold = 0.00000001):
    return v1.cross(v2).length < threshold

def are_overlapping(bbbox1, bbbox2):
    """
    Check if two bounding boxes are overlapping.
    """
    # TODO: Consider redoing this.
    # 0      1      2      3
    # min_x, max_x, min_y, max_y 
    if bbbox1[0] >= bbbox2[1] or bbbox2[0] >= bbbox1[1] or bbbox1[2] >= bbbox2[3] or bbbox2[2] >= bbbox1[3]:
        return False
    else:
        return True

##### End Utility Functions #####

class Point():
    """Wrapper to mathutils.Vector."""
    pass


class CurveObject():
    """Base class for all curves."""
    __slots__ = ("name", "_location")
    # TODO: Perhaps it makes more sense to make this the
    # TODO: Add rotation?
    # base class of only Spline and Curve.
    # The both have a lot in common.
    # Methods:
    # - boundary box
    # - intersections 
    # - self intersections
    # - create blender curve object
    
    def __init__(self, name = "Curve Object", location = Vector((0,0,0))):
        self.name = name
        self.location = location
        
    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, loc: mathutils.Vector):
        self._location = loc


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
    t0 and t1 are the parameter time at the start and the end. 
    """
    __slots__ = ("points", 
                 "t0", "t1",
                 "start_handle_left",
                 "end_handle_right",
                 "left_offset",
                 "right_offset"
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
                 name = "Bezier", 
                 location = Vector((0,0,0))
                 ) -> None:
        """ 
        Initializes the cubic Bezier and sets its points and degree. 
        The points should be mathutils.Vectors of some fixed dimension.
        The number of points should be 3 or higher. 
        """
        super().__init__(name, location)
        self.points = [p0, p1, p2, p3]
        # The dangling handles of a Bezier curve in Blender are not really part of a mathematical curve. 
        # Instead they belong to the previous or next Bezier in case of a poly-bezier curve. 
        # Since Blender uses them, it is better to keep them.
        self.start_handle_left = start_handle_left or p0
        self.end_handle_right = end_handle_right or p3
        
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
        bezier_points = cu.data.splines[0].bezier_points
        start_handle_left = bezier_points[0].handle_left
        p0: mathutils.Vector = bezier_points[0].co
        p1: mathutils.Vector = bezier_points[0].handle_right
        p2: mathutils.Vector = bezier_points[1].handle_left
        p3: mathutils.Vector = bezier_points[1].co
        end_handle_right: mathutils.Vecor = bezier_points[1].handle_right
        loc: mathutils.Vector = cu.location
        return cls(p0, p1, p2, p3, name=name, location=loc, 
                      start_handle_left = start_handle_left,
                      end_handle_right = end_handle_right)

    def handle_linear(self):
        """Handles the cases where either or both of the control handles
        coincides with the start or endpoints."""
        # TODO: Instead of doing this, sample the curve and create an approximation instead.
        p0 = self.points[0]
        p1 = self.points[1]
        p2 = self.points[2]
        p3 = self.points[3]
        print("Points prior", p1, p2)
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
        print("Points after", self.points[1], self.points[1].x, self.points[1].y, self.points[2])

    def __repr__(self):
        """Prints the name of the together with all the points. """
        p = self.points
        string = self.name + '\n' 
        string += "p0: " + str(p[0]) + '\n'
        string += "p1: " + str(p[1]) + '\n'
        string += "p2: " + str(p[2]) + '\n'
        string += "p3: " + str(p[3]) + '\n'
        string += "start_handle_left: " + str(self.start_handle_left) + '\n'
        string += "end_handle_right: " + str(self.end_handle_right)
        return string

    def __call__(self, t: float, world_space: bool = False):
        """ Returns the value at parameter t. 
        If world_space = False, the position is calculated relative 
        to the origin of the Bezier."""
        p = self.points
        pos = p[0] * (1 - t)**3 + 3 * p[1] * (1 - t)**2 * t + 3 * p[2] * (1 - t) * t**2 + p[3] * t**3
        if world_space: 
            return pos + self.location
        else:
            return pos
 
    def reverse(self):
        """Reverses the direction of the curve."""
        self.points = list(reversed(self.points))
        self.start_handle_left, self.end_handle_right = self.end_handle_right, self.start_handle_left

    def set_point(self, point: mathutils.Vector, i: int):
        """Sets the point with index i of the Bezier."""
        self.points[i] = point

    def translate_origin(self, vector: mathutils.Vector):
        """Translates the origin of the Bezier to vector without changing the 
        world position of the curve. 
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
        """Calculate the tangent at parameter t."""
        derivative = self.eval_derivative(t) 
        if derivative.length > 0.0: 
            return derivative / derivative.length #mathutils.vector.length
        else:
            return derivative # Vector((0,0,0))

    def normal(self, t: float):
        a = self.tangent(t)
        return Vector((a[1], -a[0], 0))

    def curvature(self, t):
        """Returns the curvature at parameter t."""
        d = self.eval_derivative(t)
        sd = self.eval_second_derivative(t)
        denom = math.pow(self.eval_derivative(t).length, 3 / 2)
        return (d[0] * sd[1] - sd[0] * d[1]) / denom

    def area(self) -> float:
        """Returns the (signed) area of the curve."""
        p = self.points
        # Precalculated in terms of the points using Green's theorem.
        x0 = p[0][0]
        y0 = p[0][1]
        x1 = p[1][0]
        y1 = p[1][1]
        x2 = p[2][0]
        y2 = p[2][1]
        x3 = p[3][0]
        y3 = p[3][1]
        
        area = 3 / 20 * (
                            x0 * (-2 * y1 - y2 + 3 * y3) 
                            + x1 * (2 * y0 - y2 - y3) 
                            + x2 * (y0 + y1 - 2 * y3)
                            + x3 * (-3 * y0 + y1 + 2 * y2)
                        )
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
        # TODO: This is only relevant in this version for the offset. Move there?
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
        # TODO: Figure out how to deal with the case where one of the handles
        # have length zero
        n = None
        dp = self.eval_derivative(t) 
        if dp.length > 0.0:
            s = d / dp.length
        else:
        # If the derivative is zero we approximate the derivative.
        # Can this happen at some other place than at t = 0 or t = 1?
            p0 = self.points[0]
            p3 = self.points[3]
            n = 10
            if t == 0.0:
                print("Went into t = 0")
                dp = p3 - p0
                while dp.length > 0.0 and n < 100:
                    n += 1
                    dp = self(1 / 2**n) - p0
                dp = self(0.5 / (n - 1)) - p0   
                s = d / dp.length 
            elif t == 1.0:
                print("Went into t = 1")
                dp = p3 - self(0.5)
                while dp.length > 0.0 and n < 100:
                    n += 1
                    dp = p3 - self(1.0 - 1.0 / 2**n)
                dp = p3 - self(1.0 - 0.5 / (n - 1))
                s = d / dp.length 
            else:
                s = 0
                print("The derivative was zero at t in (0, 1), not sure how this happened.")

        return mathutils.Vector((-s * dp[1], s * dp[0], 0))

    def rotate_to_x(self):
        """Rotates the curve so that the offset of this rotated curve, 
        will end up with the first point at x = y = 0 and the 
        second point will be on the y-axis."""
        # TODO: Only relevant in offset. Move there?
        r0 = self.points[0] + self.eval_offset(0, 1)
        r1 = self.points[3] + self.eval_offset(1, 1)
        th = math.atan2( (r1 - r0)[1], (r1 - r0)[0])
        rot = mathutils.Matrix.Rotation(-th, 3, 'Z')
        q0 = rot @ ( self.points[0] - r0 )
        q1 = rot @ ( self.points[1] - r0 )
        q2 = rot @ ( self.points[2] - r0 )
        q3 = rot @ ( self.points[3] - r0 )
        b = Bezier(q0, q1, q2, q3) # type: ignore
        # TODO: Should return also the rotation. 
        return b

    def aligned(self):
        """Returns the points of the corresponding aligned curve. 
        Aligned means: start point in origin, end point on x-axis. 
        """
        m = Matrix.Translation(-self.points[0])
        end = m @ self.points[3]
        if end[0] != 0.0:
            angle = -math.atan2(end[1],end[0])
        else:
            angle = 0.0
        k = Matrix.Rotation(angle, 4, 'Z') @ m

        aligned_points = []
        for p in self.points:
            aligned_points.append(k @ p)
        return aligned_points

    def bounding_box(self, world_space = False):
        """Calculate the bounding box of the curve."""
        # TODO: Make it possible to calculate the tight bounding box if this
        # is deemed useful. 
        # TODO: Consider using a Rectangle class for this?
        # Need to calculate the rotation and translation also! 
        extrema = self.extrema()
        min_x = self.__call__(extrema[0], world_space)[0]
        max_x = self.__call__(extrema[1], world_space)[0]
        min_y = self.__call__(extrema[2], world_space)[1]
        max_y = self.__call__(extrema[3], world_space)[1]
        return {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}
        # return (min_x, max_x, min_y, max_y)

    def extrema(self):
        """
        Returns the parameter values for the minimum and maximum of the curve
        in the x and y coordinate. 
        (tx_min, tx_min, ty_max, ty_max)
        If absolute we return the extremas of the aligned curve.
        """
        # TODO: This must take the rotation into account when that is added.
        p0, p1, p2, p3 = self.points

        a = 3 * (-p0 + 3 * p1 - 3 * p2 + p3)
        b = 6 * (p0 - 2*p1 + p2)
        c = 3*(p1 - p0) 
        
        # TODO: Clean up using e.g. itertools. 
        endpoints = (0.0, 1.0) # Must also check if extrema occurs on endpoint.
        tx_roots = endpoints + solvers.solve_quadratic(a[0], b[0], c[0]) 
        ty_roots = endpoints + solvers.solve_quadratic(a[1], b[1], c[1]) 

        tx_roots = [t for t in tx_roots if 0.0 <= t <= 1.0] 
        ty_roots = [t for t in ty_roots if 0.0 <= t <= 1.0] 

        x_values = [self.__call__(t, world_space = True)[0] for t in tx_roots] 
        y_values = [self.__call__(t, world_space = True)[1] for t in ty_roots] 

        tx_max = tx_roots[x_values.index(max(x_values))]
        tx_min = tx_roots[x_values.index(min(x_values))]
        ty_max = ty_roots[y_values.index(max(y_values))]
        ty_min = ty_roots[y_values.index(min(y_values))]

        return [tx_min, tx_max, ty_min, ty_max]

    def inflection_points(self, threshold = .02):
        """Returns a list of parameter values where inflection points occurs. 
        The length of the list can be zero, one, or two. 
        """
        # TODO: Align the curve to the x-axis to simplify equations. 
        # https://pomax.github.io/bezierinfo/#inflections
        p0, p1, p2, p3 = self.aligned()
        # TODO: Itertools?
        a = p2[0] * p1[1]
        b = p3[0] * p1[1]
        c = p1[0] * p2[1]
        d = p3[0] * p2[1] 
        e = 3 * a + 2 * b + 3 * c - d 
        f = 3 * a - b - 3 * c
        g = c - a 
        # TODO: Fix the below! The new quadratic does not handle a coefficient in front of x**2.
        # TODO: Also check the order of the coefficients inserted.
        # TODO: Finally 
        # inflection_points = solvers.solve_quadratic(e, f, g) 
        # inflection_points = [p for p in inflection_points if isinstance(p, float) and p >= 0.0 and p <= 1.0]
        inflection_points = None
        return inflection_points

    def reduced(self):
        """Splits the curve at each extrema and each inflection point
        and returns a list of these. 
        """
        # TODO: Remove extremas that are "too close". 
        # Should compare all values pairwise and remove one of every pair that is 
        # too close to another value. 
        # Create new empty list. 
        # Add 0.0 and 1.0. 
        # For all extremas, we add them only if they are not too close to 
        # any of the other values. 
        extrema = self.extrema()
        inflections = self.inflection_points()
        total = list(set(extrema + extrema + inflections)) # Remove any doubles. 
        total.sort() 
        total = [i for i in total if i > 0.001 and i < 0.999]
        if 0.0 not in total:
            total.insert(0, 0.0)
        if 1.0 not in total:
            total.append(1.0)

        curves = []
        for i in range(len(total)-1):
            curves.append(self.split(total[i], total[i+1]))
        return curves

    def simplified(self):
        """Reduces the curve and then make sure that all reduced parts 
        are ready to be offset."""
        beziers = self.reduced()
        # for bezier in self.reduced():
        # #     beziers += bezier.split(0.5)
        # beziers = self.split(.5)

        all_simple = False
        
        while not all_simple:
            all_simple = True
            new_set = []
            for bez in beziers:
                if bez._is_offsetable():
                    new_set.append(bez)
                else:
                    all_simple = False
                    new_set += bez.split(0.5)
            beziers = new_set

        return beziers

    def _is_offsetable(self, threshold = 0.02):
        """Check that Bezier(0.5) is not too far from the center 
        of the bounding box defined by the Bezier.points. 
        If the curve is straight, then we always return True. 
        """
        mid = self.__call__(0.5)
        mid.resize_2d()
        p = self.points
        # d = .22
        # p0 = p[0]
        # p3 = p[3]
        # n0 = self.normal(0)
        # n1 = self.normal(1)
        # a0 = p0 - n0 * d
        # a3 = p3 - n1 * d


        a = (p[3] + p[0])/2
        b = (p[2] + p[1])/2 
        c = (p[3] + p[2])/2
        d = (p[1] + p[0])/2
        if is_colinear(a-b, c-d):
            return True # Straight curves are always good.
        else:
            # TODO: Can we skip converting to 2D?
            a.resize_2d()
            b.resize_2d()
            c.resize_2d()
            d.resize_2d()
            ints = mathutils.geometry.intersect_line_line_2d(a, b, c, d)
            jam = max((a-b).length, (c-d).length)
            return (ints-mid).length / jam < threshold

    def start_point(self, world_space = False):
        """Returns the starting point."""
        if world_space:
            return self.points[0] + self.location
        else:
            return self.points[0]

    def end_point(self, world_space = False):
        """Returns the end point."""
        if world_space:
            return self.points[3] + self.location
        else: 
            return self.points[3]

    def split(self, t0: float, t1: Optional[float] = None):
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
        if t0 == 0 and t1 is not None: 
            return self.split(t1)[0]
        elif t1 == 1:
            return self.split(t0)[1]
        else: 
            p = self.points

            new1 = p[0] * (1 - t0) + p[1] * t0 
            new2 = p[1] * (1 - t0) + p[2] * t0 
            new3 = p[2] * (1 - t0) + p[3] * t0 
            new4 = new1 * (1 - t0) + new2 * t0 
            new5 = new2 * (1 - t0) + new3 * t0 
            new6 = new4 * (1 - t0) + new5 * t0

            result = [Bezier(p[0], new1, new4, new6, location=loc), Bezier(new6, new5, new3, p[3], location=loc)]

            # The new split curves should keep track for the original 
            # parameter values at the end points. 
            result[0].t0 = self.map_split_to_whole(0) 
            result[0].t1 = self.map_split_to_whole(t0) 
            result[1].t0 = self.map_split_to_whole(t0) 
            result[1].t1 = self.map_split_to_whole(1) 

            if not t1:
                return result
            else: 
                # Calculate which parameter of the split curve (result[1]) 
                # which corresponds to the point t1 on the whole curve. 
                # Then split again at that point. 
                t1p = self.map_whole_to_split(t1, t0, 1) 
        return result[1].split(t1p)[0] 

    def map_whole_to_split(self, t, ds, de):
        """Returns the parameter in the splitted curve 
        corresponding to the parameter t of the whole (unsplitted) curve. 
        t1 is the parameter value which we want to map and 
        the split curve runs from parameter ds to de of the whole curve. 

        Ref: http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node13.html
        """
        # TODO: This does not really depend on self. Move to utility.
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

    def intersections(self, bezier, threshold = THRESHOLD):
        """Returns a list of the parameters [(t, t'), ...] for the intersections 
        between self and bezier.
        """
        c1 = self.reduced()
        c2 = bezier.reduced()
        pairs = itertools.product(c1, c2)
        
        pairs = [pair for pair in pairs if pair[0].overlaps(pair[1])]

        intersections = []
        for pair in pairs:
            result = curve_intersections(*pair, threshold)
            if len(result) > 0:
                intersections += result
        return intersections

    def self_intersections(self, threshold = THRESHOLD):
        """Returns a list of self intersections of the curve."""
        # TODO: This can be combined with intersections to one function 
        # that has two different functions. 
        # However, it is probably easier to keep it like this, 
        # since this makes it easier to know which intersections are which. 
        c = self.simplified()
        pairs = itertools.combinations(c, 2)
        pairs = [
            pair for pair in pairs 
            if pair[0].overlaps(pair[1])
        ]
        intersections = []
        for pair in pairs:
            result = curve_intersections(*pair, threshold)
            if len(result) > 0 and not math.isclose(*result[0], abs_tol = TUPLE_FILTER_THRESHOLD):
                intersections += result
        return intersections

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

    def offset_curve(self, d):
        """Returns an inner and outer curve each offset a distance d from self."""
        # TODO: Clean up. More or less the same thing is done for left and right. 
        left_offset = []
        right_offset = []
        m = Matrix().Rotation(-math.pi/2, 3, 'Z') # TODO: Move to constants.
        beziers = self.simplified()

        # beziers = [self]

        loc = self.location
        i = 0
        j = 0
        while i < len(beziers) and j < 100: #max iterations
            j += 1
            bezier = beziers[i]
            p = bezier.points
            n0 = bezier.normal(0) # Normal at t = 0
            n1 = bezier.normal(1) # Normal at t = 1 

            nm = m @ (p[2] - p[1])
            nm.normalize()
            
            # Left offset 
            # Reuse the point of the last point. They should be the same. 
            if i != 0:
                a0 = left_offset[i-1].points[-1] 
            else:
                a0 = p[0] - n0 * d
            a1p = p[1] - n0 * d 
            a1mp = p[1] - nm * d
            a2p = p[2] - n1 * d
            a2mp = p[2] - nm * d
            a3 = p[3] - n1 * d

            a1 = None
            a2 = None
            if is_colinear(a1p - a0, a2mp - a1mp):
                a1 = a1mp
            if is_colinear(a2mp - a1mp, a2p - a3):
                a2 = a2mp

            if a1 is None:
                a1i = mathutils.geometry.intersect_line_line(a0, a1p, a1mp, a2mp)
                a1 = a1i[0]
            if a2 is None: 
                a2i = mathutils.geometry.intersect_line_line(a3, a2p, a1mp, a2mp)
                a2 = a2i[0]


            # if mathutils.geometry.intersect_line_line_2d(p[0], a1, p[3], a3):
            #     a1 = 2 * a0 - a1
            #     a2 = 2 * a3 - a2

            # Right offset
            # Reuse the point of the last point. They should be the same. 
            if i != 0:
                b0 = right_offset[i-1].points[-1]
            else:
                b0 = p[0] + n0 * d
            b1p = p[1] + n0 * d 
            b1mp = p[1] + nm * d
            b2p = p[2] + n1 * d
            b2mp = p[2] + nm * d
            b3 = p[3] + n1 * d

            b1 = None
            b2 = None

            if is_colinear(b1p - b0, b2mp - b1mp):
                b1 = b1mp
            if is_colinear(b2mp - b1mp, b2p - b3):
                b2 = b2mp

            if b1 is None:
                b1i = mathutils.geometry.intersect_line_line(b0, b1p, b1mp, b2mp)
                b1 = b1i[0]
            if b2 is None: 
                b2i = mathutils.geometry.intersect_line_line(b3, b2p, b1mp, b2mp)
                b2 = b2i[0]
            a = Bezier(a0, a1, a2, a3, location=loc)
            b = Bezier(b0, b1, b2, b3, location=loc)

            # add_line(a0,p[0])
            # add_line(a1,p[1])
            # add_line(a2,p[2])
            # add_line(a3,p[3])

            left_offset.append(a)
            right_offset.append(b)
            i += 1

        self.left_offset = Spline(*left_offset, name = self.name + ': Left', location = loc)
        self.right_offset = Spline(*right_offset, name = self.name + ': Right', location = loc)

        # self.left_offset.add_to_Blender()
        # self.right_offset.add_to_Blender()

        return self.left_offset, self.right_offset

    def is_clockwise(self):
        """Return True if the curve is clockwise."""
        # TODO: Not used. Remove?
        a = self.points[1] - self.points[0]
        a.resize_2d()
        b = self.points[3] - self.points[0]
        b.resize_2d()
        if a.length == 0.0 or b.length == 0.0: # TODO: Check if colinear instead.
            return True
        if a.angle_signed(b, None) >= 0.0:
            return True
        else:
            return False

    def transform(self, angle = 0.0, translation = Vector((0,0,0)), world_space = False):
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
        # print("C", c0, c1, c2, c3)
        sols: list[float] = []
        if c3 != 0.0:
            # print("cubic")
            qs = solvers.solve_cubic(c2 / c3, c1 / c3, c0 / c3)
            # print(qs)
            sols: list[float] = [s for s in qs if isinstance(s, float)]
        elif c2 != 0.0: 
            qs = solvers.solve_quadratic(c1 / c2, c0 / c2)
            sols: list[float] = [s for s in qs if isinstance(s, float)]
        elif c1 != 0.0:
            sols = [c0 / c1]
        ts = [t for t in sols if t > 0.0 and t < 1.0]
        return ts


class Spline(CurveObject): 
    """A list of Bezier curves corresponds to a single spline object.
    For each Bezier, the end point coincide with the starting point of 
    the next curve."""
    # TODO: Handle closed curves. 
    # Strategy?
    # 1. End with z. -> Always toggle closed. 
    # 2. End point = start point but does not end with z. -> Toggle closed 
    #    only if filled.
    # 3. End points different and z not set. -> Toggle closed only if filled. 
    # The above should probably not be done here, but in any code that uses
    # this class.
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
                 location = mathutils.Vector((0,0,0))
                 ):

        self.beziers = list(beziers)
        # Ensure that the end point and handles of one point, coincides with the corresponding for the next point.
        prev_bez = None
        for bez in self.beziers:
            if not prev_bez:
                continue
            bez.points[0] = prev_bez.points[3]
            bez.start_handle_left = prev_bez.points[2]
            prev_bez = bez
        self.is_closed = is_closed
        self.strokewidth = strokewidth

        for bezier in self.beziers:
            # bezier.name = self.name + ':' + bezier.name # Not really useful now.
            bezier.t0 = 0.0
            bezier.t1 = 1.0

        super().__init__(name, location)

    @classmethod
    def from_Blender(cls, name: str):
        """Alternative constructor where the Spline is imported from Blender."""
        cu = bpy.data.collections['Collection'].objects[name]
        beziers = []
        loc = cu.location
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
            beziers.append(Bezier(p0, p1, p2, p3, location = loc, 
                                  start_handle_left = handle_left, 
                                  end_handle_right = handle_right
                                  ))

        return cls(*beziers, 
                   name=name, 
                   location=loc, 
                   is_closed = is_closed,
                  )

    @CurveObject.location.setter
    def location(self, loc):
        """Set the location in world space of the Spline and all the Bezier curves."""
        for bez in self.beziers:
            bez.location = loc
        self._location = loc

    def reverse(self):
        for bez in self.beziers:
            bez.reverse()
        self.beziers = list(reversed(self.beziers))

    def append_spline(self, spline):
        """
        Add curve to the this curve at the end. 
        The start point and handles of spline will be
        moved to match with self's endpoint
        """
        a = len(self.beziers)
        last_bez = self.beziers[-1]
        loc = self.location
        s_loc = spline.location
        # Make new Bezier curves, since we do not want to modify spline.
        for bez in spline.beziers: 
            p0 = bez.points[0] + s_loc - loc
            p1 = bez.points[1] + s_loc - loc
            p2 = bez.points[2] + s_loc - loc
            p3 = bez.points[3] + s_loc - loc
            shl = bez.start_handle_left
            ehr = bez.end_handle_right
            self.beziers.append(Bezier(p0, p1, p2, p3, loc, start_handle_left = shl, end_handle_right = ehr))
        
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
            bezier.set_point(ep, 0) # Common point is same instance of Vector.
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
        """
        Toggles the curve closed.
        """
        self.closed = not self.closed 

    def _create_Blender_curve(self):
        cu = bpy.data.curves.new(self.name, 'CURVE')
        ob = bpy.data.objects.new(self.name, cu)
        ob.location = self.location
        bpy.data.collections["Collection"].objects.link(ob)
        ob.data.resolution_u = 64
        cu.splines.new('BEZIER')
        return cu

    def add_to_Blender(self, blender_curve_object = None):
        """
        Adds the curve to Blender as splines. 
        """
        # TODO: How should we choose which collection to add the curve to? 
        # TODO: Howe can we do this so that we reuse the add_to_Blender from Bezier?
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

    def self_intersections(self, threshold = THRESHOLD):
        """
        Find the intersections within the spline.
        The result is a dict e.g. {2 : [(t, t'), ...], (3, 4): [(...)]}
        where e.g. dict[2] gives the parameter values where self.beziers[2]
        intersect itself, and dict[(3,4)] (or dict[3,4]) gives a list of tuples of 
        parameter values where self.beziers[3] intersects self.beziers[4]. 
        """
        intersections = {}
        # TODO: Filter the result for doubles. 
        for i in range(len(self.beziers)):
            ints = self.beziers[i].self_intersections() 
            if ints: 
                intersections[i] = ints
        
        pairs = itertools.combinations(enumerate(self.beziers), 2) # Pair the curves. 
        pairs = [pair for pair in pairs if pair[0][1].overlaps(pair[1][1])] # Remove pairs which do not have overlapping bounding boxes. 
        for pair in pairs:
            results = pair[0][1].intersections(pair[1][1], threshold)
            if results:
                intersections[pair[0][0], pair[1][0]] = results
        return intersections

    def start_point(self, world_space = False):
        # TODO: Make this a lazy property.
        """Return the starting point of the curve."""
        return self.beziers[0].start_point(world_space) 

    def end_point(self, world_space = False):
        # TODO: Make this a lazy property.
        """Return the end point of the curve."""
        return self.beziers[-1].end_point(world_space)

    def offset_spline(self, d):
        """Creates splines called self.left_curve, and self.right_curve that
        are approximately parallel to self and are offset a distance d in each
        direction."""

        # TODO: Intersections of the inner linejoin should be removed at this stage.
        # One of the sides will always intersect and we should cut the two bezier curves at that point and join them there instead. 

        # TODO: This should handle the miter problems.
        # Everytime a new bezier is added, we should determine 
        # if the previous handle and the next handle are aligned or not. 
        # If they are not, we need to handle the miter problem by 
        # adding the corresponding filling curve.

        # self.stroke-linejoin: miter, round, bevel
        # initial miter
        # stroke-miterlimit 
        # inital 4

        stroke_linejoin = 'bevel'

        loc = self.location
        # for bez in self.beziers:
        #     bez.offset_curve(d)

        # Each offset is really a Spline, so perhaps we should call
        # everything splines below.
        first_bezier = self.beziers[0] # TODO: This assumes that there are only one Bez? 
        first_bezier.offset_curve(d)
        left_curves = [first_bezier.left_offset]
        right_curves = [first_bezier.right_offset]
        

            # left_curves.append(bez.left_offset)
            # right_curves.append(bez.right_offset)
            
        # TODO: Move the different cases to separate functions.
        # That will probably make it look nicer. 
        if stroke_linejoin == 'bevel':
            for bez in self.beziers[1:]:
                bez.offset_curve(d)

                l0e = left_curves[-1].end_point()
                l0 = left_curves[-1].beziers[-1].tangent(1) - l0e
                l1s = bez.left_offset.start_point()
                l1 = bez.left_offset.beziers[0].tangent(0) - l1s
                if l0e == l1s or is_colinear(l0, l1):
                    left_curves.append(bez.left_offset)
                else:
                    linejoin = Bezier(l0e,l0e, l1s,l1s, location = loc)
                    bez.left_offset.prepend_bezier(linejoin)
                    left_curves.append(bez.left_offset)

                r0e = right_curves[-1].end_point()
                r0 = right_curves[-1].beziers[-1].tangent(1) - r0e
                r1s = bez.right_offset.start_point()
                r1 = bez.right_offset.beziers[0].tangent(0) - r1s
                if r0e == r1s or is_colinear(r0, r1):
                    right_curve.append(bez.right_offset)
                else:
                    linejoin = Bezier(r0e, r0e, r1s, r1s, location = loc)
                    bez.right_offset.prepend_bezier(linejoin)
                    right_curves.append(bez.right_offset)

        elif stroke_linejoin == 'round':
            pass
        elif stroke_linejoin == 'miter':
            pass


        # TODO: Make this nicer.
        left_curve = left_curves[0]
        del left_curves[0]

        right_curve = right_curves[0]
        del right_curves[0]

        # TODO: Here! Check if the last handle of the previous curve
        # is in the opposite direction as the next curve's handle.
        # If not, we should add some filler spline.
        for k in left_curves:
            left_curve.append_spline(k)

        for k in right_curves:
            right_curve.append_spline(k)

        loc = self.location
        left_curve.location = loc
        right_curve.location = loc
        self.left_curve = left_curve
        self.right_curve = right_curve

        # left_curve.add_to_Blender()
        # right_curve.add_to_Blender()

    def stroke(self):
        # TODO:
        # self.endcap: butt [x] , round [x], square [x]
        # initial butt
        # self.stroke-linejoin: miter, round, bevel
        # initial miter
        # stroke-miterlimit 
        # inital 4
        # self.strokewidth
        # endcap = "butt"
        # endcap = "square"
        strokewidth = self.strokewidth
        endcap = "round"
        stroke_linejoin = "miter"
        stroke_miterlimit = .1

        # If miter, then we should extend the tangents until they meet.
        self.offset_spline(strokewidth/2)
        loc = self.location
        lc = self.left_curve
        rc = self.right_curve
        
        stroke = Spline(*lc.beziers, location = loc)

        # End points of left and right offset curves.
        lc_start = lc.start_point()
        rc_start = rc.start_point()
        lc_end = lc.end_point()
        rc_end = rc.end_point()

        if endcap == "butt":
            # The stroke is simply obtained by connecting the left and right 
            # offset curves with a straight line.
            start_cap = Bezier(rc_start, rc_start, lc_start, lc_start, 
                               location=loc)
            stroke.prepend_bezier(start_cap)
            end_cap = Bezier(lc_end, lc_end, rc_end, rc_end, 
                             location=loc)
            stroke.append_bezier(end_cap)

        elif endcap == "square":
            lc_tangent_start = lc.beziers[0].tangent(0) 
            rc_tangent_start = rc.beziers[0].tangent(0) 
            lc_tan_start = lc_start - lc_tangent_start * strokewidth / 2
            rc_tan_start = rc_start - rc_tangent_start * strokewidth / 2

            start_cap1 = Bezier(lc_tan_start, lc_tan_start, 
                                lc_start, lc_start, location = loc)
            start_cap2 = Bezier(rc_tan_start, rc_tan_start, lc_tan_start, 
                                lc_tan_start, location = loc)
            start_cap3 = Bezier(rc_start, rc_start, rc_tan_start, 
                                rc_tan_start, location = loc)

            stroke.prepend_bezier(start_cap1)
            stroke.prepend_bezier(start_cap2)
            stroke.prepend_bezier(start_cap3)

            lc_tan_end = lc_end + lc.beziers[-1].tangent(1) * strokewidth / 2 
            rc_tan_end = rc_end + lc.beziers[-1].tangent(1) * strokewidth / 2 

            end_cap1 = Bezier(lc_end, lc_end, lc_tan_end, lc_tan_end, 
                              location = loc)
            end_cap2 = Bezier(lc_tan_end, lc_tan_end, rc_tan_end, rc_tan_end, 
                              location = loc)
            end_cap3 = Bezier(rc_tan_end, rc_tan_end, rc_end, rc_end, 
                              location = loc)
            stroke.append_bezier(end_cap1)
            stroke.append_bezier(end_cap2)
            stroke.append_bezier(end_cap3)
        elif endcap == "round":
            alpha = (math.sqrt(4 + 3 * math.tan(math.pi/4)) - 1 ) / 3 * strokewidth / 2
               
            lc_tan_start = lc_start - lc.beziers[0].tangent(0) * alpha
            rc_tan_start = rc_start - rc.beziers[0].tangent(0) * alpha

            # TODO: Add function to calculate tangents and normals at different places for splines.

            spline_start = self.start_point()
            spline_start_tangent = self.beziers[0].tangent(0)

            middle_start = spline_start - strokewidth / 2 * spline_start_tangent
            spline_start_normal = self.beziers[0].normal(0)

            start_cap1 = Bezier(middle_start, 
                                middle_start - alpha * spline_start_normal,
                                lc_tan_start, 
                                lc_start,
                                location = loc)
            start_cap2 = Bezier(rc_start, 
                                rc_tan_start, 
                                middle_start + alpha * spline_start_normal,
                                middle_start,
                                location = loc)


            stroke.prepend_bezier(start_cap1)
            stroke.prepend_bezier(start_cap2)

            lc_tan_end = lc_end + lc.beziers[-1].tangent(1) * alpha
            rc_tan_end = rc_end + lc.beziers[-1].tangent(1) * alpha

            spline_end = self.end_point()
            spline_end_tangent = self.beziers[-1].tangent(1)

            middle_end = spline_end + strokewidth / 2 * spline_end_tangent
            spline_end_normal = self.beziers[-1].normal(1)

            end_cap1 = Bezier(lc_end, lc_tan_end,
                              middle_end - alpha * spline_end_normal, 
                              middle_end,
                              location = loc) 
            end_cap2 = Bezier(middle_end, 
                              middle_end + alpha * spline_end_normal,
                              rc_tan_end, rc_end,
                              location = loc)

            stroke.append_bezier(end_cap1)
            stroke.append_bezier(end_cap2)

        else:
            raise Exception("Unknown endcap type: {}".format(endcap))

        # The right side offset curve in reverse.
        stroke.append_spline(reversed(rc)) 
        stroke.closed = True
        stroke.name = self.name + ": Stroke"
        stroke.add_to_Blender()


class Curve(CurveObject):
    """Curve object, container class for splines. Mirrors the Curve Object in Blender."""
    
    __slots__ = ("splines")

    # TODO: Add alternative constructor to import this from Blender.
    # TODO: When that is implemented, check if the location setter works properly.
    # In the end, every single Bezier within each Spline should have the same location.

    def __init__(self, *splines: Spline, name = "Curve", location = Vector((0,0,0))):
        self.splines = list(splines)
        super().__init__(name, location)

    @classmethod
    def from_Blender(cls, name: str):
        # Loop over all splines, and create these (perhaps call that class.
        # Then create the Curve from the splines created.
        pass

    @CurveObject.location.setter
    def location(self, loc):
        for spline in self.splines:
            spline.location = loc

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


class OffsetBezier():
    # TODO: Add slots
    """Handles all calculation of offset curves."""
    def __init__(self, bez: Bezier, d: float):
        self.bez = bez # Original Bezier, really no need to store this.
        self.d = d
        self.is_linear = False
        rot_curve, angle, translation, endpoint = self._rotate_original_to_x(bez)
        self.rot_curve = rot_curve # The rotated curve.
        self.angle = angle
        self.translation = translation
        self.endpoint = endpoint
        self.angles = self._calculate_angles()
        self.metrics = self.calculate_metrics()

    def eval_offset(self, t: float):
        """Evaluates the offset curve position at parameter t."""
        # TODO: It might be more resonable to have this return 
        # the equivalent of cur.eval_offset. 
        # That function could probably be removed from the Bezier class
        # since it is only used for offsetting. 
        cur: Bezier = self.rot_curve
        return cur.eval_offset(t, self.d) + cur(t)
    
    def eval_offset_derivative(self, t: float):
        """Evaluates the derivative of the offset at parameter t."""
        cur: Bezier = self.rot_curve
        der = cur.derivative()
        dp = der(t)
        dpp = der.eval_derivative(t)
        k = 1.0 + dpp.cross(dp)[2] * self.d / (dp.length**3) 
        return k * dp

    def _rotate_original_to_x(self, bez: Bezier):
        """Calculates a new, transformed (rotated and translated) curve, such that the offset 
        of that will end up starting at (0,0) and ending on the x-axis (at endp).
        Returns the new Bezier, the rotation angle, the translation, and the endpoint..
        """
        d = self.d
        p0 = bez.points[0]
        p1 = bez.points[1]
        p2 = bez.points[2]
        p3 = bez.points[3]
        # There are three problematic cases: 
        # 1. p1 = p0 and/or p2 = p3, in these cases the derivative at t = 0 and/or
        #    t = 1, respectively, are zero. 
        # 2. p0, p1, p2, and p3 are in a straight line. 
        #    In this case curve is just a straight line and the offset 
        #    algorithm does not work well.
        # The solutions: 
        # In case 1 holds, then we find an approximate handle at either or both ends and 
        # use that as our p1 or p2. 
        # In case both cases hold, then we use the middle of the curve for both handles and 
        # set is_linear = True.
        # 3. In this case we set is_linear = True
        l_linear = False
        r_linear = False
        dotp0123 = abs( (p0 - p1).x * (p3 - p2).x + (p0 - p1).y * (p3 - p2).y)
        l0123 = math.sqrt((p0 - p1).x**2 + (p0 - p1).y**2) * math.sqrt((p3 - p2).x ** 2 + (p3 - p2).y**2)
        dotp023 = abs((p0 - p2).x * (p2 - p3).x + (p0 - p2).y * (p2 - p3).y)
        l023 = math.sqrt((p0 - p2).x**2 + (p0 - p2).y**2) * math.sqrt((p2 - p3).x ** 2 + (p2 - p3).y**2)
        dotp013 = abs((p0 - p1).x * (p1 - p3).x + (p0 - p1).y * (p1 - p3).y)
        l013 = math.sqrt((p0 - p1).x**2 + (p0 - p1).y**2) * math.sqrt((p1 - p3).x ** 2 + (p1 - p3).y**2)
        if (p1 == p0 and p2 == p3):
            self.is_linear = True
        # elif dotprod == l2:
        #     self.is_linear = True
        elif p0 == p1: 
            l_linear = True
            if dotp023 == l023: 
                print("Linear! 023")
                self.is_linear = True
                p1 = bez(0.5)
            else:
                # n = 0
                # p1_new = bez(1 / (n + 1)) - p0
                # while p1_new.length > 0 and n < 1000:
                #     n += 1
                #     p1_new = bez(1 / (n + 1)) - p0
                # p1 = bez(1 / n)
                # print(n)
                p1 = bez(.0001)
        elif p2 == p3: 
            r_linear = True
            if dotp013 == l013:
                print("Linear! 013")
                self.is_linear = True
                p2 = bez(0.5)
            else:
                # Make an approximate handle.
                p2 = bez(0.9999)
                # n = 0
                # p2_new = p3 - bez(1 - 1 / (n + 1))
                # while p2_new.length > 0 and n < 100:
                #     n += 1
                #     p2_new = p3 - bez(1 - 1 / (n + 1))
                # print(n)
                # p2 = bez(1 - 1 / n)
        elif dotp0123 == l0123:
            self.is_linear = True
        if l_linear:
            h0 = (p1 - p0) / (p1 - p0).length
            r0 = p0 + d * Vector((-h0.y, h0.x, 0))
        else: 
            r0 = p0 + bez.eval_offset(0, d)
        if r_linear: 
            h1 = (p3 - p2) / (p3 - p2).length
            r1 = p3 + d * Vector((-h1.y, h1.x, 0))
        else:
            r1 = p3 + bez.eval_offset(1, d)
        th = math.atan2( (r1 - r0)[1], (r1 - r0)[0])
        rot = mathutils.Matrix.Rotation(-th, 3, 'Z')
        q0 = rot @ (bez.points[0] - r0)
        q1 = rot @ (p1 - r0) if l_linear else rot @ (bez.points[1] - r0)
        q2 = rot @ (p2 - r0) if r_linear else rot @ (bez.points[2] - r0)
        q3 = rot @ (bez.points[3] - r0)
        b = Bezier(q0, q1, q2, q3) #type: ignore
        endp = b.eval_offset(1, d) + b(1) # Endpoint of the rotated offset (on the x-axis).
        return b, th, r0, endp

    def _calculate_angles(self):
        """Calculate the angles of the handles that the offset should have."""
        rp = self.rot_curve.points
        q0 = rp[1] - rp[0]
        q1 = rp[2] - rp[3]
        theta0 = math.atan2(q0[1], q0[0])
        theta1 = math.atan2(q1[1], q1[0])
        return [theta0, theta1]

    def _sample_points(self, n: int):
        """Sample the offset curve at n points."""
        samples = []
        arclen = 0.0
        co = GAUSS_LEGENDRE_COEFFS_32
        dt = 1 / (n + 1)
        for i in range(0, n):
            for j in range(0, len(co), 2):
                t = dt * (i + 0.5 + 0.5 * co[j + 1])
                arclen += co[j] * self.eval_offset_derivative(t).length
            t = dt * (i + 1)
            d = self.rot_curve.eval_offset(t, self.d)
            p = self.eval_offset(t)
            samples.append({'arclen': arclen * 0.5 * dt, 'p': p, 'd': d})
        return samples

    def _estimate_cubic_error(self, cu: Bezier, samples, tolerance):
        # TODO: Does not really use self. 
        err = 0.0
        tol2 = tolerance**2
        # For each of the samples of the expected offset...
        # print(cu.name)
        for sample in samples:
            best_err: float = None # type: ignore
            # We find the corresponding point on the candidate cu. 
            samps = cu.intersect_ray(sample['p'], sample['d'])
            print("samps", samps)
            # print("sample", sample)
            # print("samps", samps)
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
                print("Failed")
                break
        # print("Err", math.sqrt(err))
        return math.sqrt(err)

    def find_cubic_approximation(self, tolerance = 0.1, sign = 1):
        if not self.is_linear:
            cubics = self._find_cubic_candidates()
            samples = self._sample_points(20)
            best_curve: Bezier = None # type: ignore
            best_err: float = None # type: ignore
            errs: list[float] = []
            for i, cur in enumerate(cubics):
                err = self._estimate_cubic_error(cur, samples, tolerance)
                errs.append(err)
                # cur.transform(self.angle, self.translation)
                # cur.name = str(i) + str(err)
                # cur.add_to_Blender()
                if best_curve is None or err < best_err:
                    best_err = err
                    best_curve = cur
        # print("Errs", errs)
        else: 
            p = self.rot_curve.points
            der = (p[3] - p[0]) / (p[3] - p[0]).length
            disp = self.d * Vector((-der.y, der.x, 0))
            p0 = p[0] + disp
            p1 = p[1] + disp
            p2 = p[2] + disp
            p3 = p[3] + disp
            best_curve = Bezier(p0, p1, p2, p3)
        best_curve.transform(self.angle, self.translation)
        best_curve.add_to_Blender() 
            # TODO: Return error?

    def _find_cubic_candidates(self):
        """Solve the quartic equation for delta0 and delta1 that give the offset which closest 
        matches the required metrics."""
        mx: float = self.metrics["moment_x"]
        a: float = self.metrics["area"]
        x3: float = self.endpoint[0] # The x-value of the endpoint.
        th0 = self.angles[0]
        th1 = self.angles[1]
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
        if k4 != 0.0:
            sols = solvers.solve_quartic(k3 / k4, k2 / k4, k1 / k4, k0 / k4)
        elif k3 != 0.0:
            sols = solvers.solve_cubic(k2 / k3, k1 / k3, k0 / k3)
        elif k2 != 0.0:
            sols = solvers.solve_quadratic(k1 / k2, k0 / k2)
        elif k1 != 0.0:
            sols = [k0 / k1]
        else:
            print("NO CURVE FOUNDS")
            sols = []
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
                p0 = Vector((0, 0, 0))
                p1 = Vector((d0 * c0, d0 * s0, 0))
                p3 = Vector((x3, 0, 0))
                p2 = p3 + Vector((d1 * c1, d1 * s1, 0))
                b = Bezier(p0, p1, p2, p3)
                b.name = "Bezier" + str(n)
                n += 1
                cubics.append(b)
                # b.add_to_Blender() # For testing.
            # print("d0, d1", d0, d1, a, mx)
        return cubics

    def calculate_metrics(self):
        """Calculates the x-moment, area, and arc length that the offset should have."""
        arclen = 0.0
        area = 0.0
        moment_x = 0.0
        co = GAUSS_LEGENDRE_COEFFS_32
        # Uses a Gauss-Legendre quadrature.
        # TODO: Read up on this.
        for i in range(0, len(co), 2): 
            t = 0.5 * (1 + co[i + 1])
            wi = co[i]
            dp = self.eval_offset_derivative(t)
            p = self.eval_offset(t)
            d_area = wi * dp[0] * p[1]
            arclen += wi * dp.length
            area += d_area;
            moment_x += p.x * d_area; 

        return {"area": 0.5 * area, "length": 0.5 * arclen, "moment_x": 0.5 * moment_x}


def ray_intersect(p0: Vector, d0: Vector, p1: Vector, d1: Vector): 
    # TODO: Figure out what this does exactly. 
    det = d0.x * d1.y - d0.y * d1.x
    t = (d0.x * (p0.y - p1.y) - d0.y * (p0.x - p1.x)) / det
    return Vector((p1.x + d1.x * t, p1.y + d1.y * t))


# Testing values relevant for trying to "approximate" another cubic
# with positions 
# Vector((0.0, 0.0, 0.0)), 
# Vector((1.2317887544631958, 0.9850826859474182, 0.002845139242708683))
# Vector((2.3887062072753906, 0.602931022644043, -0.0027865534648299217))
# Vector((2.999779224395752, 0.0, 0.0))
# mx = 1.869782404656517
# a = 1.2274109939627678
# th0 = 0.6745684953714891
# th1 = 2.362901117301283
# x3 = 2.999779224395752
