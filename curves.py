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
    # print('curve', c1.t1, c1.t2, c2.t1, c2.t2)
    if c1.t2 - c1.t1 < threshold and c2.t2 - c2.t1 < threshold:
        return [((c1.t1 + c1.t2)/2 , (c2.t1 + c2.t2)/2)]
        # return [(c1.t1, c2.t1)]

    cc1 = c1.split(0.5)
    cc2 = c2.split(0.5)
    pairs = itertools.product(cc1, cc2)

    # pairs = list(filter(lambda x: are_overlapping(x[0].bounding_box(world_space = True), x[1].bounding_box(world_space = True)), pairs))
    pairs = list(filter(lambda x: x[0].overlaps(x[1]), pairs))
    print(pairs)
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


class CurveObject():
    """Base class for all curves."""
    __slots__ = ("name", "_location")
    # TODO: Perhaps it makes more sense to make this the
    # base class of only Spline and Curve.
    # The both have a lot in common.
    
    def __init__(self, name = "Curve Object", location = mathutils.Vector([0,0,0])):
        self.name = name
        self.location = location
        
    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, loc):
        self._location = loc

    # TODO: Add rotation?

    # Methods:
    # - boundary box
    # - intersections 
    # - self intersections
    # - create blender curve object


class Bezier(CurveObject):
    """
    Bezier curve of 3rd order. 
    p0, p1, p2, p3 are mathutils.Vector
    t1 and t2 are the parameter time at the start and the end. 
    Defaults are 0 and 1 respectively. 
    """
    __slots__ = ("points", 
                 "t1", "t2",
                 "start_handle_left",
                 "end_handle_right",
                 "left_offset",
                 "right_offset",
                 )

    def __init__(self, 
                 p0: mathutils.Vector,
                 p1: mathutils.Vector,
                 p2: mathutils.Vector,
                 p3: mathutils.Vector,
                 t1: float = 0,
                 t2: float = 1, 
                 name = "Bezier", 
                 location = Vector((0,0,0)),
                 start_handle_left: mathutils.Vector = None,
                 end_handle_right: mathutils.Vector = None 
                 ) -> None:
        """ 
        Initializes the cubic Bezier and sets its points and degree. 
        The points should be mathutils.Vectors of some fixed dimension.
        The number of points should be 3 or higher. 
        """
        super().__init__(name, location)
        self.points = [p0, p1, p2, p3]
        # The dangling handles of a Bezier curve in Blender are not 
        # really part of a mathematical curve. 
        # Instead they belong to the previous or next Bezier in case of a
        # poly-bezier curve. 
        # Since Blender uses them, it is better to keep them.
        self.start_handle_left = start_handle_left or p0
        self.end_handle_right = end_handle_right or p3
        
        # t1 and t2 give the parameter values of the parent curve 
        # in case this is created from a split. 
        # Needed for keeping track of intersections. 
        # TODO: Might not need this with the new algorithm. 
        # But perhaps to find intersections!
        self.t1 = t1
        self.t2 = t2

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

    def set_point(self, point, i):
        """Sets the point with index i of the Bezier."""
        self.points[i] = point

    def translate_origin(self, vector):
        """Translates the origin of the Bezier to vector without changing the 
        world position of the curve. 
        """
        dist = self.location - vector
        self.points = [p + dist for p in self.points]
        self.start_handle_left = self.start_handle_left +  dist
        self.end_handle_right = self.end_handle_right + dist
        self.location = vector

    def derivative(self, t):
        p = self.points
        return 3 * (p[1] - p[0]) * (1-t)**2 + 6 * (p[2] - p[1]) * (1-t) * t + 3 * (p[3] - p[2]) * t**2 

    def second_derivative(self, t):
        """Evaluate the second derivative at parameter t."""
        # TODO: Only used in curvature, which is currently not used for anything. 
        p = self.points
        return 6 * (p[0] - 2 * p[1] + p[2]) * (1-t) + 6 * (p[1] - 2 * p[2] + p[3]) * t

    def tangent(self, t): 
        """Calculate the tangent at parameter t."""
        derivative = self.derivative(t) 
        if derivative.length > 0: 
            return derivative / derivative.length #mathutils.vector.length
        else:
            return derivative # Vector((0,0,0))

    def normal(self, t):
        return self.tangent(t).cross(Vector((0,0,1))) 
        # This is probably faster: 
        # a = self.tangent(t)
        # return Vector((a[1], -a[0], 0))
        # TODO: Run tests and use the faster version.

    def curvature(self, t):
        """Returns the curvature at parameter t."""
        d = self.derivative(t)
        sd = self.second_derivative(t)
        denom = math.pow(self.derivative(t).length, 3 / 2)
        return (d[0] * sd[1] - sd[0] * d[1]) / denom

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
        inflection_points = solvers.solve_quadratic(e, f, g)
        inflection_points = [p for p in inflection_points if p >= 0.0 and p <= 1.0]
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

    def split(self, t1, t2 = None):
        """Splits the Bezier curve at the parameter(s) t1 (and t2). 
        In case just one parameter value is given, a list of two curves 
        is returned. 
        Else, a single curve, corresponding to the curve between 
        t1 and t2 is returned. 
        Based on: https://github.com/Pomax/BezierInfo-2 
        The code for this function is almost a straight translation
        of the JavaScript code in the ref above int Python.
        """
        loc = self.location
        if t1 == 0 and t2 is not None: 
            return self.split(t2)[0]
        elif t2 == 1:
            return self.split(t1)[1]
        else: 
            p = self.points

            new1 = p[0] * (1 - t1) + p[1] * t1 
            new2 = p[1] * (1 - t1) + p[2] * t1 
            new3 = p[2] * (1 - t1) + p[3] * t1 
            new4 = new1 * (1 - t1) + new2 * t1 
            new5 = new2 * (1 - t1) + new3 * t1 
            new6 = new4 * (1 - t1) + new5 * t1

            result = [Bezier(p[0], new1, new4, new6, location=loc), Bezier(new6, new5, new3, p[3], location=loc)]

            # The new split curves should keep track for the original 
            # parameter values at the end points. 
            result[0].t1 = self.map_split_to_whole(0) 
            result[0].t2 = self.map_split_to_whole(t1) 
            result[1].t1 = self.map_split_to_whole(t1) 
            result[1].t2 = self.map_split_to_whole(1) 

            if not t2:
                return result
            else: 
                # Calculate which parameter of the split curve (result[1]) 
                # which corresponds to the point t2 on the whole curve. 
                # Then split again at that point. 
                t2p = self.map_whole_to_split(t2, t1, 1) 
        return result[1].split(t2p)[0] 

    def map_whole_to_split(self, t, ds, de):
        """Returns the parameter in the splitted curve 
        corresponding to the parameter t of the whole (unsplitted) curve. 
        t2 is the parameter value which we want to map and 
        the split curve runs from parameter ds to de of the whole curve. 

        Ref: http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node13.html
        """
        # TODO: This does not really depend on self. Move to utility.
        return (t - ds) / (de - ds)
    
    def map_split_to_whole(self, t: float):
        """Returns the parameter value of the whole curve, 
        corresponding to the parameter t in the splitted curve. 
        """
        return self.t1 + (self.t2 - self.t1) * t

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
        if a.angle_signed(b) >= 0:
            return True
        else:
            return False


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
            bezier.t1 = 0.0
            bezier.t2 = 1.0

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