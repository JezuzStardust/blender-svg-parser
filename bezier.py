"""
Classes and utility functions for Bezier curves.  
"""

# TODO: Design options
# We will have three classes: Bezier, Spline (stringed together Beziers), and Curve (collection of splines). 
# They can all inherit from a base class which contains their common variables (e.g. location, rotation) and methods (e.g. add to blender?). The base class can perhaps be an ABC where the implementation of some methods are required.  
# The base class can also be responsible for the connection to Blender. 

# Bezier: Contains all data and operations needed to define and work with a Bezier curve. 
# Spline: Contains a list of Bezier curves. 
# 1. How should we init this class? Either we init by passing all the points and the class creates and stores the Bezier instances, or we can init by passing pre-fabricated Bezier instances. 
# 2. Can we programme for both options? Either with wargs or kwargs. 
# Curves
# Again we need to think about how to init these. 

# TODO: Remove premature optimization. 
# Once we are done, we can optimize needed parts.
# import cProfile
# cProfile.run('<commands here as quoted string'>)


# Begin constants
# TODO: Move to separate module. 
THRESHOLD = 0.00001 # TODO: Expose this in the plugin version?
# End constants

from mathutils import Vector, Matrix # TODO: Remove this and use explicit refs.
import mathutils 
# import numpy as np
import math
import bpy
import itertools
import operator

class CurveObject():
    """Base class for all curve."""
    # TODO: Perhaps it makes more sense to make this the
    # base class of only Spline and Curve.
    # The both have a lot in common.
    def __init__(self, name, location, rotation):
        self.name = name
        self.location = location
        self.rotation = rotation
    
# BBOX
# overlaps with
# intersects 
# points (only bezier curves have points really.. splines have beziers and 
# curves have splines...? 

### Utility Functions ###
# TODO: Move to separate module or put in some baseclass.  

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

def quadratic_solve(a,b,c): 
    """Returns the solution of a quadratic equation."""
    """
    Numpy version
    def quadratic_solve(a,b,c): 
         roots = np.roots([a,b,c])
         rot = []
         for root in roots:
             if np.isreal(root):
                 rot.append(root)
         return tuple(rot)
     """
    if a == 0:
        if b == 0:
            return ()
        else: 
            return (-c / b, )
    
    d = - b / (2 * a)
    e = b**2 - 4 * a * c
    if e == 0:
        return (d,)
    elif e > 0: 
        f = math.sqrt(e) / (2 * a)
        return (d + f, d - f) 
    elif e < 0: 
        # We do not want the complex solutions. 
        # TODO: Make this more general and handle the complex
        # case from the caller instead?
        return () 

def curve_intersections(c1, c2, threshold = 0.01):
    """Recursive method used for finding the intersection between 
    two Bezier curves, c1, and c2. 
    """
    # TODO: Make this a method of Bezier and/or Curve. 
    # It can have the signature _find_intersections_recursively(self, curve, threshold)

    if c1._t2 - c1._t1 < threshold and c2._t2 - c2._t1 < threshold:
        # return [((c1._t1 + c1._t2)/2 , (c2._t1 + c2._t2)/2)]
        return [(c1._t1, c2._t1)]

    cc1 = c1.split(0.5)
    cc2 = c2.split(0.5)
    pairs = itertools.product(cc1, cc2)
    # pairs = [pair for pair in pairs if are_overlapping(pair[0].bounding_box(), pair[1].bounding_box())] 

    pairs = list(filter(lambda x: are_overlapping(x[0].bounding_box(), x[1].bounding_box()), pairs))
    results = [] 
    if len(pairs) == 0:
        return results
    
    for pair in pairs:
        results += curve_intersections(pair[0], pair[1], threshold)
    results = filter_duplicates(results, threshold)
    return results

def filter_duplicates(tuples, threshold = 0.001):
    """
    Filter out tuples that differ less than threshold. 
    """
    result = []
    for tup in tuples:
        if not any(is_close(tup, other, threshold) for other in result):
            result.append(tup)
    return result

def is_close(a, b, threshold = 0.01):
    comps = all(math.isclose(*c, abs_tol = threshold) for c in zip(a,b))
    return comps

def is_colinear(v1, v2, threshold = 0.00000001):
    return v1.cross(v2).length < threshold

def bezier_from_Blender(name):
    """Read and import a curve from Blender. 
    Used mainly during developement (probably).
    """
    cu = bpy.data.collections['Collection'].objects[name]
    p0 = cu.data.splines[0].bezier_points[0].co
    p1 = cu.data.splines[0].bezier_points[0].handle_right
    p2 = cu.data.splines[0].bezier_points[1].handle_left
    p3 = cu.data.splines[0].bezier_points[1].co
    loc = cu.location
    return Bezier(p0, p1, p2, p3, name=name, location=loc)

def spline_from_Blender(name):
    """Read and import a curve from Blender. 
    Used mainly during development (probably).
    """
    # TODO: Should check if the spline is closed.
    cu = bpy.data.collections['Collection'].objects[name]
    # not be good. 
    beziers = []
    # Iterate over all splines..
    loc = cu.location
    closed = cu.data.splines[0].use_cyclic_u
    for spline in cu.data.splines: 
        # and within all splines over all points. 
        i = len(spline.bezier_points) - 1
        for j in range(0, i):
            p0 = cu.data.splines[0].bezier_points[j].co
            p1 = cu.data.splines[0].bezier_points[j].handle_right
            p2 = cu.data.splines[0].bezier_points[j + 1].handle_left
            p3 = cu.data.splines[0].bezier_points[j + 1].co
            beziers.append(Bezier(p0, p1, p2, p3, location = loc))
    return Spline(*beziers, name=name, location=loc, closed = closed) 

### End: Utility Functions ###

class Bezier():
    """
    Bezier curve of 3rd order. 
    p0, p1, p2, p3 are mathutils.Vector
    """
    # TODO: Use slots if that optimizes performance. 
    # TODO: Use @property for lazy evaluation of properties. See end of this file.
    #       Probably more Pythonic than using __getattr__ the way I have. 

    def __init__(self, p0, p1, p2, p3, t1 = 0, t2 = 1, name = None, location = Vector((0,0,0))):
        """ 
        Initializes the curve and sets its points and degree. 
        The points should be mathutils.Vectors of some fixed dimension.
        The number of points should be 3 or higher. 
        For n points the curve is of order n-1. 
        """
        self.points = [p0, p1, p2, p3]
        self.location = location
        
        if name:
            self.name = name
        else:
            self.name = 'Bezier'
        # _t1 and _t2 give the parameter values of the parent curve 
        # in case this is created from a split. 
        # Needed for keeping track of intersections. 
        self._t1 = t1
        self._t2 = t2

    def __repr__(self):
        """Prints the name of the together with all the points. """
        # TODO: Consider being more explicit with the coordinates. 
        # Printing mathutils.Vector shows only a limited number of decimals. 
        p = self.points
        # string = self.name + '\n' + str(p[0]) + '\n' + str(p[1]) + '\n' + str(p[2]) + '\n' + str(p[3])
        string = self.name + str(p[0]) + '\n' + str(p[1]) + '\n' + str(p[2]) + '\n' + str(p[3])
        return string

    def __call__(self, t):
        """ Returns the value at parameter t."""
        p = self.points
        return p[0] * (1 - t)**3 + 3 * p[1] * (1 - t)**2 * t + 3 * p[2] * (1 - t) * t**2 + p[3] * t**3 + self.location
    
    def __reversed__(self):
        self.points = list(reversed(self.points))

    def set_point(self, point, i):
        """Sets the point with index i of the Bezier."""
        self.points[i] = point

    def translate_origin(self, vector):
        """Translates the origin of the Bezier to vector without changing the 
        world position of the curve. 
        """
        dist = self.location - vector
        self.points = [p + dist for p in self.points]
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
        denom = math.pow(self.derivative(t).length,3/2)
        return (d[0] * d[1] - sd[0] * sd[1]) / denom

    def aligned(self):
        """Returns the points of the corresponding aligned curve. 
        Aligned means: start point in origin, end point on x-axis. 
        """
        m = Matrix.Translation(-self.points[0])
        end = m @ self.points[3]
        if end[0] != 0:
            angle = -math.atan(end[1] / end[0])
        else:
            angle = 0
        m = Matrix.Rotation(angle, 4, 'Z') @ m

        aligned_points = []
        for p in self.points:
            aligned_points.append(m @ p)
        return aligned_points

    def bounding_box(self):
        """Calculate the bounding box of the curve."""
        # TODO: Make it possible to calculate the tight bounding box if this
        # is deemed useful. 
        # TODO: Consider returning the box in some other format. 
        # TODO: Consider using a Rectangle class for this. 
        # Need to calculate the rotation and translation also! 
        extrema = self.extrema() # TODO: Abs does not work! 
        min_x = self.__call__(extrema[0])[0]
        max_x = self.__call__(extrema[1])[0]
        min_y = self.__call__(extrema[2])[1]
        max_y = self.__call__(extrema[3])[1]
        return (min_x, max_x, min_y, max_y)

    def extrema(self):
        """
        Returns the parameter values for the minimum and maximum of the curve
        in the x and y coordinate. 
        (tx_min, tx_min, ty_max, ty_max)
        If absolute we return the extremas of the aligned curve.
        """
        p0, p1, p2, p3 = self.points

        a = 3 * (-p0 + 3 * p1 - 3 * p2 + p3)
        b = 6 * (p0 - 2*p1 + p2)
        c = 3*(p1 - p0) 
        
        # TODO: Clean up using e.g. itertools. 
        endpoints = (0.0, 1.0) # Must also check if extrema occurs on endpoint.
        tx_roots = endpoints + quadratic_solve(a[0], b[0], c[0]) 
        ty_roots = endpoints + quadratic_solve(a[1], b[1], c[1]) 

        tx_roots = [t for t in tx_roots if 0.0 <= t <= 1.0] 
        ty_roots = [t for t in ty_roots if 0.0 <= t <= 1.0] 

        x_values = [self.__call__(t)[0] for t in tx_roots] 
        y_values = [self.__call__(t)[1] for t in ty_roots] 

        tx_max = tx_roots[x_values.index(max(x_values))]
        tx_min = tx_roots[x_values.index(min(x_values))]
        ty_max = ty_roots[y_values.index(max(y_values))]
        ty_min = ty_roots[y_values.index(min(y_values))]

        return [tx_min, tx_max, ty_min, ty_max]

    def inflection_points(self):
        """Returns a list of parameter values where inflection points occurs. 
        The length of the list can be zero, one, or two. 
        """
        # TODO: Make this a lazily calculated data attribute. 
        # Align the curve to the x-axis to simplify equations. 
        # https://pomax.github.io/bezierinfo/#inflections
        p0, p1, p2, p3 = self.aligned()
        # TODO: Itertools?
        a = p2[0] * p1[1]
        b = p3[0] * p1[1]
        c = p1[0] * p2[1]
        d = p3[0] * p2[1] 
        e = - 3 * a + 2 * b + 3 * c - d 
        f = 3 * a - b - 3 * c
        g = c - a 
        inflection_points = quadratic_solve(e, f, g)
        inflection_points = [p for p in inflection_points if p >= 0 and p <= 1]
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
        """Splits all reduced curves down the middle. Keeps splitting 
        until none of the curves has the middle of the curve too far
        from the center of the box created by the points. 
        """
        beziers = []
        for bezier in self.reduced():
            beziers += bezier.split(0.5)

        all_simple = False
        
        while not all_simple:
            all_simple = True
            new_set = []
            for bez in beziers:
                if bez._is_offsetable(0.03):
                    new_set.append(bez)
                else:
                    all_simple = False
                    new_set += bez.split(0.5)
            beziers = new_set

        return beziers

    def _is_offsetable(self, threshold = 0.05):
        """Check that Bezier(0.5) is not too far from the center 
        of the bounding box defined by the Bezier.points. 
        If the curve is straight, then we always return True. 
        """
        # __call__ gives the world space location, but points are 
        # defined in local space. 
        # We subtract the object location of the call.
        mid = self.__call__(0.5) - self.location
        mid.resize_2d()
        p = self.points
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
            result[0]._t1 = self.map_split_to_whole(0) 
            result[0]._t2 = self.map_split_to_whole(t1) 
            result[1]._t1 = self.map_split_to_whole(t1) 
            result[1]._t2 = self.map_split_to_whole(1) 

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

    def map_split_to_whole(self, t):
        """Returns the parameter value of the whole curve, 
        corresponding to the parameter t in the splitted curve. 
        """
        # TODO: Move to utility. Does not depend on self. 
        return self._t1 + t * (self._t2 - self._t1) 

    def _create_Blender_curve(self):
        cu = bpy.data.curves.new(self.name, 'CURVE')
        ob = bpy.data.objects.new(self.name, cu)
        ob.location = self.location
        bpy.data.collections['Collection'].objects.link(ob)
        ob.data.resolution_u = 64
        cu.splines.new('BEZIER')
        return cu

    def add_to_Blender(self, blender_curve_object = None, stringed=False):
        """Adds the curve to Blender as splines."""

        # Stringed = True means that the Bezier is added as a series
        # where the end point of one curve coincides with the start point
        # of the next. 

        # If there is already a curve object to add this to. 
        # Perhaps this should signal if it is stringed or not, 
        # since this could perhaps never happen if the curve 
        # is not part of a spline? 
        if blender_curve_object is None: 
            cu = self._create_Blender_curve()
        else:
            cu = blender_curve_object

        spline = cu.splines[-1]
        bezier_points = spline.bezier_points

        p = self.points

        if not stringed:
            bezier_points[-1].co = p[0]
        bezier_points[-1].handle_right = p[1]
        bezier_points.add(1)
        bezier_points[-1].handle_left = p[2]
        bezier_points[-1].co = p[3]

    def intersections(self, bezier, threshold=THRESHOLD):
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

    def self_intersections(self, threshold=THRESHOLD):
        """Returns a list of self intersections of the curve."""
        # TODO: This can be combined with intersections to one function 
        # that has two different functions. 
        # However, it is probably easier to keep it like this, 
        # since this makes it easier to know which intersections are which. 
        c = self.reduced()
        pairs = itertools.combinations(c, 2)
        pairs = [
            pair for pair in pairs 
            if pair[0].overlaps(pair[1])
        ]
        intersections = []
        for pair in pairs:
            result = curve_intersections(*pair, threshold)
            if len(result) > 0:
                intersections += result
        return intersections

    def overlaps(self, bezier):
        """Check if the bounding box of self and Bezier overlaps."""
        # 0      1      2      3
        # min_x, max_x, min_y, max_y 
        bb1 = self.bounding_box()
        bb2 = bezier.bounding_box()
        if bb1[0] >= bb2[1] or bb2[0] >= bb1[1] or bb1[2] >= bb2[3] or bb2[2] >= bb1[3]:
            return False
        else:
            return True

    def offset_curve(self, d):
        """Returns an inner and outer curve each offset a distance d from self."""
        # TODO: Clean up. More or less the same thing is done for left and right. 
        left_offset = []
        right_offset = []
        m = Matrix().Rotation(-math.pi/2, 3, 'Z') # TODO: Move to constants.
        first = True # For the first curve we need to create the first point.
        beziers = self.simplified()

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

            a = Bezier(a0, a1, a2, a3)
            b = Bezier(b0, b1, b2, b3)
            left_offset.append(a)
            right_offset.append(b)
            i += 1

        loc = self.location
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

class Spline(): 
    """A list of Bezier curves corresponds to a single spline object."""
    # TODO: Would it make sense to combine Curve and Bezier? 
    # Perhaps Curve can inherit Bezier? Perhaps not. It is not a Bezier curve 
    # (or rather, it might be a higher order Bezier curve). 
    # TODO: Handle closed curves. 
    # 1. End with z. -> Always toggle closed. 
    # 2. End point = start point but does not end with z. -> Toggle closed 
    #    only if filled.
    # 3. End points different and z not set. -> Toggle closed only if filled. 
    # TODO: Handle multiple splines. 
    # TODO: Handle offsets when the handles at a point are not aligned. 
    # TODO: Handle endcaps.
    # TODO: Handle intersections. 
    # TODO: Handle massaging of the offset curve so that all intersections are 
    # combined. 

    # Need to somehow figure out how to show that there are multiple splines to handle. 

    # For stroking:
    # If z is set, draw nice joint between start and end points. 
    # Else, draw stroke-endcap (butt, round, square) at start and end. 
    # Stroking does not care about whether the curve is filled or not. 

    # TODO: Problem! If we manually change the location, the location of each
    # bezier within the spline are not updated.
    def __init__(self, 
                 *beziers, 
                 closed = False, 
                 name = "Spline", 
                 location = Vector((0,0,0)), 
                 start_handle_left = None, # The last handles are not included in any Bezier. 
                 end_handle_right = None,
                 ):

        self.beziers = list(beziers) # TODO: UGLY!
        self.closed = closed
        self.name = name
        self.location = location
        # Append curve-name to each Bezier. (Might reconsider this later). 
        for bezier in self.beziers:
            bezier.name = self.name + ':' + bezier.name 
            self.start_handle_left = start_handle_left
            self.end_handle_right = end_handle_right
            bezier._t1 = 0
            bezier._t2 = 1

    def __reversed__(self):
        self.beziers = list(reversed(self.beziers))
        for bez in self.beziers:
            bez = reversed(bez)
        return self

    def append_spline(self, spline):
        """
        Add curve to the this curve at the end. 
        End point of this and start point of curve must coincide. 
        """
        a = len(self.beziers)
        ep = self.end_point()
        # spline.beziers[0].points[0] = self.end_point() 
        self.beziers += spline.beziers
        self.beziers[a].points[0] = ep

        # if self.end_point() == curve.start_point():
        #     # Set the curve points equal so that they are actually the same point. 
        #     curve.beziers[0].points[0] = self.end_point() 
        #     self.beziers += curve.beziers
        #     # for bezier in curve.beziers:
        #     #     self.bezier.append(bezier) 
        # else:
        #     raise Exception('Start and end points of curves must be equal.')

    def prepend_spline(self, spline):
        """
        Add curve to the this curve at the end. 
        End point of this and start point of curve must coincide. 
        """
        if self.start_point() == curve.end_point():
            # Set the curve points equal so that they are actually the same point. 
            ep = curve.end_point()
            ep = self.start_point() 
            self.beziers = curve.beziers + self.beziers
            # for bezier in curve.beziers:
            #     self.bezier.append(bezier) 
        else:
            raise Exception('Start point of this curve must equal end point of the prepended curve.')
    
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
        bpy.data.collections['Collection'].objects.link(ob)
        ob.data.resolution_u = 64
        cu.splines.new('BEZIER')
        return cu

    def add_to_Blender(self, blender_curve_object = None):
        """
        Adds the curve to Blender as splines. 
        """
        # TODO: How should we choose which collection to add the curve to? 
        # TODO: Howe can we do this so that we reuse the add_to_Blender 
        # present in Bezier?
        # 1. The creation of the curve object and linking 
        # can be done in a common superclass method. 
        # 2. This method can, in all cases, accept an optional 
        # parameter that specifies the curve object.

        if blender_curve_object is None: 
            cu = self._create_Blender_curve()
        else:
            cu = blender_curve_object

        spline = cu.splines[-1]
        spline.use_cyclic_u = self.closed
        bezier_points = spline.bezier_points


        # TODO: This is just a temporary hack. Fix this!

        # TODO: This actually sets each bezier_point.co twice! Rethink!
        
        sp = self.start_point(world_space = False)
        bezier_points[0].handle_left = sp # Set the first handle. 
        bezier_points[0].co = sp # Set the first point. 
        
        for bez in self.beziers:
            bez.add_to_Blender(cu, stringed=True)

        # Set the last handle
        bezier_points[-1].handle_right = self.beziers[-1].points[-1]

    def intersections(self, threshold = 0.001):
        """
        Find the intersections within the curve. 
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

        # off = self.beziers[0].offset_curve(d)
        # off = self.beziers[1].offset_curve(d)
        # self.beziers[0].left_offset.add_to_Blender()
        # self.beziers[1].left_offset.add_to_Blender()

        for bez in self.beziers:
            bez.offset_curve(d)

        left_curves = []
        right_curves = []
        for bez in self.beziers:
            left_curves.append(bez.left_offset)
            right_curves.append(bez.right_offset)

        left_curve = left_curves[0]
        del left_curves[0]

        right_curve = right_curves[0]
        del right_curves[0]

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
        # TODO: This actually overwrites the left and right offset splines. 
        # This should not be done! 
        # self.endcap: butt, round, square
        # initial butt
        # self.stroke-linejoin: miter, round, bevel
        # initial miter
        # stroke-miterlimit 
        # inital 4
        # self.strokewidth
        strokewidth = .1
        endcap = "butt"
        stroke_linejoin = "miter"
        stroke_miterlimit = .1

        # If miter, then we should extend the tangents until they meet.
        self.offset_spline(strokewidth)
        
        if endcap == "butt":
            left_start = self.left_curve.start_point()
            right_start = self.right_curve.start_point()
            start_cap = Bezier(right_start, right_start, left_start, left_start)

            self.left_curve.prepend_bezier(start_cap)

            left_end = self.left_curve.end_point()
            right_end = self.right_curve.end_point()
            end_cap = Bezier(right_end, right_end, left_end, left_end)
            self.right_curve.append_bezier(end_cap)

        self.left_curve.append_spline(reversed(self.right_curve))
        # self.left_curve.add_to_Blender()
        # self.right_curve.add_to_Blender()
        self.left_curve.add_to_Blender()


class Curve():
    """Curve object, container class for splines."""
    def __init__(self, *splines, name="Curve"):
        self.splines = splines
        self.name = name
        self.location = location 

    def add_spline(self, spline):
        """Add a new spline to the curve object."""
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
