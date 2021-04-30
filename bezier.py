"""
Classes and utility functions for Bezier curves.  
"""

# Begin constants
# TODO: Move to separate module. 
THRESHOLD = 0.00001
# End constants

from mathutils import Vector, Matrix # TODO: Remove this and use explicit refs.
import mathutils 
import math
import bpy
import itertools
import operator

### Utility Functions ###
# TODO: Move to separate module. 

def are_overlapping(bbbox1, bbbox2):
    """
    Check if two bounding boxes are overlapping.
    Thinks about redoing this.
    """
    # 0      1      2      3
    # min_x, max_x, min_y, max_y 
    if bbbox1[0] >= bbbox2[1] or bbbox2[0] >= bbbox1[1] or bbbox1[2] >= bbbox2[3] or bbbox2[2] >= bbbox1[3]:
        return False
    else:
        return True

def quadratic_solve(a,b,c): 
    if a == 0:
        if b == 0:
            return (0,)
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
    """
    Recursive method used for finding the intersection between 
    two Bezier curves, c1, and c2. 
    """
    # TODO: Make this a method of Bezier and/or Curve. 
    # Rename to find_intersections_recursively?

    if c1._t2 - c1._t1 < threshold and c2._t2 - c2._t1 < threshold:
        return [((c1._t1 + c1._t2)/2 , (c2._t1 + c2._t2)/2)]

    cc1 = c1.split(0.5)
    cc2 = c2.split(0.5)
    pairs = itertools.product(cc1, cc2)
    pairs = [pair for pair in pairs if are_overlapping(pair[0].bounding_box, pair[1].bounding_box)] 

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
    # We need to translate with the object position to get the global 
    # positions. 
    # A better way would probably be to get the origin of current curve
    # and instead translate all objects to that origin. 
    # The new curves will all be created at the origin which might 
    # not be good. 
    location = cu.location
    p0 = cu.data.splines[0].bezier_points[0].co + location
    p1 = cu.data.splines[0].bezier_points[0].handle_right + location
    p2 = cu.data.splines[0].bezier_points[1].handle_left + location
    p3 = cu.data.splines[0].bezier_points[1].co + location
    return Bezier(p0, p1, p2, p3)

### End: Utility Functions ###

class Bezier():
    """
    Bezier curve of 3rd order. 
    p0, p1, p2, p3 are mathutils.Vector
    """
    # TODO: Use slots if that optimizes performance. 
    # TODO: Use @property for lazy evaluation of properties. See end of this file.
    #       Probably more Pythonic than using __getattr__ the way I have. 

    def __init__(self, p0, p1, p2, p3, t1 = 0, t2 = 1, name = None):
        """ 
        Initializes the curve and sets its points and degree. 
        The points should be mathutils.Vectors of some fixed dimension.
        The number of points should be 3 or higher. 
        For n points the curve is of order n-1. 
        """
        self.points = [p0, p1, p2, p3]
        
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
        string = self.name + '\n' + str(p[0]) + '\n' + str(p[1]) + '\n' + str(p[2]) + '\n' + str(p[3])
        return string

    def __call__(self, t):
        """ Returns the value at parameter t."""
        return self.function(t)
    
    def __getattr__(self, attribute):
        """The first time an attribute e.g. the derivative, is called,
        the attribute is calculated, i.e. lazy calculations.
        To prevent from doing a lot of calculations at init,
        and to prevent having to redo calculations multiple times. 
        """
        # TODO: The expectation is that we do not need all these for 
        # most curves. 
        # However, if this turns out to be wrong, we should consider 
        # changing this. 
        # TODO: Change to @properties instead. 
        if attribute == 'function':
            self.function = self._get_function()
            return self.function
        elif attribute == 'derivative':
            self.derivative = self._get_derivative()
            return self.derivative
        elif attribute == 'second_derivative':
            self.second_derivative = self._get_second_derivative()
            return self.second_derivative
        elif attribute == 'tangent':
            self.tangent = self._get_tangent()
            return self.tangent
        elif attribute == 'normal':
            self.normal = self._get_normal()
            return self.normal
        elif attribute == 'curvature':
            self.curvature = self._get_curvature()
            return self.curvature
        elif attribute == 'aligned':
            self.aligned = self._get_aligned()
            return self.aligned
        elif attribute == 'bounding_box':
            self.bounding_box = self._get_bounding_box()
            return self.bounding_box
        elif attribute == 'extrema':
            self.extrema = self._get_extrema()
            return self.extrema
        elif attribute == 'absolute_extrema':
            self.extrema = self._get_extrema(absolute=True)
            return self.extrema
        elif attribute == 'inflection_points':
            self.inflection_points = self._get_inflection_points()
            return self.inflection_points
        elif attribute == 'reduced':
            self.reduced = self._get_reduced()
            return self.reduced
        else:
            raise AttributeError(f"{self} has no attribute called {attribute}.")

    def _get_function(self):
        """Create the function the first time the Bezier curve is __call__:ed."""
        p = self.points
        def function(t): 
            return p[0] * (1 - t)**3 + 3 * p[1] * (1 - t)**2 * t + 3 * p[2] * (1 - t) * t**2 + p[3] * t**3

        return function

    def _get_derivative(self):
        """Returns the value of the derivative at t."""
        p = self.points
        def derivative(t): 
            return 3 * (p[1] - p[0]) * (1-t)**2 + 6 * (p[2] - p[1]) * (1-t) * t + 3 * (p[3] - p[2]) * t**2 
        
        return derivative

    def _get_second_derivative(self):
        """Evaluate the second derivative at parameter t."""
        p = self.points

        def second_derivative(t):
            return 6 * (p[0] - 2 * p[1] + p[2]) * (1-t) + 6 * (p[1] - 2 * p[2] + p[3]) * t

        return second_derivative

    def _get_tangent(self):
        """Return the tangent at parameter value t."""
        # TODO: Fix problem case. Linear curve? Still there?
        def tangent(t): 
            derivative = self.derivative(t) 
            if derivative.length > 0: 
                return derivative / derivative.length #mathutils.vector.length
            else:
                return derivative
        
        return tangent

    def _get_normal(self):
        """Return the normal at parameter value t."""
        def normal(t): 
            return self.tangent(t).cross(Vector((0,0,1)))

        return normal

    def _get_curvature(self):
        """Calculate the curvature at parameter t."""
        def curvature(t): 
            return (self.derivative(t)[0] * self.second_derivative(t)[1] - self.second_derivative(t)[0] * self.derivative(t)[1]) / (self.derivative(t).length)**(3/2)

        return curvature 

    def _get_aligned(self):
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

    def _get_bounding_box(self, tight=False):
        """Calculate the bounding box of the curve."""
        # TODO: Make it possible to calculate the tight bounding box if this
        # is deemed useful. 
        # TODO: Consider returning the box in some other format. 
        # TODO: Consider using a Rectangle class for this. 
        # Need to calculate the rotation and translation also! 
        extrema = self._get_extrema(absolute=tight) # TODO: Does not work! 
        min_x = self.__call__(extrema[0])[0]
        max_x = self.__call__(extrema[1])[0]
        min_y = self.__call__(extrema[2])[1]
        max_y = self.__call__(extrema[3])[1]
        return (min_x, max_x, min_y, max_y)

    def _get_extrema(self, absolute=False):
        """
        Returns the parameter values for the minimum and maximum of the curve
        in the x and y coordinate. 
        (tx_min, tx_min, ty_max, ty_max)
        If absolute we return the extremas of the aligned curve.
        """
        if absolute:
            p0, p1, p2, p3 = self.aligned
        else:
            p0, p1, p2, p3 = self.points

        a = (3 * (-p0 + 3 * p1 - 3 * p2 + p3))
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

    def _get_inflection_points(self):
        """Returns a list of parameter values where inflection points occurs. 
        The length of the list can be zero, one, or two. 
        """
        # TODO: Make this a lazily calculated data attribute. 
        # Align the curve to the x-axis to simplify equations. 
        # https://pomax.github.io/bezierinfo/#inflections
        p0, p1, p2, p3 = self.aligned
        # p1 = self.aligned[1]
        # p2 = self.aligned[2]
        # p3 = self.aligned[3]
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

    def _get_reduced(self):
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
        abs_extrema = self.absolute_extrema
        print('abs', abs_extrema)
        extrema = self.extrema
        print('ext', extrema)
        inflections = self.inflection_points
        total = list(set(abs_extrema + extrema + inflections)) # Remove any doubles. 
        total.sort() 
        total = [i for i in total if i > 0.001 and i < 0.999]
        if 0.0 not in total:
            total.insert(0, 0.0)
        if 1.0 not in total:
            total.append(1.0)

        print('total', total)
        curves = []
        for i in range(len(total)-1):
            curves.append(self.split(total[i], total[i+1]))

        return curves

    def _get_simplified(self):
        """Splits all reduced curves down the middle. Keeps splitting 
        until none of the curves has the middle of the curve too far
        from the center of the box created by the points. 
        """
        beziers = []
        for bezier in self.reduced:
            beziers += bezier.split(0.5)

        all_simple = False
        
        while not all_simple:
            all_simple = True
            new_set = []
            for bez in beziers:
                if bez._is_good(0.03):
                    new_set.append(bez)
                else:
                    all_simple = False
                    new_set += bez.split(0.5)
            beziers = new_set

        return beziers

    def _is_good(self, threshold = 0.05):
        """Check that Bezier(0.5) is not too far from the center 
        of the bounding box defined by the Bezier.points. 
        If the curve is straight, then we always return True. 
        """
        mid = self.__call__(0.5)
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
        
    def is_simple(self):
        """For 3D curves, this returns True if both handles 
        are on the same side of the curve. 
        """
        # TODO: Is this needed? Remove otherwise. 
        pass

    def start_point(self):
        """Returns the starting point."""
        return self.points[0]

    def end_point(self):
        """Returns the end point."""
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

            result = [Bezier(p[0], new1, new4, new6), Bezier(new6, new5, new3, p[3])]

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

    def add_to_Blender(self):
        """Adds the curve to Blender as splines."""
        # TODO: How should we choose which collection to add the curve to? 
        # TODO: Make it possible to choose the collection? 
        # TODO: Perhaps this method should be somewhere else, e.g. in a 
        #       a general class for Blender objects. 
        cu = bpy.data.curves.new(self.name, 'CURVE')
        ob = bpy.data.objects.new(self.name, cu)
        bpy.data.collections['Collection'].objects.link(ob)
        ob.data.resolution_u = 64
        cu.splines.new('BEZIER')
        spline = cu.splines[-1]
        
        spline.bezier_points[-1].co = self.points[0]
        spline.bezier_points[-1].handle_right = self.points[1]
        
        spline.bezier_points.add(1)
        spline.bezier_points[-1].co = self.points[3]
        spline.bezier_points[-1].handle_left = self.points[2]

    def intersections(self, bezier):
        """Returns a list of the parameters [(t, t'), ...] for the intersections 
        between self and bezier.
        """
        threshold = THRESHOLD
        c1 = self.reduced
        c2 = bezier.reduced
        pairs = itertools.product(c1, c2)
        
        pairs = [pair for pair in pairs if are_overlapping(pair[0].bounding_box, pair[1].bounding_box)]

        intersections = []
        for pair in pairs:
            result = curve_intersections(*pair, threshold)
            if len(result) > 0:
                intersections += result
        return intersections

    def self_intersections(self):
        """Returns a list of self intersections of the curve."""
        threshold = THRESHOLD
        c = self.reduced
        pairs = itertools.combinations(c, 2)
        pairs = [
            pair for pair in pairs 
            if are_overlapping(pair[0].bounding_box, pair[1].bounding_box)
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
        bb1 = self.bounding_box
        bb2 = bezier.bounding_box
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
        beziers = self._get_simplified()

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

        self.left_offset = Curve(*left_offset, name = self.name + ': Left')
        self.right_offset = Curve(*right_offset, name = self.name + ': Right')

        self.left_offset.add_to_Blender()
        self.right_offset.add_to_Blender()


    def is_clockwise(self):
        """Return True if the curve is clockwise."""
        # TODO: Perhaps this function is not needed! 
        a = self.points[1] - self.points[0]
        a.resize_2d()
        b = self.points[3] - self.points[0]
        b.resize_2d()
        if a.length == 0.0 or b.length == 0.0: # TODO: Fix this. 
            # Should instead check if a and b are colinear. 
            return True
        if a.angle_signed(b) >= 0:
            return True
        else:
            return False

class Curve(): 
    """A list of Bezier curves corresponds to a single curve object."""
    # TODO: Would it make sense to combine Curve and Bezier? 
    # Perhaps Curve can inherit Bezier? Perhaps not. It is not a Bezier curve 
    # (or rather, it might be a higher order Bezier curve). 
    # TODO: Handle offset. Utilize the offset method of Bezier when doing this. 
    # TODO: Handle closed curves. 
    # 1. End with z. -> Always toggle closed. 
    # 2. End point = start point but does not end with z. -> Toggle closed 
    #    only if filled.
    # 3. End points different and z not set. -> Toggle closed only if filled. 

    # For stroking:
    # If z is set, draw nice joint between start and end points. 
    # Else, draw stroke-endcap (butt, round, square) at start and end. 
    # Stroking does not care about whether the curve is filled or not. 

    def __init__(self, *beziers, is_closed = False, name = 'Curves'):
        self.beziers = beziers
        self.is_closed = is_closed
        self.name = name
        # Append curve-name to each Bezier. (Might reconsider this later). 
        for bezier in self.beziers:
            bezier.name = self.name + ':' + bezier.name 
            bezier._t1 = 0
            bezier._t2 = 1

    def add_curve(self, curve):
        """
        Add curve to the current (this) curve. 
        End point of this and start point of curve must coincide. 
        """
        if self.bezier[-1].end_point() == curve.start_point():
            # Set the curve points equal so that they are actually the same point. 
            curve.bezier[0].points[0] = self.bezier[-1].end_point() 
            for bezier in curve.beziers:
                self.bezier.append(bezier) 
        else:
            raise Exception('Start and end points of curves must be equal.')
    
    def add_bezier(self, bezier):
        """
        Add a single Bezier curve in the end of the curve. 
        End and start points must match. 
        """
        if self.end_point() == bezier.start_point():
            self.beziers.append(bezier)
        else:
            raise Exception('Start and end points of curves must be equal.')
    
    def toggle_closed(self):
        """
        Toggles the curve closed.
        """
        pass

    def is_closed(self):
        """
        Returns True/False if curve is closed/open.
        """
        pass
    
    def add_to_Blender(self):
        """
        Adds the curve to Blender as splines. 
        """
        # TODO: How should we choose which collection to add the curve to? 
        cu = bpy.data.curves.new(self.name, 'CURVE')
        ob = bpy.data.objects.new(self.name, cu)
        ob.data.resolution_u = 64
        bpy.data.collections['Collection'].objects.link(ob)
        cu.splines.new('BEZIER')
        spline = cu.splines[-1]
        
        # spline.bezier_points[-1].co = self.beziers[0].points[0]
        # spline.bezier_points[-1].handle_right = self.beziers[0].points[1]
        
        spline.bezier_points[-1].co = self.beziers[0].points[0]

        for bez in self.beziers:
            spline.bezier_points[-1].handle_right = bez.points[1]
            spline.bezier_points.add(1)
            spline.bezier_points[-1].co = bez.points[3]
            spline.bezier_points[-1].handle_left = bez.points[2]

    def intersections(self, threshold = 0.001):
        """
        Find the intersections within the curve. 
        The result is a dict e.g. {2 : [(t, t'), ...], (3, 4): [(...)]}
        where e.g. dict[2] gives the parameter values where self.beziers[2]
        intersect itself, and dict[(3,4)] (or dict[3,4]) gives a list of tuples of 
        parameter values where self.beziers[3] intersects self.beziers[4]. 
        """
        intersections = {}

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

    def start_point(self):
        """Return the starting point of the curve."""
        return self.beziers[0].start_point() 

    def end_point(self):
        """Return the end point of the curve."""
        return self.beziers[-1].end_point()

# TODO: Is the below a good idea?
# Idea: Create geometry classes here. 
# These can be initiated using the same parameters as SVGGeometry-classes. 
# These classes can be responsible for creating the geometry when adding to Blender.
# SVGArguments - The arguments from the corresponding SVG-class. These should be parsed and ready to use. E.g. any percentages should be resolved. 
# SVGTransform - A matrix corresponding to the full viewport transform. Calculated in the SVG classes and passed in here. 
# BlenderContext - The Blender context. 
# BlenderCollection - The collection which all geometry should be added to. 

def Geometry():
    def __init__(self, is_closed = True):
        self.is_closed = is_closed
    
    def add_to_Blender(self):
        pass

    def transform(self, transformation):
        self.points = [transformation @ point for point in self.points]


def Rectangle(Geometry):
    pass


def Ellipse(Geometry):
    pass


def Circle(Geometry):
    pass


def Line(Geometry):
    pass


def PolyLine(Geometry):
    pass


def Path(Geometry):
    pass


############# 


# A smart construction for making a property lazy. 
# def lazy_property(fn):
#     attr_name = '_lazy_' + fn.__name__

#     @property
#     def _lazy_property(self):
#         if not hasattr(self, attr_name):
#             setattr(self, attr_name, fn(self))
#         return getattr(self, attr_name)
#     return _lazy_property

# class Country:
#     def __init__(self, name, capital):
#         self.name = name
#         self.capital = capital

#     @lazy_property
#     def cities(self):
#         # expensive operation to get all the city names (API call)
#         print("cities property is called")
#         return ["city1", "city2"]

# china = Country("china", "beijing")
# print(china.capital)
## beijing
# print(china.cities)
## cities property is called
## ['city1', 'city2']
