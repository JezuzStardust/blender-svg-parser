"""
Classes and utility functions for Bezier curves.  
"""
from mathutils import Vector, Matrix
import mathutils
import math
import bpy
import itertools
import operator

### Utility Functions ###

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
        # case from the caller instead.
        return () 

def curve_intersections(c1, c2, threshold):
    """
    Recursive method used for finding the intersection between 
    two Bezier curves, c1, and c2. 
    """
    # TODO: Make this a static method of Bezier. Or maybe not. It is till 
    # limited to this module. 
    # TODO: Rename with something like find_intersections_recursively(..)

    if c1._t2 - c1._t1 < threshold and c2._t2 - c2._t1 < threshold:
        return [((c1._t1 + c1._t2)/2 , (c2._t1 + c2._t2)/2)]

    cc1 = c1.split(0.5)
    cc2 = c2.split(0.5)
    # for i in cc1:
    #     i.name = c1.name[0] + ':' + str(i._t1) + '-' + str(i._t2)
    # for i in cc2:
    #     i.name = c2.name[0] + ':' + str(i._t1) + '-' + str(i._t2)
    pairs = itertools.product(cc1, cc2)
    pairs = [pair for pair in pairs if are_overlapping(pair[0].bounding_box, pair[1].bounding_box)] 

    results = [] 
    if len(pairs) == 0:
        return results
    
    for pair in pairs:
        results += curve_intersections(pair[0], pair[1], threshold)
    results = filter_duplicates(results, threshold)
    return results

def filter_duplicates(tuples, threshold = 0.01):
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

# Temporary during development. Probably not useful later. 

def bezier_from_Blender(name):
    cu = bpy.data.collections['Collection'].objects[name]
    p0 = cu.data.splines[0].bezier_points[0].co
    p1 = cu.data.splines[0].bezier_points[0].handle_right
    p2 = cu.data.splines[0].bezier_points[1].handle_left
    p3 = cu.data.splines[0].bezier_points[1].co
    return Bezier(p0, p1, p2, p3)

### End: Utility Functions ###

# TODO: Use slots if that optimizes performance. 
# TODO: Use @property for lazy evaluation of properties.

class Bezier():
    """
    Bezier curve of 3rd order. 
    p0, p1, p2, p3 are mathutils.Vector
    """
    # TODO: Consider using slots. 
    # TODO: Consider using property for both lazy evaluation of attributes
    #       and automatic updating of e.g. bounding_box when needed. 
    #       Probably more Pythonic than using __getattr__ the way I have. 

    def __init__(self, p0, p1, p2, p3, t1 = 0, t2 = 1, name = None):
        """ 
        Initializes the curve and sets its points and degree. 
        The points should be vectors of some fixed dimension.
        The number of points should be 3 or higher. 
        For n points the curve is of order n-1. 
        """
        self.points = [p0, p1, p2, p3]
        
        if name:
            self.name = name
        else:
            self.name = 'Bezier'
        # If this is a split of another curve, then _t1 and _t2 gives the 
        # parameter range of the original curve. 
        # Needed for keeping track of intersections. 
        self._t1 = t1
        self._t2 = t2

    def __repr__(self):
        p = self.points
        string = self.name + '\n' + str(p[0]) + '\n' + str(p[1]) + '\n' + str(p[2]) + '\n' + str(p[3])
        return string

    def __call__(self, t):
        """
        Returns the value at parameter t. 
        """
        return self.function(t)
    
    def __getattr__(self, attribute):
        """
        The first time an attribute e.g. the derivative, is called,
        the attribute is calculated, i.e. lazy calculations.
        To prevent from doing a lot of calculations at init,
        and to prevent having to redo calculations multiple times. 
        """
        # TODO: The expectation is that we do not need all these for 
        # most curves. 
        # However, if this turns out to be wrong, we should consider 
        # changing this. 
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
        elif attribute == 'inflection_points':
            self.inflection_points = self._get_inflection_points()
            return self.inflection_points
        elif attribute == 'reduced':
            self.reduced = self._get_reduced()
            return self.reduced
        else:
            raise AttributeError(f"{self} has no attribute called {attribute}.")

    def _get_function(self):
        """
        Create the function the first time the Bezier curve is __call__:ed. 
        """
        p = self.points
        def function(t): 
            return p[0] * (1 - t)**3 + 3 * p[1] * (1 - t)**2 * t + 3 * p[2] * (1 - t) * t**2 + p[3] * t**3

        return function

    def _get_derivative(self):
        """
        Returns the value of the derivative at t. 
        """
        p = self.points
        def derivative(t): 
            return 3 * (p[1] - p[0]) * (1-t)**2 + 6 * (p[2] - p[1]) * (1-t) * t + 3 * (p[3] - p[2]) * t**2 
        
        return derivative

    def _get_second_derivative(self):
        """
        Evaluate the second derivative at parameter t. 
        """
        p = self.points

        def second_derivative(t):
            return 6 * (p[0] - 2 * p[1] + p[2]) * (1-t) + 6 * (p[1] - 2 * p[2] + p[3]) * t

        return second_derivative

    def _get_tangent(self):
        """
        Return the tangent at parameter value t. 
        """
        def tangent(t): 
            derivative = self.derivative(t) 
            return derivative / derivative.length
        
        return tangent

    def _get_normal(self):
        """
        Return the normal at parameter value t. 
        """
        def normal(t): 
            return self.tangent(t).cross(Vector((0,0,1)))

        return normal

    def _get_curvature(self):
        """
        Calculate the curvature at parameter t. 
        """
        def curvature(t): 
            return (self.derivative(t)[0] * self.second_derivative(t)[1] - self.second_derivative(t)[0] * self.derivative(t)[1]) / (self.derivative(t).length)**(3/2)

        return curvature 

    def _get_aligned(self):
        """
        Returns the points of the corresponding aligned curve. 
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

    def _get_bounding_box(self):
        """
        Calculate the bounding box of the curve. 
        """
        # TODO: Make it possible to calculate the tight bounding box if this
        # is deemed useful. 
        # TODO: Consider returning the box in some other format. 
        # TODO: Consider using a Rectangle class for this. 
        extrema = self.extrema
        min_x = self.__call__(extrema[0])[0]
        max_x = self.__call__(extrema[1])[0]
        min_y = self.__call__(extrema[2])[1]
        max_y = self.__call__(extrema[3])[1]
        return (min_x, max_x, min_y, max_y)

    def _get_extrema(self):
        """
        Returns the parameter values for the minimum and maximum of the curve
        in the x and y coordinate. 
        (tx_min, tx_min, ty_max, ty_max)
        """
        p0 = self.points[0]
        p1 = self.points[1]
        p2 = self.points[2]
        p3 = self.points[3]

        a = (3 * (-p0 + 3 * p1 - 3 * p2 + p3))
        b = 6 * (p0 - 2*p1 + p2)
        c = 3*(p1 - p0) 
        
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
        """
        Returns a list of parameter values where inflection points occurs. 
        The length of the list can be zero, one, or two. 
        """
        # TODO: Make this a lazily calculated data attribute. 
        # Align the curve to the x-axis to simplify equations. 
        # https://pomax.github.io/bezierinfo/#inflections
        p0 = self.aligned[0]
        p1 = self.aligned[1]
        p2 = self.aligned[2]
        p3 = self.aligned[3]
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
        """
        Splits the curve at each extrema and each inflection point
        and returns a list of these. 
        """
        extrema = self.extrema
        inflections = self.inflection_points
        total = list(set(extrema + inflections)) # Remove any doubles. 
        total.sort() 
        if 0 not in total:
            total.insert(0, 0.0)
        if 1 not in total:
            total.append(1)

        curves = []
        for i in range(len(total)-1):
            curves.append(self.split(total[i], total[i+1]))

        return curves

    def is_simple(self):
        """
        For 3D curves, this returns True if both handles 
        are on the same side of the curve. 
        """
        pass

    def start_point(self):
        """
        Returns the starting point. 
        """
        return self.points[0]

    def end_point(self):
        """
        Returns the end point.
        """
        return self.points[3]

    def split(self, t1, t2 = None):
        """
        Splits the Bezier curve at the parameter(s) t1 (and t2). 
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
        """
        Returns the parameter in the splitted curve 
        corresponding to the parameter t of the whole (unsplitted) curve. 
        t2 is the parameter value which we want to map and 
        the split curve runs from parameter ds to de of the whole curve. 

        Ref: http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node13.html
        """
        # TODO: This does not really depend on self. 
        # Perhaps it would make sense to move this out of the class and 
        # perhaps into a utility module. 
        return (t - ds) / (de - ds)

    def map_split_to_whole(self, t):
        """ 
        Returns the parameter value of the whole curve, 
        corresponding to the parameter t in the splitted curve. 
        """
        return self._t1 + t * (self._t2 - self._t1) 

    def tight_bounding_box(self):
        pass

    def add_to_Blender(self):
        """
        Adds the curve to Blender as splines. 
        """
        # TODO: How should we choose which collection to add the curve to? 
        cu = bpy.data.curves.new(self.name, 'CURVE')
        ob = bpy.data.objects.new(self.name, cu)
        bpy.data.collections['Collection'].objects.link(ob)
        cu.splines.new('BEZIER')
        spline = cu.splines[-1]
        
        spline.bezier_points[-1].co = self.points[0]
        spline.bezier_points[-1].handle_right = self.points[1]
        
        spline.bezier_points.add(1)
        spline.bezier_points[-1].co = self.points[3]
        spline.bezier_points[-1].handle_left = self.points[2]

    def reduce(self):
        """
        Splits the curve up in simple parts
        """
        results = [] 
        extrema = self.extrema() 

    def intersections(self, bezier, threshold = 0.00001):
        """
        Returns a list of the parameters [(t, t'), ...] for the intersections 
        between self and bezier.
        """
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
        """
        Returns a list of self intersections of the curve. 
        """
        threshold = 0.00001

        c = self.reduced
        pairs = itertools.combinations(c, 2)

        pairs = [pair for pair in pairs if are_overlapping(pair[0].bounding_box, pair[1].bounding_box)]

        intersections = []
        for pair in pairs:
            result = curve_intersections(*pair, threshold)
            if len(result) > 0:
                intersections += result
        return intersections

    def overlaps(self, bezier):
        """
        Check if the bounding box of self and bezier overlaps. 
        """
        # 0      1      2      3
        # min_x, max_x, min_y, max_y 
        bb1 = self.bounding_box
        bb2 = bezier.bounding_box
        if bb1[0] >= bb2[1] or bb2[0] >= bb1[1] or bb1[2] >= bb2[3] or bb2[2] >= bb1[3]:
            return False
        else:
            return True

    def offset_curve(self, d):
        """
        Returns an inner and outer curve each offset a distance d from self.
        """
        left_offset = []
        right_offset = []
        m1 = Matrix().Rotation(math.radians(90), 3, 'Z')
        m2 = Matrix().Rotation(-math.radians(90), 3, 'Z')
        first = True # For the first curve we need to create the first point.
        test = True
        beziers = []

        beziers = self.reduced
        # TODO: Consider if there is some smarter way to do this loop. 
        # In case the curve was not suitable for offsetting, 
        # we split again and try again. 
        # Perhaps we can find a watertight test for splitting the curve
        # in advance and simply add that to Bezier.reduced so that we are
        # sure offsettinga will work. 
        i = 0
        while i < len(beziers):
            bezier = beziers[i]
            p = bezier.points
            n0 = bezier.normal(0)
            n1 = bezier.normal(1)
            if bezier.is_clockwise():
                m = m1
                l = 1
            else:
                m = m2
                l = -1 
            nn = l * m @ (p[2] - p[1])
            nn.normalize()

            # TODO: Go back to making a0 and b0 the same as the previous endpoints. 
            # If the intersect_line_line does not work, then we split to avoid 
            # problems. 
            a0 = p[0] - n0 * d
            a1p = p[1] - n0 * d 
            a1pp = p[1] + nn * d
            a2p = p[2] - n1 * d
            a2pp = p[2] + nn * d
            a3 = p[3] - n1 * d

            a1i = mathutils.geometry.intersect_line_line(a0, a1p, a1pp, a2pp)
            a2i = mathutils.geometry.intersect_line_line(a3, a2p, a1pp, a2pp)
            if a1i is None or a2i is None:
                beziers[i:i+1] = bezier.split(0.5)
                continue
            a1 = a1i[0]
            a2 = a2i[0]

            b0 = p[0] + n0 * d
            b1p = p[1] + n0 * d 
            b1pp = p[1] - nn * d
            b2p = p[2] + n1 * d
            b2pp = p[2] - nn * d
            b3 = p[3] + n1 * d

            b1i = mathutils.geometry.intersect_line_line(b0, b1p, b1pp, b2pp)
            b2i = mathutils.geometry.intersect_line_line(b3, b2p, b1pp, b2pp)
            if not b1i or not b2i:
                beziers[i:i+1] = beziers.split(0.5)
            b1 = b1i[0]
            b2 = b2i[0]

            a = Bezier(a0, a1, a2, a3)
            b = Bezier(b0, b1, b2, b3)
            left_offset.append(a)
            right_offset.append(b)
            bezier.add_to_Blender()
            i += 1

        self.left_offset = Curve(*left_offset, name = self.name + ': Left')
        self.right_offset = Curve(*right_offset, name = self.name + ': Right')
        
        self.left_offset.add_to_Blender()
        self.right_offset.add_to_Blender()

        # For the following curves, they start where the previous one ended.
        # for bezier in beziers:
        #     p = bezier.points 
        #     n0 = bezier.normal(0)
        #     n1 = bezier.normal(1) 
        #     if bezier.is_clockwise():
        #         m = m1
        #         l = 1
        #     else:
        #         m = m2
        #         l = -1

        #     nn = l * m @ (p[2] - p[1]) 
        #     nn.normalize() 

        #     if first: 
        #         a0 = p[0] - n0 * d
        #         b0 = p[0] + n0 * d
        #         first = False
        #     else: 
        #         a0 = left_offset[-1].points[-1]
        #         b0 = right_offset[-1].points[-1]

        #     a1p  = p[1] - n0 * d 
        #     a1pp = p[1] + nn * d
        #     a2p  = p[2] - n1 * d
        #     a2pp = p[2] + nn * d
        #     a3   = p[3] - n1 * d
        #     a1 = mathutils.geometry.intersect_line_line(a0, a1p, a1pp, a2pp)[0]
        #     a2 = mathutils.geometry.intersect_line_line(a3, a2p, a1pp, a2pp)[0]
            
        #     b1p  = p[1] + n0 * d 
        #     b1pp = p[1] - nn * d
        #     b2p  = p[2] + n1 * d
        #     b2pp = p[2] - nn * d
        #     b3   = p[3] + n1 * d
        #     b1 = mathutils.geometry.intersect_line_line(b0, b1p, b1pp, b2pp)[0]
        #     b2 = mathutils.geometry.intersect_line_line(b3, b2p, b1pp, b2pp)[0]

        #     left_offset.append(Bezier(a0, a1, a2, a3))
        #     right_offset.append(Bezier(b0, b1, b2, b3))

        # self.left_offset = Curve(*left_offset, name = 'Left')
        # self.right_offset = Curve(*right_offset, name = 'Right')

        # self.left_offset.add_to_Blender()
        # self.right_offset.add_to_Blender()

    def is_clockwise(self):
        # TODO: Check if curve is_simple. 
        a = self.points[1] - self.points[0]
        a.resize_2d()
        b = self.points[3] - self.points[0]
        b.resize_2d()
        if a.angle_signed(b) >= 0:
            return True
        else:
            return False


class Curve(): 
    """
    A list of Bezier curves corresponds to a single curve object. 
    """
    # TODO: Would it make sense to combine Curve and Bezier. 
    # Perhaps Curve can inherit Bezier? 

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

    def intersections(self, threshold = 0.00001):
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
        """
        Return the starting point of the curve. 
        """
        return self.beziers[0].start_point() 

    def end_point(self):
        """
        Return the end point of the curve. 
        """
        return self.beziers[-1].end_point()

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


def find_parameter_at_position(point1, point2, x, y):  
    pass

def classify_curve(p1, p2, p3, p4): 
    """
    Classify the curve depending into following:
    1. Simple curve
    2. One inflection point.
    3. Two inflection points. 
    4. One loop.
    5. One cusp. 
    """

    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    x3 = p3[0]
    y3 = p3[1]
    x4 = p4[0]
    y4 = p4[1]

    # Check if collinear
    if p1[0] * (p2[1] - p3[1]) + p2[0] * (p2[1] - p1[1]) + p3[0] * (p1[1] - p2[1]) == 0:
        print('Collinear, switching direction.')
        p1, p2, p3, p4 = p4, p3, p2, p1

    # if x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) == 0: 
    #     print('Collinear, switching direction')
    #     x1, y1, x2, y2, x3, y3, x4, y4 = x4, y4, x3, y3, x2, y2, x1, y1

    #Translated
    
    p2 -= p1 
    p3 -= p1
    p4 -= p1
    p1 -= p1 

    # x2 -= x1
    # y2 -= y1

    # x3 -= x1
    # y3 -= y1

    # x4 -= x1
    # y4 -= y1

    # x1 = 0
    # y1 = 0

    if p2[1] != 0: 
        y42 = p4[1] / p2[1]
        y32 = p3[1] / p2[1] 
        x43 = ( p4[0] - p2[0] * y42) / (p3[0] - p2[0] * y32) 
        result = (x43, y42 + x43 * (1 - y32))
    else: 
        print('Degenerate case')
        print('Rotate by e.g. 45 degrees and try again')

    # if y2 != 0: 
    #     y42 = y4 / y2
    #     y32 = y3 / y2 
    #     x43 = ( x4 - x2 * y42) / (x3 - x2 * y32) 
    #     result = (x43, y42 + x43 * (1 - y32))
    # else: 
    #     print('Degenerate case')
    #     print('Rotate by e.g. 45 degrees and try again')
    

    # This function should really return a string defining the case I think. 


    # Degenerate cases
    # 1. All on single point -> Point. We can remove this. 
    # 2. All points collinear -> We have a straight line. 
    # 3. B0 B1 B2 collinear. Reverse order of the points B3, B2, B1, B0 (E.g. if the B1 = B0, i.e. handle on same position). 

    # Also, when constructing the transformation we need to take care when moving 
    # the second point B1 to (0,1). In case B1.y is zero (after translation -B0.x, -B0.y) we instead do a rotation and scale. 

    # All points collinear -> Straight curve. 
    # B1 = B2 -> Full cubic with inflections at each end point. 
    # 
    # TODO: Handle the case where x3 - x2 * y32 = 0? Is this just the case 3 above?

    return result



# Rewrite the lazy properties. 
"""
self.cities = lazy_property(cities) 
lazy_property(cities) -
    attr_name = '_lazy_cities' 

    def _lazy_property(cities) - 
        if not hasattr(self, '_lazy_cities'):
            setattr(self, '_lazy_cities', cities(self)) 
        return getattr(self, attr_name) 
    return _lazy_property 

        
"""

def lazy_property(fn):
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property

# A smart construction for making a property lazy. 


class Country:
    def __init__(self, name, capital):
        self.name = name
        self.capital = capital

    @lazy_property
    def cities(self):
        # expensive operation to get all the city names (API call)
        print("cities property is called")
        return ["city1", "city2"]

# china = Country("china", "beijing")
# print(china.capital)
## beijing
# print(china.cities)
## cities property is called
## ['city1', 'city2']
