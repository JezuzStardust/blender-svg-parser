"""
Classes and utility functions for Bezier curves.  
"""
from mathutils import Vector, Matrix
# import numpy as np
import math
import bpy

### General Utils ###

def quadratic_solve(a,b,c): 
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

class Bezier():
    """
    Bezier curve of 3rd order. 
    p0, p1, p2, p3 are mathutils.Vector
    """
    # TODO: Make it possible to use also numpy.array if this is an improvement.
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
            self.name = 'BezierCurve'
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
        the function is created. 
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

    def _get_function(self):
        """
        Create the function the first time the Bezier curve is __call__:ed. 
        """
        print('function created')
        p = self.points
        def function(t): 
            return p[0] * (1 - t)**3 + 3 * p[1] * (1 - t)**2 * t + 3 * p[2] * (1 - t) * t**2 + p[3] * t**3

        return function

    def _get_derivative(self):
        """
        Returns the value of the derivative at t. 
        """
        print('derivative created')
        p = self.points
        def derivative(t): 
            return 3 * (p[1] - p[0]) * (1-t)**2 + 6 * (p[2] - p[1]) * (1-t) * t + 3 * (p[3] - p[2]) * t**2 
        
        return derivative

    def _get_second_derivative(self):
        """
        Evaluate the second derivative at parameter t. 
        """
        print('second derivative created')
        p = self.points

        def second_derivative(t):
            return 6 * (p[0] - 2 * p[1] + p[2]) * (1-t) + 6 * (p[1] - 2 * p[2] + p[3]) * t

        return second_derivative

    def _get_tangent(self):
        """
        Return the tangent at parameter value t. 
        """
        print('tangent created')
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
        print('curvature created')
        def curvature(t): 
            return (self.derivative(t)[0] * self.second_derivative(t)[1] - self.second_derivative(t)[0] * self.derivative(t)[1]) / (self.derivative(t).length)**(3/2)

        return curvature 

    def is_simple(self):
        """
        For 3D curves, this returns True if both handles 
        are on the same side of the curve. 
        """
        pass

    def extrema(self):
        """
        Returns the parameter values for the minimum and maximum of the curve
        in the x and y coordinate. 
        (tx_min, tx_min, ty_max, ty_max)
        """
        p0 = self.points[0]
        p1 = self.points[1]
        p2 = self.points[2]
        p3 = self.points[3]

        a = 3 * (-p0 + 3 * p1 - 3 * p2 + p3) 
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

        return (tx_min, tx_max, ty_min, ty_max)

    def get_inflection_points(self):
        """
        Returns a list of parameter values where inflection points occurs. 
        ((x1, y1, t1), (x2, y2, t2)). 
        The length of the list can be zero, one, or two. 
        """
        pass

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

    # def map(self, v, ds, de, ts, te):
    #     """
    #     Maps the parameter of a point on a split curve 
    #     to the whole curve or vice versa.
    #     If ds = 0, de = 1, we map from the split to the parent.  
    #     In this case, ts is the starting parameter of the parent (ts = 0 
    #     if the parent was the original curve, i.e. never splitted), 
    #     and te is the end parameter value. 
    #     If ts = 0, and te = 1, we map a parameter from the parent to the split.
    #     """
    #     # TODO: Split this up into two separate functions. 
    #     return ts + (te - ts) * (v - ds) / (de - ds)

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

    def bounding_box(self):
        """
        Calculate the bounding box of the curve. 
        """
        # TODO: Make it possible to calculate the tight bounding box if this
        # is deemed useful. 
        extrema = self.extrema()
        min_x = self.__call__(extrema[0])[0]
        max_x = self.__call__(extrema[1])[0]
        min_y = self.__call__(extrema[2])[1]
        max_y = self.__call__(extrema[3])[1]
        return (min_x, max_x, min_y, max_y)


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

        


class Curve(): 
    pass


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
