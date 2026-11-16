import numpy as np

# --- Helper classes and functions to mimic the Paper.js environment ---
# These are simplified versions to make the translation runnable.
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def is_close(self, other, epsilon):
        return np.linalg.norm(np.array([self.x, self.y]) - np.array([other.x, other.y])) < epsilon

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class Curve:
    @staticmethod
    def get_values(curve, matrix=None):
        # In a full implementation, this would apply a matrix.
        # For this translation, we assume values are already in a flat array format.
        return curve

    @staticmethod
    def is_straight(v, threshold=1e-6):
        # A simple test for flatness: check if handles are on the line between endpoints.
        p0, p1, p2, p3 = np.array(v).reshape(-1, 2)
        line_vec = p3 - p0
        dist1 = np.linalg.norm(np.cross(line_vec, p1 - p0))
        dist2 = np.linalg.norm(np.cross(line_vec, p2 - p0))
        return dist1 < threshold and dist2 < threshold

    @staticmethod
    def subdivide(v, t):
        # De Casteljau subdivision at a given parameter t
        p0, p1, p2, p3 = np.array(v).reshape(-1, 2)
        q0 = (1 - t) * p0 + t * p1
        q1 = (1 - t) * p1 + t * p2
        q2 = (1 - t) * p2 + t * p3
        r0 = (1 - t) * q0 + t * q1
        r1 = (1 - t) * q1 + t * q2
        s0 = (1 - t) * r0 + t * r1

        v1 = np.concatenate([p0, q0, r0, s0])
        v2 = np.concatenate([s0, r1, q2, p3])
        return [v1, v2]

    @staticmethod
    def get_part(v, t_start, t_end):
        if t_start == 0 and t_end == 1:
            return v
        
        # A full implementation would subdivide twice, once at t_end then again at t_start/t_end
        # This is a simplified version for translation purposes
        return v
    
    @staticmethod
    def get_point(v, t):
        p0, p1, p2, p3 = np.array(v).reshape(-1, 2)
        return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

    @staticmethod
    def get_time_of(v, point):
        # Placeholder for finding the t parameter for a given point on a curve.
        # This is a complex problem and a full solution is not provided here.
        # Returns a mock value or None.
        return 0.5 # Mock value

    @staticmethod
    def solve_cubic(rv, component_index, value, roots, min_t, max_t):
        # Placeholder for a cubic solver.
        # This would solve the cubic polynomial for a specific component (e.g., y=0)
        roots.append((min_t + max_t) / 2) # Mock value
        
    @staticmethod
    def classify(v):
        # Placeholder for curve classification
        return {'type': 'line', 'roots': []}

class Line:
    @staticmethod
    def get_signed_distance(x1, y1, x2, y2, px, py):
        # Get signed distance from a point to a line segment
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        return (dy * (px - x1) - dx * (py - y1)) / np.sqrt(dx**2 + dy**2)

    @staticmethod
    def get_distance(x1, y1, dx, dy, px, py, squared=False):
        # Distance from a point to an infinite line
        numerator = abs(dy * (px - x1) - dx * (py - y1))
        denominator = np.sqrt(dx**2 + dy**2)
        if denominator == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        dist = numerator / denominator
        return dist**2 if squared else dist
    
    @staticmethod
    def intersect(x1, y1, x2, y2, x3, y3, x4, y4):
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
        return None

class Numerical:
    EPSILON = 1e-12
    CURVETIME_EPSILON = 1e-8
    GEOMETRIC_EPSILON = 1e-8
    
    @staticmethod
    def is_zero(value, epsilon=EPSILON):
        return abs(value) < epsilon

class CurveLocation:
    def __init__(self, curve, t, point=None, overlap=False):
        self.curve = curve
        self.t = t
        self.point = point
        self.overlap = overlap
        self._intersection = None

    @staticmethod
    def insert(locations, loc, sorted=False):
        locations.append(loc) # simplified insert for this translation

# --- Translated functions ---

def add_location(locations, include, c1, t1, c2, t2, overlap=False):
    # This function is not fully translated since it relies on Curve's Path
    # and neighbor checks which are not part of the scope.
    # It adds two intersecting CurveLocation objects to the list.
    t_min = Numerical.CURVETIME_EPSILON
    t_max = 1 - t_min

    if t1 is not None and t1 >= 0 and t1 <= 1:
        if t2 is not None and t2 >= 0 and t2 <= 1:
            loc1 = CurveLocation(c1, t1, None, overlap)
            loc2 = CurveLocation(c2, t2, None, overlap)
            loc1._intersection = loc2
            loc2._intersection = loc1
            locations.append(loc1)


def add_curve_intersections(v1, v2, c1, c2, locations, include, flip, recursion, calls, t_min, t_max, u_min, u_max):
    if calls >= 4096 or recursion >= 40:
        return calls + 1

    fat_line_epsilon = 1e-9

    q0x, q0y, _, _, _, _, q3x, q3y = v2
    
    # Calculate fat-line L for curve 2
    d1 = Line.get_signed_distance(q0x, q0y, q3x, q3y, v2[2], v2[3])
    d2 = Line.get_signed_distance(q0x, q0y, q3x, q3y, v2[4], v2[5])
    factor = 0.75 if d1 * d2 > 0 else 4/9
    d_min = factor * min(0, d1, d2)
    d_max = factor * max(0, d1, d2)

    # Calculate non-parametric bezier curve D for curve 1
    dp0 = Line.get_signed_distance(q0x, q0y, q3x, q3y, v1[0], v1[1])
    dp1 = Line.get_signed_distance(q0x, q0y, q3x, q3y, v1[2], v1[3])
    dp2 = Line.get_signed_distance(q0x, q0y, q3x, q3y, v1[4], v1[5])
    dp3 = Line.get_signed_distance(q0x, q0y, q3x, q3y, v1[6], v1[7])

    hull_top, hull_bottom = get_convex_hull(dp0, dp1, dp2, dp3)

    if (Numerical.is_zero(d1) and Numerical.is_zero(d2) and
        Numerical.is_zero(dp0) and Numerical.is_zero(dp1) and
        Numerical.is_zero(dp2) and Numerical.is_zero(dp3)):
        return calls

    t_min_clip = clip_convex_hull(hull_top, hull_bottom, d_min, d_max)
    t_max_clip = clip_convex_hull(hull_top[::-1], hull_bottom[::-1], d_min, d_max)

    if t_min_clip is None or t_max_clip is None:
        return calls

    t_min_new = t_min + (t_max - t_min) * t_min_clip
    t_max_new = t_min + (t_max - t_min) * t_max_clip

    if max(u_max - u_min, t_max_new - t_min_new) < fat_line_epsilon:
        t = (t_min_new + t_max_new) / 2
        u = (u_min + u_max) / 2
        add_location(locations, include,
                     c2 if flip else c1, u if flip else t,
                     c1 if flip else c2, t if flip else u)
    else:
        v1_part = Curve.get_part(v1, t_min_clip, t_max_clip)
        if t_max_clip - t_min_clip > 0.8:
            if t_max_new - t_min_new > u_max - u_min:
                parts = Curve.subdivide(v1_part, 0.5)
                t = (t_min_new + t_max_new) / 2
                calls = add_curve_intersections(
                        v2, parts[0], c2, c1, locations, include, not flip,
                        recursion, calls, u_min, u_max, t_min_new, t)
                calls = add_curve_intersections(
                        v2, parts[1], c2, c1, locations, include, not flip,
                        recursion, calls, u_min, u_max, t, t_max_new)
            else:
                parts = Curve.subdivide(v2, 0.5)
                u = (u_min + u_max) / 2
                calls = add_curve_intersections(
                        parts[0], v1_part, c2, c1, locations, include, not flip,
                        recursion, calls, u_min, u, t_min_new, t_max_new)
                calls = add_curve_intersections(
                        parts[1], v1_part, c2, c1, locations, include, not flip,
                        recursion, calls, u, u_max, t_min_new, t_max_new)
        else:
            if u_max - u_min >= fat_line_epsilon:
                calls = add_curve_intersections(
                        v2, v1_part, c2, c1, locations, include, not flip,
                        recursion, calls, u_min, u_max, t_min_new, t_max_new)
            else:
                calls = add_curve_intersections(
                        v1_part, v2, c1, c2, locations, include, flip,
                        recursion, calls, t_min_new, t_max_new, u_min, u_max)
    return calls


def get_convex_hull(dq0, dq1, dq2, dq3):
    p0 = [0, dq0]
    p1 = [1 / 3, dq1]
    p2 = [2 / 3, dq2]
    p3 = [1, dq3]

    dist1 = dq1 - (2 * dq0 + dq3) / 3
    dist2 = dq2 - (dq0 + 2 * dq3) / 3
    hull = []

    if dist1 * dist2 < 0:
        hull = [[p0, p1, p3], [p0, p2, p3]]
    else:
        dist_ratio = dist1 / dist2 if dist2 != 0 else float('inf')
        if dist_ratio >= 2:
            hull = [[p0, p1, p3], [p0, p3]]
        elif dist_ratio <= 0.5:
            hull = [[p0, p2, p3], [p0, p3]]
        else:
            hull = [[p0, p1, p2, p3], [p0, p3]]
    
    if (dist1 or dist2) < 0:
        return hull[1], hull[0]
    else:
        return hull[0], hull[1]


def clip_convex_hull(hull_top, hull_bottom, d_min, d_max):
    if hull_top[0][1] < d_min:
        return clip_convex_hull_part(hull_top, True, d_min)
    elif hull_bottom[0][1] > d_max:
        return clip_convex_hull_part(hull_bottom, False, d_max)
    else:
        return hull_top[0][0]


def clip_convex_hull_part(part, top, threshold):
    px, py = part[0]
    for i in range(1, len(part)):
        qx, qy = part[i]
        if (top and qy >= threshold) or (not top and qy <= threshold):
            if qy == threshold:
                return qx
            else:
                return px + (threshold - py) * (qx - px) / (qy - py)
        px, py = qx, qy
    return None


def get_curve_line_intersections(v, px, py, vx, vy):
    is_zero = Numerical.is_zero
    if is_zero(vx) and is_zero(vy):
        t = Curve.get_time_of(v, Point(px, py))
        return [t] if t is not None else []
    
    angle = np.arctan2(-vy, vx)
    sin, cos = np.sin(angle), np.cos(angle)
    
    rv = np.zeros(8)
    for i in range(4):
        x = v[i*2] - px
        y = v[i*2 + 1] - py
        rv[i*2] = x * cos - y * sin
        rv[i*2 + 1] = x * sin + y * cos
    
    roots = []
    # Solve the cubic for y = 0
    Curve.solve_cubic(rv, 1, 0, roots, 0, 1)
    return roots


def add_curve_line_intersections(v1, v2, c1, c2, locations, include, flip):
    x1, y1, _, _, _, _, x2, y2 = v2
    roots = get_curve_line_intersections(v1, x1, y1, x2 - x1, y2 - y1)
    
    for t1 in roots:
        p1 = Curve.get_point(v1, t1)
        t2 = Curve.get_time_of(v2, p1)
        if t2 is not None:
            add_location(locations, include,
                         c2 if flip else c1, t2 if flip else t1,
                         c1 if flip else c2, t1 if flip else t2)


def add_line_intersection(v1, v2, c1, c2, locations, include):
    pt = Line.intersect(v1[0], v1[1], v1[6], v1[7], v2[0], v2[1], v2[6], v2[7])
    if pt is not None:
        add_location(locations, include,
                     c1, Curve.get_time_of(v1, Point(pt[0], pt[1])),
                     c2, Curve.get_time_of(v2, Point(pt[0], pt[1])))


def get_curve_intersections(v1, v2, c1, c2, locations, include):
    epsilon = Numerical.EPSILON
    v1_np = np.array(v1).reshape(-1, 2)
    v2_np = np.array(v2).reshape(-1, 2)
    
    # Bounding box check
    if (np.max(v1_np[:, 0]) + epsilon > np.min(v2_np[:, 0]) and
        np.min(v1_np[:, 0]) - epsilon < np.max(v2_np[:, 0]) and
        np.max(v1_np[:, 1]) + epsilon > np.min(v2_np[:, 1]) and
        np.min(v1_np[:, 1]) - epsilon < np.max(v2_np[:, 1])):
        
        overlaps = get_overlaps(v1, v2)
        if overlaps:
            for overlap in overlaps:
                add_location(locations, include, c1, overlap[0], c2, overlap[1], True)
        else:
            straight1 = Curve.is_straight(v1)
            straight2 = Curve.is_straight(v2)
            straight = straight1 and straight2
            flip = straight1 and not straight2
            before = len(locations)

            if straight:
                add_line_intersection(v1, v2, c1, c2, locations, include)
            elif straight1 or straight2:
                add_curve_line_intersections(v2 if flip else v1, v1 if flip else v2, c2 if flip else c1, c1 if flip else c2, locations, include, flip)
            else:
                add_curve_intersections(v2 if flip else v1, v1 if flip else v2, c2 if flip else c1, c1 if flip else c2, locations, include, flip,
                                        0, 0, 0, 1, 0, 1)

            if not straight or len(locations) == before:
                for i in range(4):
                    t1 = i >> 1
                    t2 = i & 1
                    p1 = Point(v1[t1 * 6], v1[t1 * 6 + 1])
                    p2 = Point(v2[t2 * 6], v2[t2 * 6 + 1])
                    if p1.is_close(p2, epsilon):
                        add_location(locations, include, c1, t1, c2, t2)
    return locations


def get_loop_intersection(v1, c1, locations, include):
    info = Curve.classify(v1)
    if info['type'] == 'loop':
        roots = info['roots']
        add_location(locations, include, c1, roots[0], c1, roots[1])
    return locations


def get_intersections(curves1, curves2=None, include=None, matrix1=None, matrix2=None, return_first=False):
    self_intersection = curves2 is None
    if self_intersection:
        curves2 = curves1

    values2 = [c.get_values(matrix2) for c in curves2]
    arrays = []
    locations = []

    for i in range(len(curves1)):
        curve1 = curves1[i]
        values1 = values2[i] if self_intersection else curve1.get_values(matrix1)
        path1 = None # Placeholder for path logic
        
        if path1 is None:
            locations = []
            arrays.append(locations)
        
        if self_intersection:
            get_loop_intersection(values1, curve1, locations, include)

        for j in range(i + 1 if self_intersection else 0, len(curves2)):
            if return_first and locations:
                return locations
            get_curve_intersections(values1, values2[j], curve1, curves2[j], locations, include)

    locations = [loc for sublist in arrays for loc in sublist]
    return locations


def get_overlaps(v1, v2):
    def get_squared_line_length(v):
        x, y = v[6] - v[0], v[7] - v[1]
        return x * x + y * y

    time_epsilon = Numerical.CURVETIME_EPSILON
    geom_epsilon = Numerical.GEOMETRIC_EPSILON
    
    straight1 = Curve.is_straight(v1)
    straight2 = Curve.is_straight(v2)
    straight_both = straight1 and straight2
    
    flip = get_squared_line_length(v1) < get_squared_line_length(v2)
    l1, l2 = (v2, v1) if flip else (v1, v2)
    
    px, py = l1[0], l1[1]
    vx, vy = l1[6] - px, l1[7] - py

    if Line.get_distance(px, py, vx, vy, l2[0], l2[1], True) < geom_epsilon**2 and \
       Line.get_distance(px, py, vx, vy, l2[6], l2[7], True) < geom_epsilon**2:
        if not straight_both and \
           Line.get_distance(px, py, vx, vy, l1[2], l1[3], True) < geom_epsilon**2 and \
           Line.get_distance(px, py, vx, vy, l1[4], l1[5], True) < geom_epsilon**2 and \
           Line.get_distance(px, py, vx, vy, l2[2], l2[3], True) < geom_epsilon**2 and \
           Line.get_distance(px, py, vx, vy, l2[4], l2[5], True) < geom_epsilon**2:
            straight1 = straight2 = straight_both = True
    elif straight_both:
        return None

    if straight1 ^ straight2:
        return None
    
    v = [v1, v2]
    pairs = []
    
    for i in range(4):
        if len(pairs) >= 2: break
        
        i1 = i & 1
        i2 = i1 ^ 1
        t1 = i >> 1
        
        p = Point(v[i2][t1 * 6], v[i2][t1 * 6 + 1])
        t2 = Curve.get_time_of(v[i1], p)
        
        if t2 is not None:
            pair = [t1, t2] if i1 else [t2, t1]
            if not pairs or (abs(pair[0] - pairs[0][0]) > time_epsilon and abs(pair[1] - pairs[0][1]) > time_epsilon):
                pairs.append(pair)
    
    if len(pairs) != 2:
        return None
    elif not straight_both:
        # Additional checks for non-straight curves (simplified for translation)
        o1 = Curve.get_part(v1, pairs[0][0], pairs[1][0])
        o2 = Curve.get_part(v2, pairs[0][1], pairs[1][1])
        if abs(o2[2] - o1[2]) > geom_epsilon or abs(o2[3] - o1[3]) > geom_epsilon or \
           abs(o2[4] - o1[4]) > geom_epsilon or abs(o2[5] - o1[5]) > geom_epsilon:
            return None
    
    return pairs
