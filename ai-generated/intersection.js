new function() { // Scope for bezier intersection using fat-line clipping

    function addLocation(locations, include, c1, t1, c2, t2, overlap) {
        // Determine if locations at the beginning / end of the curves should be
        // excluded, in case the two curves are neighbors, but do not exclude
        // connecting points between two curves if they were part of overlap
        // checks, as they could be self-overlapping.
        // NOTE: We don't pass p1 and p2, because v1 and v2 may be transformed
        // by their path.matrix, while c1 and c2 are untransformed. Passing null
        // for point in CurveLocation() will do the right thing.
        var excludeStart = !overlap && c1.getPrevious() === c2,
            excludeEnd = !overlap && c1 !== c2 && c1.getNext() === c2,
            tMin = /*#=*/Numerical.CURVETIME_EPSILON,
            tMax = 1 - tMin;
        // Check t1 and t2 against correct bounds, based on excludeStart/End:
        // - excludeStart means the start of c1 connects to the end of c2
        // - excludeEnd means the end of c1 connects to the start of c2
        // - If either c1 or c2 are at the end of the path, exclude their end,
        //   which connects back to the beginning, but only if it's not part of
        //   a found overlap. The normal intersection will already be found at
        //   the beginning, and would be added twice otherwise.
        if (t1 !== null && t1 >= (excludeStart ? tMin : 0) &&
            t1 <= (excludeEnd ? tMax : 1)) {
            if (t2 !== null && t2 >= (excludeEnd ? tMin : 0) &&
                t2 <= (excludeStart ? tMax : 1)) {
                var loc1 = new CurveLocation(c1, t1, null, overlap),
                    loc2 = new CurveLocation(c2, t2, null, overlap);
                // Link the two locations to each other.
                loc1._intersection = loc2;
                loc2._intersection = loc1;
                if (!include || include(loc1)) {
                    CurveLocation.insert(locations, loc1, true);
                }
            }
        }
    }

    function addCurveIntersections(v1, v2, c1, c2, locations, include, flip,
            recursion, calls, tMin, tMax, uMin, uMax) {
        // Avoid deeper recursion, by counting the total amount of recursions,
        // as well as the total amount of calls, to avoid massive call-trees as
        // suggested by @iconexperience in #904#issuecomment-225283430.
        // See also: #565 #899 #1074
        if (++calls >= 4096 || ++recursion >= 40)
            return calls;
        // Use an epsilon smaller than CURVETIME_EPSILON to compare curve-time
        // parameters in fat-line clipping code.
        var fatLineEpsilon = 1e-9,
            // Let P be the first curve and Q be the second
            q0x = v2[0], q0y = v2[1], q3x = v2[6], q3y = v2[7],
            getSignedDistance = Line.getSignedDistance,
            // Calculate the fat-line L for Q is the baseline l and two
            // offsets which completely encloses the curve P.
            d1 = getSignedDistance(q0x, q0y, q3x, q3y, v2[2], v2[3]),
            d2 = getSignedDistance(q0x, q0y, q3x, q3y, v2[4], v2[5]),
            factor = d1 * d2 > 0 ? 3 / 4 : 4 / 9,
            dMin = factor * Math.min(0, d1, d2),
            dMax = factor * Math.max(0, d1, d2),
            // Calculate non-parametric bezier curve D(ti, di(t)):
            // - di(t) is the distance of P from baseline l of the fat-line
            // - ti is equally spaced in [0, 1]
            dp0 = getSignedDistance(q0x, q0y, q3x, q3y, v1[0], v1[1]),
            dp1 = getSignedDistance(q0x, q0y, q3x, q3y, v1[2], v1[3]),
            dp2 = getSignedDistance(q0x, q0y, q3x, q3y, v1[4], v1[5]),
            dp3 = getSignedDistance(q0x, q0y, q3x, q3y, v1[6], v1[7]),
            // Get the top and bottom parts of the convex-hull
            hull = getConvexHull(dp0, dp1, dp2, dp3),
            top = hull[0],
            bottom = hull[1],
            tMinClip,
            tMaxClip;
        // Stop iteration if all points and control points are collinear.
        if (d1 === 0 && d2 === 0
                && dp0 === 0 && dp1 === 0 && dp2 === 0 && dp3 === 0
            // Clip convex-hull with dMin and dMax, taking into account that
            // there will be no intersections if one of the results is null.
            || (tMinClip = clipConvexHull(top, bottom, dMin, dMax)) == null
            || (tMaxClip = clipConvexHull(top.reverse(), bottom.reverse(),
                dMin, dMax)) == null)
            return calls;
        // tMin and tMax are within the range (0, 1). Project it back to the
        // original parameter range for v2.
        var tMinNew = tMin + (tMax - tMin) * tMinClip,
            tMaxNew = tMin + (tMax - tMin) * tMaxClip;
        if (Math.max(uMax - uMin, tMaxNew - tMinNew) < fatLineEpsilon) {
            // We have isolated the intersection with sufficient precision
            var t = (tMinNew + tMaxNew) / 2,
                u = (uMin + uMax) / 2;
            addLocation(locations, include,
                    flip ? c2 : c1, flip ? u : t,
                    flip ? c1 : c2, flip ? t : u);
        } else {
            // Apply the result of the clipping to curve 1:
            v1 = Curve.getPart(v1, tMinClip, tMaxClip);
            if (tMaxClip - tMinClip > 0.8) {
                // Subdivide the curve which has converged the least.
                if (tMaxNew - tMinNew > uMax - uMin) {
                    var parts = Curve.subdivide(v1, 0.5),
                        t = (tMinNew + tMaxNew) / 2;
                    calls = addCurveIntersections(
                            v2, parts[0], c2, c1, locations, include, !flip,
                            recursion, calls, uMin, uMax, tMinNew, t);
                    calls = addCurveIntersections(
                            v2, parts[1], c2, c1, locations, include, !flip,
                            recursion, calls, uMin, uMax, t, tMaxNew);
                } else {
                    var parts = Curve.subdivide(v2, 0.5),
                        u = (uMin + uMax) / 2;
                    calls = addCurveIntersections(
                            parts[0], v1, c2, c1, locations, include, !flip,
                            recursion, calls, uMin, u, tMinNew, tMaxNew);
                    calls = addCurveIntersections(
                            parts[1], v1, c2, c1, locations, include, !flip,
                            recursion, calls, u, uMax, tMinNew, tMaxNew);
                }
            } else { // Iterate
                if (uMax - uMin >= fatLineEpsilon) {
                    calls = addCurveIntersections(
                            v2, v1, c2, c1, locations, include, !flip,
                            recursion, calls, uMin, uMax, tMinNew, tMaxNew);
                } else {
                    // The interval on the other curve is already tight enough,
                    // therefore we keep iterating on the same curve.
                    calls = addCurveIntersections(
                            v1, v2, c1, c2, locations, include, flip,
                            recursion, calls, tMinNew, tMaxNew, uMin, uMax);
                }
            }
        }
        return calls;
    }

    /**
     * Calculate the convex hull for the non-parametric bezier curve D(ti, di(t))
     * The ti is equally spaced across [0..1] — [0, 1/3, 2/3, 1] for
     * di(t), [dq0, dq1, dq2, dq3] respectively. In other words our CVs for the
     * curve are already sorted in the X axis in the increasing order.
     * Calculating convex-hull is much easier than a set of arbitrary points.
     *
     * The convex-hull is returned as two parts [TOP, BOTTOM]:
     * (both are in a coordinate space where y increases upwards with origin at
     * bottom-left)
     * TOP: The part that lies above the 'median' (line connecting end points of
     * the curve)
     * BOTTOM: The part that lies below the median.
     */
    function getConvexHull(dq0, dq1, dq2, dq3) {
        var p0 = [ 0, dq0 ],
            p1 = [ 1 / 3, dq1 ],
            p2 = [ 2 / 3, dq2 ],
            p3 = [ 1, dq3 ],
            // Find vertical signed distance of p1 and p2 from line [p0, p3]
            dist1 = dq1 - (2 * dq0 + dq3) / 3,
            dist2 = dq2 - (dq0 + 2 * dq3) / 3,
            hull;
        // Check if p1 and p2 are on the opposite side of the line [p0, p3]
        if (dist1 * dist2 < 0) {
            // p1 and p2 lie on different sides of [p0, p3]. The hull is a
            // quadrilateral and line [p0, p3] is NOT part of the hull so we are
            // pretty much done here. The top part includes p1, we will reverse
            // it later if that is not the case.
            hull = [[p0, p1, p3], [p0, p2, p3]];
        } else {
            // p1 and p2 lie on the same sides of [p0, p3]. The hull can be a
            // triangle or a quadrilateral and line [p0, p3] is part of the
            // hull. Check if the hull is a triangle or a quadrilateral. We have
            // a triangle if the vertical distance of one of the middle points
            // (p1, p2) is equal or less than half the vertical distance of the
            // other middle point.
            var distRatio = dist1 / dist2;
            hull = [
                // p2 is inside, the hull is a triangle.
                distRatio >= 2 ? [p0, p1, p3]
                // p1 is inside, the hull is a triangle.
                : distRatio <= 0.5 ? [p0, p2, p3]
                // Hull is a quadrilateral, we need all lines in correct order.
                : [p0, p1, p2, p3],
                // Line [p0, p3] is part of the hull.
                [p0, p3]
            ];
        }
        // Flip hull if dist1 is negative or if it is zero and dist2 is negative
        return (dist1 || dist2) < 0 ? hull.reverse() : hull;
    }

    /**
     * Clips the convex-hull and returns [tMin, tMax] for the curve contained.
     */
    function clipConvexHull(hullTop, hullBottom, dMin, dMax) {
        if (hullTop[0][1] < dMin) {
            // Left of hull is below dMin, walk through the hull until it
            // enters the region between dMin and dMax
            return clipConvexHullPart(hullTop, true, dMin);
        } else if (hullBottom[0][1] > dMax) {
            // Left of hull is above dMax, walk through the hull until it
            // enters the region between dMin and dMax
            return clipConvexHullPart(hullBottom, false, dMax);
        } else {
            // Left of hull is between dMin and dMax, no clipping possible
            return hullTop[0][0];
        }
    }

    function clipConvexHullPart(part, top, threshold) {
        var px = part[0][0],
            py = part[0][1];
        for (var i = 1, l = part.length; i < l; i++) {
            var qx = part[i][0],
                qy = part[i][1];
            if (top ? qy >= threshold : qy <= threshold) {
                return qy === threshold ? qx
                        : px + (threshold - py) * (qx - px) / (qy - py);
            }
            px = qx;
            py = qy;
        }
        // All points of hull are above / below the threshold
        return null;
    }

    /**
     * Intersections between curve and line becomes rather simple here mostly
     * because of Numerical class. We can rotate the curve and line so that the
     * line is on the X axis, and solve the implicit equations for the X axis
     * and the curve.
     */
    function getCurveLineIntersections(v, px, py, vx, vy) {
        var isZero = Numerical.isZero;
        if (isZero(vx) && isZero(vy)) {
            // Handle special case of a line with no direction as a point,
            // and check if it is on the curve.
            var t = Curve.getTimeOf(v, new Point(px, py));
            return t === null ? [] : [t];
        }
        // Calculate angle to the x-axis (1, 0).
        var angle = Math.atan2(-vy, vx),
            sin = Math.sin(angle),
            cos = Math.cos(angle),
            // (rlx1, rly1) = (0, 0)
            // Calculate the curve values of the rotated curve.
            rv = [],
            roots = [];
        for (var i = 0; i < 8; i += 2) {
            var x = v[i] - px,
                y = v[i + 1] - py;
            rv.push(
                x * cos - y * sin,
                x * sin + y * cos);
        }
        // Solve it for y = 0. We need to include t = 0, 1 and let addLocation()
        // do the filtering, to catch important edge cases.
        Curve.solveCubic(rv, 1, 0, roots, 0, 1);
        return roots;
    }

    function addCurveLineIntersections(v1, v2, c1, c2, locations, include,
            flip) {
        // addCurveLineIntersections() is called so that v1 is always the curve
        // and v2 the line. flip indicates whether the curves need to be flipped
        // in the call to addLocation().
        var x1 = v2[0], y1 = v2[1],
            x2 = v2[6], y2 = v2[7],
            roots = getCurveLineIntersections(v1, x1, y1, x2 - x1, y2 - y1);
        // NOTE: count could be -1 for infinite solutions, but that should only
        // happen with lines, in which case we should not be here.
        for (var i = 0, l = roots.length; i < l; i++) {
            // For each found solution on the rotated curve, get the point on
            // the real curve and with that the location on the line.
            var t1 = roots[i],
                p1 = Curve.getPoint(v1, t1),
                t2 = Curve.getTimeOf(v2, p1);
            if (t2 !== null) {
                // Only use the time values if there was no recursion, and let
                // addLocation() figure out the actual time values otherwise.
                addLocation(locations, include,
                        flip ? c2 : c1, flip ? t2 : t1,
                        flip ? c1 : c2, flip ? t1 : t2);
            }
        }
    }

    function addLineIntersection(v1, v2, c1, c2, locations, include) {
        var pt = Line.intersect(
                v1[0], v1[1], v1[6], v1[7],
                v2[0], v2[1], v2[6], v2[7]);
        if (pt) {
            addLocation(locations, include,
                    c1, Curve.getTimeOf(v1, pt),
                    c2, Curve.getTimeOf(v2, pt));
        }
    }

    function getCurveIntersections(v1, v2, c1, c2, locations, include) {
        // Avoid checking curves if completely out of control bounds.
        var epsilon = /*#=*/Numerical.EPSILON,
            min = Math.min,
            max = Math.max;

        if (max(v1[0], v1[2], v1[4], v1[6]) + epsilon >
            min(v2[0], v2[2], v2[4], v2[6]) &&
            min(v1[0], v1[2], v1[4], v1[6]) - epsilon <
            max(v2[0], v2[2], v2[4], v2[6]) &&
            max(v1[1], v1[3], v1[5], v1[7]) + epsilon >
            min(v2[1], v2[3], v2[5], v2[7]) &&
            min(v1[1], v1[3], v1[5], v1[7]) - epsilon <
            max(v2[1], v2[3], v2[5], v2[7])) {
            // Now detect and handle overlaps:
            var overlaps = getOverlaps(v1, v2);
            if (overlaps) {
                for (var i = 0; i < 2; i++) {
                    var overlap = overlaps[i];
                    addLocation(locations, include,
                            c1, overlap[0],
                            c2, overlap[1], true);
                }
            } else {
                var straight1 = Curve.isStraight(v1),
                    straight2 = Curve.isStraight(v2),
                    straight = straight1 && straight2,
                    flip = straight1 && !straight2,
                    before = locations.length;
                // Determine the correct intersection method based on whether
                // one or curves are straight lines:
                (straight
                    ? addLineIntersection
                    : straight1 || straight2
                        ? addCurveLineIntersections
                        : addCurveIntersections)(
                            flip ? v2 : v1, flip ? v1 : v2,
                            flip ? c2 : c1, flip ? c1 : c2,
                            locations, include, flip,
                            // Define the defaults for these parameters of
                            // addCurveIntersections():
                            // recursion, calls, tMin, tMax, uMin, uMax
                            0, 0, 0, 1, 0, 1);
                // Handle the special case where the first curve's start- / end-
                // point overlaps with the second curve's start- / end-point,
                // but only if haven't found a line-line intersection already:
                // #805#issuecomment-148503018
                if (!straight || locations.length === before) {
                    for (var i = 0; i < 4; i++) {
                        var t1 = i >> 1, // 0, 0, 1, 1
                            t2 = i & 1,  // 0, 1, 0, 1
                            i1 = t1 * 6,
                            i2 = t2 * 6,
                            p1 = new Point(v1[i1], v1[i1 + 1]),
                            p2 = new Point(v2[i2], v2[i2 + 1]);
                        if (p1.isClose(p2, epsilon)) {
                            addLocation(locations, include,
                                    c1, t1,
                                    c2, t2);
                        }
                    }
                }
            }
        }
        return locations;
    }

    function getLoopIntersection(v1, c1, locations, include) {
        var info = Curve.classify(v1);
        if (info.type === 'loop') {
            var roots = info.roots;
            addLocation(locations, include,
                    c1, roots[0],
                    c1, roots[1]);
        }
      return locations;
    }

    function getIntersections(curves1, curves2, include, matrix1, matrix2,
            _returnFirst) {
        var self = !curves2;
        if (self)
            curves2 = curves1;
        var length1 = curves1.length,
            length2 = curves2.length,
            values2 = [],
            arrays = [],
            locations,
            current;
        // Cache values for curves2 as we re-iterate them for each in curves1.
        for (var i = 0; i < length2; i++)
            values2[i] = curves2[i].getValues(matrix2);
        for (var i = 0; i < length1; i++) {
            var curve1 = curves1[i],
                values1 = self ? values2[i] : curve1.getValues(matrix1),
                path1 = curve1.getPath();
            // NOTE: Due to the nature of getCurveIntersections(), we use
            // separate location arrays per path1, to make sure the circularity
            // checks are not getting confused by locations on separate paths.
            // The separate arrays are then flattened in the end.
            if (path1 !== current) {
                current = path1;
                locations = [];
                arrays.push(locations);
            }
            if (self) {
                // First check for self-intersections within the same curve.
                getLoopIntersection(values1, curve1, locations, include);
            }
            // Check for intersections with other curves.
            // For self-intersection, we can start at i + 1 instead of 0.
            for (var j = self ? i + 1 : 0; j < length2; j++) {
                // There might be already one location from the above
                // self-intersection check:
                if (_returnFirst && locations.length)
                    return locations;
                getCurveIntersections(values1, values2[j], curve1, curves2[j],
                        locations, include);
            }
        }
        // Flatten the list of location arrays to one array and return it.
        locations = [];
        for (var i = 0, l = arrays.length; i < l; i++) {
            locations.push.apply(locations, arrays[i]);
        }
        return locations;
    }

    /**
     * Code to detect overlaps of intersecting based on work by
     * @iconexperience in #648
     */
    function getOverlaps(v1, v2) {
        // Linear curves can only overlap if they are collinear. Instead of
        // using the #isCollinear() check, we pick the longer of the two curves
        // treated as lines, and see how far the starting and end points of the
        // other line are from this line (assumed as an infinite line). But even
        // if the curves are not straight, they might just have tiny handles
        // within geometric epsilon distance, so we have to check for that too.

        function getSquaredLineLength(v) {
            var x = v[6] - v[0],
                y = v[7] - v[1];
            return x * x + y * y;
        }

        var abs = Math.abs,
            getDistance = Line.getDistance,
            timeEpsilon = /*#=*/Numerical.CURVETIME_EPSILON,
            geomEpsilon = /*#=*/Numerical.GEOMETRIC_EPSILON,
            straight1 = Curve.isStraight(v1),
            straight2 = Curve.isStraight(v2),
            straightBoth = straight1 && straight2,
            flip = getSquaredLineLength(v1) < getSquaredLineLength(v2),
            l1 = flip ? v2 : v1,
            l2 = flip ? v1 : v2,
            // Get l1 start and end point values for faster referencing.
            px = l1[0], py = l1[1],
            vx = l1[6] - px, vy = l1[7] - py;
        // See if the starting and end point of curve two are very close to the
        // picked line. Note that the curve for the picked line might not
        // actually be a line, so we have to perform more checks after.
        if (getDistance(px, py, vx, vy, l2[0], l2[1], true) < geomEpsilon &&
            getDistance(px, py, vx, vy, l2[6], l2[7], true) < geomEpsilon) {
            // If not both curves are straight, check against both of their
            // handles, and treat them as straight if they are very close.
            if (!straightBoth &&
                getDistance(px, py, vx, vy, l1[2], l1[3], true) < geomEpsilon &&
                getDistance(px, py, vx, vy, l1[4], l1[5], true) < geomEpsilon &&
                getDistance(px, py, vx, vy, l2[2], l2[3], true) < geomEpsilon &&
                getDistance(px, py, vx, vy, l2[4], l2[5], true) < geomEpsilon) {
                straight1 = straight2 = straightBoth = true;
            }
        } else if (straightBoth) {
            // If both curves are straight and not very close to each other,
            // there can't be a solution.
            return null;
        }
        if (straight1 ^ straight2) {
            // If one curve is straight, the other curve must be straight too,
            // otherwise they cannot overlap.
            return null;
        }

        var v = [v1, v2],
            pairs = [];
        // Iterate through all end points:
        // First p1 of curve 1 & 2, then p2 of curve 1 & 2.
        for (var i = 0; i < 4 && pairs.length < 2; i++) {
            var i1 = i & 1,  // 0, 1, 0, 1
                i2 = i1 ^ 1, // 1, 0, 1, 0
                t1 = i >> 1, // 0, 0, 1, 1
                t2 = Curve.getTimeOf(v[i1], new Point(
                    v[i2][t1 ? 6 : 0],
                    v[i2][t1 ? 7 : 1]));
            if (t2 != null) {  // If point is on curve
                var pair = i1 ? [t1, t2] : [t2, t1];
                // Filter out tiny overlaps.
                if (!pairs.length ||
                    abs(pair[0] - pairs[0][0]) > timeEpsilon &&
                    abs(pair[1] - pairs[0][1]) > timeEpsilon) {
                    pairs.push(pair);
                }
            }
            // We checked 3 points but found no match, curves can't overlap.
            if (i > 2 && !pairs.length)
                break;
        }
        if (pairs.length !== 2) {
            pairs = null;
        } else if (!straightBoth) {
            // Straight pairs don't need further checks. If we found 2 pairs,
            // the end points on v1 & v2 should be the same.
            var o1 = Curve.getPart(v1, pairs[0][0], pairs[1][0]),
                o2 = Curve.getPart(v2, pairs[0][1], pairs[1][1]);
            // Check if handles of the overlapping curves are the same too.
            if (abs(o2[2] - o1[2]) > geomEpsilon ||
                abs(o2[3] - o1[3]) > geomEpsilon ||
                abs(o2[4] - o1[4]) > geomEpsilon ||
                abs(o2[5] - o1[5]) > geomEpsilon)
                pairs = null;
        }
        return pairs;
    }

    return /** @lends Curve# */{
        /**
         * Returns all intersections between two {@link Curve} objects as an
         * array of {@link CurveLocation} objects.
         *
         * @param {Curve} curve the other curve to find the intersections with
         *     (if the curve itself or `null` is passed, the self intersection
         *     of the curve is returned, if it exists)
         * @return {CurveLocation[]} the locations of all intersections between
         *     the curves
         */
        getIntersections: function(curve) {
            var v1 = this.getValues(),
                v2 = curve && curve !== this && curve.getValues();
            return v2 ? getCurveIntersections(v1, v2, this, curve, [])
                      : getLoopIntersection(v1, this, []);
        },

        statics: /** @lends Curve */{
            getOverlaps: getOverlaps,
            // Exposed for use in boolean offsetting
            getIntersections: getIntersections,
            getCurveLineIntersections: getCurveLineIntersections
        }
    };
});
