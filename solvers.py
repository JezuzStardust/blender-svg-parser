# Quartic, cubic and quadratic solvers implemented in Python.
# Copyright (C) 2023 JezuzStardust https://github.com/JezuzStardust.
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

import math
import cmath
from typing import Union

def solve_quadratic(c2: Union[float, complex], c1: Union[float, complex], c0: Union[float, complex]):
    """Solves c2 * x**2 + c1 * x + c0 = 0. ci can be complex numbers.
    Returns a list of the roots."""
    
    if c2 == 0.0:
        return [] if c1 == 0.0 else [c0 / c1]

    c1 /= c2
    c0 /= c2

    if isinstance(c1, complex) or isinstance(c0, complex):
        gamma1 = - c1 / 2 + cmath.sqrt(c1**2 / 4 - c0)
        gamma2 = - c1 / 2 - cmath.sqrt(c1**2 / 4 - c0)

        if abs(gamma1) > abs(gamma2):
            eta1 = gamma1
        else:
            eta1 = gamma2
        x0 = eta1
        x1 = c0 / eta1
    else: 
        delta = c1**2 - 4 * c0
        if delta < 0.0: 
            x0 = - c1 / 2 + 1/2 * math.sqrt(-delta) * complex(0,1)
            x1 = - c1 / 2 - 1/2 * math.sqrt(-delta) * complex(0,1)
        else: 
            if c1 >= 0.0:
                etaM = - c1 / 2 - math.sqrt(delta) / 2 
            else:
                etaM = - c1 / 2 + math.sqrt(delta) / 2 
            if etaM == 0.0:
                etam = 0.0
            else:
                etam = c0 / etaM

            x0 = etaM
            x1 = etam

    return [x0, x1]

def solve_cubic(c3i: float, c2i: float, c1i: float, c0i: float, eps = 1e-12): 
    """Solves the cubic equation c3i * x**3 + c2i * x**2 + c1i * x + c0i = 0.
    See: https://momentsingraphics.de/CubicRoots.html.
    Returns only the real solutions."""

    if abs(c3i) < eps:
        # Equation is really quadratic.
        return solve_quadratic(c0i, c1i, c2i)

    c2 = c2i / (3 * c3i) 
    c1 = c1i / (3 * c3i)
    c0 = c0i / c3i
    
    # (d0, d1, d2) are called Delta in the article. 
    d0 = -c2**2 + c1
    d1 = -c1 * c2 + c0
    d2 = c2 * c0 - c1**2
    # d is called the discriminant.
    d = 4 * d0 * d2 - d1**2
    # de is called Depressed.x. Depressed.y = d0
    de = -2 * c2 * d0 + d1
    
    if d < -eps:
        sq = math.sqrt(-0.25 * d)
        r = - 0.5 * de
        # x**(1/3) can return complex number if x < 0. 
        # We take abs(x)**1/3 and then put back the sign to avoid this.
        t1: float = math.copysign(abs(r + sq)**(1/3), r + sq) + math.copysign(abs(r - sq)**(1/3), r - sq)
        return [t1 - c2]
    elif abs(d) < eps: 
        t1 = math.copysign(math.sqrt(-d0), de)
        return [t1 - c2, - 2 * t1 - c2]
    else:
        th = math.atan2(math.sqrt(d), -de) / 3
        # r0, r1, r2 is called "Root"
        r0 = math.cos(th)
        ss3 = math.sin(th) * math.sqrt(3) 
        r1 = .5 * (-r0 + ss3)
        r2 = .5 * (-r0 - ss3)
        t = 2 * math.sqrt(-d0)
        return [t * r0 - c2, t * r1 - c2, t * r2 - c2]

def test_cubic(a, b, c):
    print("Input: ", sorted([a, b, c]))
    c2 = - (a + b + c)
    c1 = (a * b + a * c + b * c)
    c0 = (- a * b * c)
    print("Output: ", solve_cubic(1.0, c2, c1, c0))

def solve_quartic(c4: float, c3:float, c2: float, c1: float, c0: float): 
    """Solves the quartic equation c4 * x**4 + c3 * x**3 + c2 * x**2 + c1 * x + c0 = 0.
    ci are floats (and hence real numbers). 
    The return value is a list of the solutions, including complex solutions.
    Note, if x + i * y is one solution, one of the other roots is always the complex conjugate.
    A direct implementation of the algorithm in the paper:
    Alberto Giacomo Orellana and Cristiano De Michele. 2020. 
    Algorithm 1010: Boosting Efficiency in Solving Quartic Equations with No Compromise in Accuracy. 
    ACM Trans. Math. Softw. 46, 2, Article 20 (June 2020), 28 pages. https://doi.org/10.1145/3386241
    """
    if c4 == 0.0: 
        return solve_cubic(c3, c2, c1, c0)

    a = c3 / c4
    b = c2 / c4
    c = c1 / c4
    d = c0 / c4
    # TODO: Rewrite function so that it accepts five coefficients instead. 
    # It is easier if the function handles also the case where there is a coefficient
    # in front of x**4. 
    solutions: list[Union[float, complex]] = []
    rescale_quartic = False
    phi = None
    try: 
        phi = dominant_root(a, b, c, d, rescale = False)
    except OverflowError: 
        print("Overflow error. Will try to rescale the quartic equation.")
    
    K_Q = 7.16e76
    if phi is None: 
        rescale_quartic = True
        a = a / K_Q
        b = b / K_Q**2
        c = c / K_Q**3
        d = d / K_Q**4
        try: 
            phi = dominant_root(a, b, c, d, rescale = False)
        except OverflowError:
            print("Still overflowing. Will try to rescale also the cubic equation.")

    if phi is None:
        try:
            phi = dominant_root(a, b, c, d, rescale = True)
        except OverflowError:
            print("No solutions found.")
            return solutions
        
    l1, l2, l3, d2 = ldl_coefficients(a, b, c, d, phi)
    
    alpha1, beta1, alpha2, beta2 = calculate_alpha_beta(a, b, c, d, l1, l2, l3, d2, phi)
    
    x0, x1 = solve_quadratic(1.0, alpha1, beta1)
    x2, x3 = solve_quadratic(1.0, alpha2, beta2)
    
    if rescale_quartic: 
        x0 *= K_Q
        x1 *= K_Q
        x2 *= K_Q
        x3 *= K_Q

    solutions = [x0, x1, x2, x3]
    return solutions

def ldl_coefficients(a: float, b: float, c: float, d: float, phi: float):
    """Calculates the coefficients l1, l2, l3, d2, d3 that are used to factor
    the quartic equation into two quadratic equations."""
    l1 = a / 2
    l3 = (b + 3 * phi) / 6
    delta_2 = c - a * l3
    # Pairs of candidate d2 and l2
    pairs = []
    d21 = 2 * b / 3 - phi - l1**2
    if d21 != 0:
        l21 = delta_2 / (2 * d21)
        pairs.append((d21, l21))
    if delta_2 != 0: 
        l22 = 2 * (d - l3**2) / delta_2
        if l22 != 0:
            d22 = delta_2 / (2 * l22)
            pairs.append((d22, l22))
        l23 = l22
        # d23 = d21 = 2 * b / 3 - phi - l1**2 
        pairs.append((d21, l23))
    if len(pairs) > 0:
    # Find the best pairs.
        cur_best = None
        best = 0
        for i, pair in enumerate(pairs): 
            e = epsilon_l(b, c, l1, l3, d, pair[0], pair[1])
            if cur_best is None or e < cur_best:
                cur_best = e
                best = i
        d2, l2 = pairs[best]
    else: 
        d2 = 0.0
        l2 = 0.0

    return [l1, l2, l3, d2] 

def calculate_alpha_beta(a: float, b: float, c: float, d: float, l1: float, l2: float, l3: float, d2: float, phi: float):
    """Calculates alpha1,  beta1, alpha2, beta2 which are the coefficients of the two quadratic equations."""

    # We set them all to zero to stop pyright from complaining that they might be unbound. 
    # They will not be, but I just got annoyed.
    alpha1 = 0.0
    alpha2 = 0.0
    beta1 = 0.0
    beta2 = 0.0
    candidates = []

    # Three different cases d2 < 0 done, d2 > 0, d2 == 0
    # In case d2 non-zero but close to zero, we are not sure, so in that case
    # we calculate both the cases where d2 = 0, and d2 > 0 or d2 < 0 (depending on the sign.
    # We finally check which alpha1, beta1, alpha2, beta2 are closest to the exact results.
    EPSILON_M = 2.22045e-16 # Suitable for double precision (which is what Python uses).
    if d2 == 0 or abs(d2) <= EPSILON_M * max(abs(2 * b / 3), abs(phi), l1**2):
        alpha1, beta1, alpha2, beta2 = case3(l1, l3, d)
        candidates.append((alpha1, beta1, alpha2, beta2))

    if d2 < 0: 
        alpha1, beta1, alpha2, beta2 = case1(a, b, c, d, l1, l2, l3, d2)
        candidates.append((alpha1, beta1, alpha2, beta2))
    elif d2 > 0:
        alpha1, beta1, alpha2, beta2 = case2(l1, l2, l3, d2)
        candidates.append((alpha1, beta1, alpha2, beta2))
    
    if len(candidates) == 2:
        # TODO: Move this to epsilon_q2? Note that epsilon_q2 is also used by refine_alpha_beta() so we should 
        # create a separate function in that case.
        val1 = epsilon_q2(a, b, c, d, *candidates[0])
        val2 = epsilon_q2(a, b, c, d, *candidates[1])
        if val1 < val2:
            alpha1, beta1, alpha2, beta2 = candidates[0]
        else:
            alpha1, beta1, alpha2, beta2 = candidates[1]

    alpha1, beta1, alpha2, beta2 = refine_alpha_beta(a, b, c, d, alpha1, beta1, alpha2, beta2)

    return [alpha1, beta1, alpha2, beta2]

def case1(a, b, c, d, l1, l2, l3, d2):
    """Calculates alpha1, beta1, alpha2, beta2 in the case where d2 < 0."""
    alpha1 = l1 + math.sqrt(-d2)
    beta1 = l3 + math.sqrt(-d2) * l2
    alpha2 = l1 - math.sqrt(-d2)
    beta2 = l3 - math.sqrt(-d2) * l2
    if abs(beta2) < abs(beta1): 
        beta2 = d / beta1
    elif abs(beta2) > abs(beta1):
        beta1 = d / beta2 

    # Find best alpha1 and alpha2
    if abs(alpha1) <= abs(alpha2): # Trust alpha2 and find the best alpha1.
        cands = []
        if beta2 != 0.0:
            alpha11 = (c - beta1 * alpha2) / beta2
            cands.append(alpha11)
        if alpha2 != 0.0:
            alpha12 = (b - beta2 - beta1) / alpha2
            cands.append(alpha12)
        alpha13 = a - alpha2 
        cands.append(alpha13)
    
        cur_best = None
        best = 0
        for i, alph1 in enumerate(cands):
            e = epsilon_q(a, b, c, alph1, beta1, alpha2, beta2)
            if cur_best is None or e < cur_best:
                cur_best = e
                best = i
            alpha1 = cands[best]
    else: # Trust alpha1 and find the best alpha2.
        cands = []
        if beta1 != 0.0:
            alpha21 = (c - alpha1 * beta2) / beta1
            cands.append(alpha21)
        if alpha1 != 0.0:
            alpha22 = (b - beta2 - beta1) / alpha1 
            cands.append(alpha22)
        alpha23 = a - alpha1
        cands.append(alpha23)
       
        cur_best = None
        best = 0
        for i, alph2 in enumerate(cands):
            e = epsilon_q(a, b, c, alpha1, beta1, alph2, beta2)
            if cur_best is None or e < cur_best:
                cur_best = e
                best = i
        alpha2 = cands[best]

    return alpha1, beta1, alpha2, beta2

def case2(l1, l2, l3, d2):
    """Calculates alpha1, beta1, alpha2, and beta2 in the case where d2 > 0."""
    alpha1 = l1 + complex(0,1) * math.sqrt(d2)
    beta1 = l3 + complex(0,1) * math.sqrt(d2) * l2 
    alpha2 = l1 - complex(0,1) * math.sqrt(d2)
    beta2 = l3 - complex(0,1) * math.sqrt(d2) * l2
    return alpha1, beta1, alpha2, beta2 

def case3(l1, l3, d):
    """Calculates alpha1, beta1, alpha2, and beta2 in the case where d2 = 0 (or close to 0)."""
    d3 = d - l3**2
    alpha1 = l1
    beta1 = l3 + math.sqrt(-d3)
    alpha2 = l1
    beta2 = l3 - math.sqrt(-d3)
    if abs(beta1) > abs(beta2):
        beta2 = d / beta1
    else: 
        beta1 = d / beta2
    return alpha1, beta1, alpha2, beta2

def epsilon_l(b: float, c: float, l1: float, l3: float, d: float, d2: float, l2: float):
    # TODO: Move the entire testing to this function.
    # Must accept all pairs d2, l2 at the same time. 
    if b == 0:
        eps0 = abs(d2 + l1**2 + 2 * l3)
    else:
        eps0 = abs((d2 + l1**2 + 2 * l3 - b) / b)
    if c == 0:
        eps1 = abs(2 * d2 * l2 + 2 * l1 * l3)
    else:
        eps1 = abs((2 * d2 * l2 + 2 * l1 * l3 - c) / c)
    if d == 0:
        eps2 = abs(d2 * l2**2 + l3**2)
    else:
        eps2 = abs((d2 * l2**2 + l3**2 - d) / d)

    return eps0 + eps1 + eps2

def epsilon_q(a: float, b: float, c: float, 
              alpha1: Union[float, complex], beta1: Union[float, complex], 
              alpha2: Union[float, complex], beta2: Union[float, complex]):
    if a == 0:
        epsa = abs(alpha1 + alpha2)
    else:
        epsa = abs((alpha1 + alpha2 - a) / a)
    if b == 0:
        epsb = abs(beta1 + alpha1 * alpha2 + beta2)
    else:
        epsb = abs((beta1 + alpha2 * alpha1 + beta2 - b) / b)
    if c == 0:
        epsc = abs(beta1 * alpha2 + alpha1 * beta2)
    else:
        epsc = abs((beta1 * alpha2 + alpha1 * beta2 - c) / c)

    return epsa + epsb + epsc 

def epsilon_q2(a: float, b: float, c: float, d: float, 
               alpha1: Union[float, complex], beta1: Union[float, complex], 
               alpha2: Union[float, complex], beta2: Union[float, complex]):
    if a == 0:
        epsa = abs(alpha1 + alpha2)
    else:
        epsa = abs((alpha1 + alpha2 - a) / a)
    if b == 0:
        epsb = abs(beta1 + alpha1 * alpha2 + beta2)
    else:
        epsb = abs((beta1 * alpha2 + alpha1 * beta2 - b) / b)
    if c == 0:
        epsc = abs(beta1 * alpha2 + alpha1 * beta2)
    else:
        epsc = abs((beta1 * alpha2 + alpha1 * beta2 - c) / c)
    if d == 0:
        epsd = abs(beta1 * beta2)
    else:
        epsd = abs((beta1 * beta2 - d) / d)
    return epsa + epsb + epsc + epsd

def dominant_root(a: float, b: float, c: float, d: float, rescale = False):
    """Returns the dominant root of the depressed cubic equation phi**3 + g phi + h = 0."""

    # TODO: In case this fails to produce a solution then: 
    # First try to rescale the quartic using kq = 7.16e76
    # Else, if that fails to produce a root for the cubic, then 
    # try rescaling the cubic with kc = 3.49e102
    # All this code can actually go into the cubic solver, not here

    # Determine s so that b'(s) = 0, or db'/ds = 0. 

    if 9 * a**2 - 24 * b >= 0.0:
        s = - 2 * b / (3 * a + math.copysign(1,a) * math.sqrt(9 * a**2 - 24 * b))
    else:
        s = - a / 4
    
    ap = a + 4 * s 
    bp = b + 3 * s * (a + 2 * s)
    cp = c + s * (2 * b + s * (3 * a + 4 * s))
    dp = d + s * (c + s * (b + s * (a + s)))

    K_C = 3.49e102
    if rescale: 
        ap /= K_C
        bp /= K_C
        cp /= K_C
        dp /= K_C
        gp = ap * cp - 4 * dp / K_C - bp**2 / 3
        hp = ( ap * cp + 8 * dp / K_C - 2 * bp**2 / 9) * bp / 3 - cp * cp / K_C - ap**2 * dp
    else:
        gp = ap * cp - 4 * dp - bp**2 / 3
        hp = (ap * cp + 8 * dp - 2 * bp**2 / 9) * bp / 3 - cp**2 - ap**2 * dp

    Q = - gp / 3
    R = hp / 2

    # Handle cases where Q and R are large. 
    if abs(Q) >= 1e102 or abs(R) >= 1e154: 
        if R == 0:
            if gp > 0:
                phi = 0
            else: 
                phi = math.sqrt(-gp)
        else: 
            if abs(Q) < abs(R):
                K = 1 - Q * (Q / R)**2
            else:
                K = math.copysign(1,Q) * ((R/Q)**2 / Q - 1)
            if K < 0:
                theta = math.acos(R / Q * 1 / math.sqrt(Q))
                if theta < math.pi / 2:
                    phi = - 2 * math.sqrt(Q) * math.cos(theta / 3)
                else:
                    phi = - 2 * math.sqrt(Q) * math.cos((theta + 2 * math.pi) / 3)
            else: 
                if abs(Q) < abs(R):
                    A = - math.copysign(1,R) * (abs(R) * (1 + math.sqrt(K)))**(1/3)
                else:
                    A = - math.copysign(1,R) * (abs(R) + math.sqrt(abs(Q)) * abs(Q) * math.sqrt(K))**(1/3)
                if A == 0: 
                    B = 0
                else:
                    B = Q / A
                phi = A + B
    # Cases where Q and R are not too large.  
    else:
        if R**2 < Q**3:
            theta = math.acos( R / math.sqrt(Q**3))
            if theta < math.pi / 2:
                phi = - 2 * math.sqrt(Q) * math.cos(theta / 3)
            else:
                phi = - 2 * math.sqrt(Q) * math.cos((theta + 2 * math.pi) / 3)
        else:
            A = - math.copysign(1, R) * (abs(R) + math.sqrt(R**2 - Q**3))**(1/3)
            if A == 0.0:
                B = 0
            else:
                B = Q / A
            phi = A  + B
    phi = dominant_root_refine(phi, gp, hp)

    if rescale:
        phi *= K_C

    return phi

def dominant_root_refine(phi: float, gp: float, hp: float):
    """Refine the dominant root using Newton-Raphson."""
    # TODO: Should check for nan and inf. 
    EPSILON_M = 2.22045e-16
    x = phi
    f = (x**2 + gp) * x + hp
    if abs(f) < EPSILON_M * max(abs(x**3), abs(gp * x), abs(hp)):
        return x
    n = 0
    while(n < 1000):
        df = 3 * x**2 + gp 
        if df == 0.0:
            return x 

        x0 = x
        f0 = f
        x = x - f / df
        f = (x**2 + gp) * x + hp
        if abs(f) == 0.0:
            return x
        if abs(f) > abs(f0):
            return x0
        n += 1
    return phi

def refine_alpha_beta(a: float, b: float, c: float, d: float, 
                      alpha1: Union[float, complex], beta1: Union[float, complex], 
                      alpha2: Union[float, complex], beta2: Union[float, complex]):
    """Refine the coefficients alpha1, beta1, alpha2, and beta2 using the Newton-Raphson method."""

    # Try 8 times (usually converges fast. 
    for _ in range(0,8):
        z = [alpha1, beta1, alpha2, beta2]
        epsilon_t0 = epsilon_q2(a, b, c, d, *z)
        if epsilon_t0 == 0.0: 
            return z
        detJ = beta1**2 - beta1 * (alpha2 * (alpha1 - alpha2) + 2 * beta2) + beta2 * (alpha1 * (alpha1 - alpha2) + beta2)
        if detJ == 0.0:
            return z
        z0 = z.copy()
        C1 = alpha1 - alpha2
        C2 = beta2 - beta1
        C3 = beta1 * alpha2 - alpha1 * beta2 
        F1 = beta1 * beta2 - d
        F2 = beta1 * alpha2 + alpha1 * beta2 - c
        F3 = beta1 + alpha1 * alpha2 + beta2 - b
        F4 = alpha1 + alpha2 - a
        JF1 = C1 * F1 + C2 * F2 + C3 * F3 - (beta1 * C2 + alpha1 * C3) * F4
        JF2 = (alpha1 * C1 + C2) * F1 - beta1 * C1 * F2 - beta1 * C2 * F3 - beta1 * C3 * F4
        JF3 = - C1 * F1 - C2 * F2 - C3 * F3 + (alpha2 * C3 + beta2 * C2) * F4
        JF4 = (- alpha2 * C1 - C2) * F1 + beta2 * C1 * F2 + beta2 * C2 * F3 + beta2 * C3 * F4
        z[0] -= JF1 / detJ
        z[1] -= JF2 / detJ
        z[2] -= JF3 / detJ
        z[3] -= JF4 / detJ
        if z == z0:
            return z
        epsilon_t1 = epsilon_q2(a, b, c, d, *z)
        if epsilon_t1 == 0.0:
            return z
        if epsilon_t1 > epsilon_t0:
            return z0
    return [alpha1, beta1, alpha2, beta2] 

def vieta(x1, x2, x3, x4):
    c3 = -(x1 + x2 + x3 + x4)
    c2 = x1 * (x2 + x3) + x2 * (x3 + x4) + x4 * (x1 + x3)
    c1 = -x1 * x2 * (x3 + x4) - x3 * x4 * (x1 + x2)
    c0 = x1 * x2 * x3 * x4
    roots = solve_quartic(1.0, c3.real, c2.real, c1.real, c0.real)
    return roots

def test_quartic(i):
    """All test cases from the article."""
    roots = [[1e9, 1e6, 1e3, 2],
             [2.003, 2.002, 2.001, 2],
             [1e53, 1e50, 1e49, 1e47],
             [1e14, 2.0, 1.0, -1.0],
             [-2e7, 1e7, 1.0, -1.0],
             [1e7, -1e6, complex(1.0, 1.0), complex(1.0, -1.0)],
             [-7.0, -4.0, complex(-1e6, 1e5), complex(-1e6, -1e5)],
             [1e8, 11.0, complex(1e3, 1), complex(1e3, -1)],
             [complex(1e7, 1e6), complex(1e7, -1e6), complex(1, 2), complex(1, -2)],
             [complex(1e4, 3), complex(1e4, -3), complex(-7, 1e3), complex(-7, -1e3)],
             [complex(1.001, 4.998), complex(1.001, -4.998), complex(1.0, 5.001), complex(1.0, -5.001)],
             [complex(1e3, 3), complex(1e3, -3), complex(1e3, 1), complex(1e3, -1)],
             [complex(2, 1e4), complex(2, -1e4), complex(1, 1e3), complex(1, -1e3)],
             [1000, 1000, 1000, 1000],
             [1000, 1000, 1000, 1e-15],
             [complex(1.0, 0.1), complex(1.0, -0.1), complex(1e16, 1e7), complex(1e16, -1e7)],
             [10000, 10001, 10010, 10100],
             [complex(40000, 300), complex(40000, -300), complex(30000, 7000), complex(30000, -7000)],
             [1e44, 1e30, 1e30, 1],
             [1e14, 1e7, 1e7, 1],
             [1e15, 1e7, 1e7, 1],
             [1e154, 1e152, 10, 1]
             ]
    print(sorted(vieta(*roots[i-1]), key=lambda x: x.real))
    print(sorted(roots[i-1], key=lambda x: x.real))




