import itertools 
import math

def solve_quadratic(a, b, c):
    """Solve quadratic equation a * x**2 + b * x + c = 0.
    Discards complex solutions. Avoids cancellation errors."""
    if a == 0:
        if b == 0:
            return tuple()
        else: 
            return tuple(- c / b)
    else: 
        d = b**2 - 4 * a * c
        if d > 0: 
            q = -.5 * (b + math.copysign(math.sqrt(d) , b))
            x1 = q / a
            x2 = c / q
            return (x1 , x2) if x1 > x2 else (x2, x1)
        elif d == 0: 
            return tuple(- .5 * b / a)
        else: # Complex solutions only.
            return tuple()

def solve_cubic(in_c0: float, in_c1: float, in_c2: float, in_c3: float): 
    """Solve the cubic equation a*x**3 + b*x**2 + c*x + d = 0.
    See: https://momentsingraphics.de/CubicRoots.html"""
    # If in_c3 = 0, we really have a quadratic equation.
    if in_c3 == 0: 
        return list(solve_quadratic(in_c2, in_c1, in_c0))
    c2 = in_c2 / ( 3 * in_c3)
    c1 = in_c1 / ( 3 * in_c3)
    c0 = in_c0 / in_c3
    
    # (d0, d1, d2) are called Delta in the article. 
    d0 = -c2**2 + c1
    d1 = -c1 * c2 + c0
    d2 = c2 * c0 - c1**2
    # d is called the discriminant.
    d = 4 * d0 * d2 - d1**2
    # de is called Depressed.x. Depressed.y = d0
    de = -2 * c2 * d0 + d1
    
    if d < 0:
        sq = math.sqrt(-.25 * d)
        r = - .5 * de
        t1 = math.cbrt(r + sq) + math.cbrt(r - sq)
        return [t1 - c2]
    elif d == 0: 
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

def testing_cubic(a, b, c, d):
    print("Input: ", sorted([a, b, c]))
    c2 = - d * (a + b + c)
    c1 = d * (a * b + a * c + b * c)
    c0 = d * (- a * b * c)
    print("Output: ", sorted(solve_cubic(c0, c1, c2, d)))


def solve_quartic(a: float, b: float, c: float, d: float, rescale = False):
    """Solves the quartic equation x**4 + a * x**3 + b * x**2 + c * x + d = 0."""
    # TODO: Move error handling to a wrapper function.
    # try:
    #     phi = dominant_root(a, b, c, d)
    # except OverflowError or ValueError:
    #     return solve_quartic(a, b, c, d, True)
    phi = dominant_root(a, b, c, d)
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

    # Find the best pairs.
    cur_best = None
    best = 0
    for i, pair in enumerate(pairs): 
        e = epsilon_l(b, c, l1, l3, d, pair[0], pair[1])
        if cur_best is None or e < cur_best:
            cur_best = e
            best = i
    d2, l2 = pairs[best]

    #Three different cases d2 < 0 done, d2 > 0, d2 == 0
    EPSILON_M = 2.22045e-16
    candidates = []
    if d2 == 0 or abs(d2) <= EPSILON_M * max(abs(2 * b / 3), abs(phi), l1**2):
        print("Complex case.")
        alpha1, beta1, alpha2, beta2 = case3(l1, l3, d)
        candidates.append((alpha1, beta1, alpha2, beta2))

    if d2 < 0: 
        alpha1, beta1, alpha2, beta2 = case1(a, b, c, d, l1, l2, l3, d2)
        candidates.append((alpha1, beta1, alpha2, beta2))
    elif d2 > 0:
        alpha1, beta1, alpha2, beta2 = case2(l1, l2, l3, d2)
        candidates.append((alpha1, beta1, alpha2, beta2))
    
    if len(candidates) == 2:
        # TODO: Move this to epsilon_q2
        val1 = epsilon_q2(a, b, c, d, *candidates[0])
        val2 = epsilon_q2(a, b, c, d, *candidates[1])
        if val1 < val2:
            alpha1, beta1, alpha2, beta2 = candidates[0]
        else:
            alpha1, beta1, alpha2, beta2 = candidates[1]

    # TODO: Fix the complaint that beta1 might be undbound.
    solutions = solve_quad(alpha1, beta1, alpha2, beta2)
    print("SOL: ", solutions)
    return solutions
    # TODO: We have now found the coefficients of the equivalent second degree equations p1(x) and p2(x).
    # We must now refine the coefficients. 

def case1(a, b, c, d, l1, l2, l3, d2):
    alpha1 = l1 + math.sqrt(-d2)
    beta1 = l3 + math.sqrt(-d2) * l2
    alpha2 = l1 - math.sqrt(-d2)
    beta2 = l3 - math.sqrt(-d2) * l2

    if abs(beta2) <= abs(beta1): 
        beta2 = d / beta1
    else:
        beta1 = d / beta2 
   
    if abs(alpha1) <= abs(alpha2): # Trust alpha2 and find the best alpha1.
        cands = []
        if beta2 != 0:
            alpha11 = (c - beta1 * alpha2) / beta2
            cands.append(alpha11)
        if alpha2 != 0:
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
        if beta1 != 0:
            alpha21 = (c - alpha1 * beta2) / beta1
            cands.append(alpha21)
        if alpha1 != 0:
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
    alpha1 = l1 + complex(0,1) * math.sqrt(d2)
    beta1 = l3 + complex(0,1) * math.sqrt(d2) * l2 
    alpha2 = l1 - complex(0,1) * math.sqrt(d2)
    beta2 = l3 - complex(0,1) * math.sqrt(d2) * l2
    return alpha1, beta1, alpha2, beta2 

def case3(l1, l3, d):
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

def epsilon_q(a, b, c, alpha1, beta1, alpha2, beta2):
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

def epsilon_q2(a, b, c, d, alpha1, beta1, alpha2, beta2):
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

def solve_quad(alpha1, beta1, alpha2, beta2):
    # if isntance(a, complex) for a in (alpha1, beta1, alpha2, beta2):
    #     pass
    # else:
    # TODO: This is for real alpha, beta.
    # Add case for complex alpha, beta.
    delta1 = alpha1**2 - 4 * beta1
    delta2 = alpha2**2 - 4 * beta2
    if delta1 < 0: 
        x0 = - alpha1 / 2 + 1/2 * math.sqrt(-delta1) * complex(0,1)
        x1 = - alpha1 / 2 - 1/2 * math.sqrt(-delta1) * complex(0,1)
    else: 
        if alpha1 >= 0:
            etaM1 = - alpha1 / 2 - math.sqrt(delta1) / 2 
        else:
            etaM1 = - alpha1 / 2 + math.sqrt(delta1) / 2 
        if etaM1 == 0:
            etam1 = 0
        else:
            etam1 = beta1 / etaM1
        x0 = etaM1
        x1 = etam1
    if delta2 < 0:
        x2 = - alpha2 / 2 + 1/2 * math.sqrt(-delta2) * complex(0,1)
        x3 = - alpha2 / 2 - 1/2 * math.sqrt(-delta2) * complex(0,1)
    else:
        if alpha2 >= 0:
            etaM2 = - alpha2 / 2 - math.sqrt(delta2) / 2 
        else:
            etaM2 = - alpha2 / 2 + math.sqrt(delta2) / 2 
        if etaM2 == 0:
            etam2 = 0
        else:
            etam2 = beta2 / etaM2
        x2 = etaM2
        x3 = etam2 
    return x0, x1, x2, x3

def dominant_root(a: float, b: float, c: float, d: float, rescale = False):
    """Returns the dominant root of the depressed cubic equation phi**3 + g phi + h = 0."""

    # TODO: In case this fails to produce a solution then: 
    # First try to rescale the quartic using kq = 7.16e76
    # Else, if that fails to produce a root for the cubic, then 
    # try rescaling the cubic with kc = 3.49e102
    # All this code can actually go into the cubic solver, not here

    # Determine s so that b'(s) = 0, or db'/ds = 0. 
    if 9 * a**2 - 24 * b >= 0:
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
        print("Large Q or R:", Q, R)
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
            print("R2 > Q3")
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
    EPSILON_M = 2.22045e-16
    x = phi
    f = (x**2 + gp) * x + hp

    if abs(f) < EPSILON_M * max(abs(x**3), abs(gp * x), abs(hp)):
        print("Already good.")
        return x
    n = 0
    while(n < 1000):
        df = 3 * x**2 + gp 
        if df == 0.0:
            print("Already good 2.")
            return x 

        x0 = x
        f0 = f
        x = x - f / df
        f = (x**2 + gp) * x + hp
        if abs(f) == 0.0:
            print("Good 3")
            return x
        if abs(f) > abs(f0):
            print("Getting worse.")
            return x0
        n += 1
    return phi
    # If not terminated, the calculation must have Overflown.
    # raise ValueError("Refinement did not converge fast enough.")
    # TODO: If this blows up, we need to do something. 
    # TODO: Add the two things below for testing.
    # except OverflowError: x = float('inf')
    # except DomainError: x = float('NaN') 

    print(a, b, c)
