import itertools 
import math

def solve_quadratic(a, b, c):
    """Solve quadratic equation a * x**2 + b * x + c = 0.
    Discards complex solutions. Avoids cancellation errors."""
    if a == 0:
        if b == 0:
            return ()
        else: 
            return (- c / b)
    else: 
        d = b**2 - 4 * a * c
        if d > 0: 
            q = -.5 * (b + math.copysign(math.sqrt(d) , b))
            x1 = q / a
            x2 = c / q
            return (x1 , x2) if x1 > x2 else (x2, x1)
        elif d == 0: 
            return (- .5 * b / a)
        else: # Complex solutions only.
            return ()


def solve_quartic(a, b, c, d):
    """Solves x**4 + a * x**3 + b * x**2 + c * x1 + d = 0.
    Returns a list of solutions."""

    phi = 0 # Should be found by solving the quadratic equation. 
    l1 = a / 2
    l3 = (b + 3 * phi) / 6
    d1 = 1 
    d2 = (8 * b - 3 * a**2 - 12 * phi) / 12 # Risk of cancellation errors
    
    # This is the main case.
    if d2 != 0:
        # We need to determine which expression for l2 and d2 to use.
        # We calculate each using three different expressions and then use the
        # ones with least error. 
        d3 = 0     
        delta2 = c - a * l3
        # The different candidates. 
        d21 = 2 * b / 3 - phi - l1**2 
        l21 = delta2 / (2 * d2)
        l22 = 2 * (d - l3**2) / delta2 if delta2 != 0 else None
        d22 = (c - a * l3) / ( 2 * l22) if l22 != 0 else None
        d23 = 2 * b / 3 - phi - l1**2 
        l23 = 2 * (d - l3**2) / delta2 if delta2 != 0 else None
        
        l2_candidates = [l for l in [l21, l22, l23] if l is not None]
        d2_candidates = [d for d in [d21, d22, d23] if d is not None]
        print("l2 and d2 candidates:", l2_candidates, "\n", d2_candidates)
        eps_min = None
        eps_index = (None, None)
        for i_l, l2 in enumerate(l2_candidates):
            for i_d, d2 in enumerate(d2_candidates):
                eps0 = abs(d2 + l1**2 + 2 * l3) if b == 0 else abs((d2 + l1**2 + 2 * l3 - b) / b)
                eps1 = abs(2 * d2 * l2 + 2 * l1 + l3) if c == 0 else abs((2 * d2 * l2 + 2 * l1 * l3 - c) / c)
                eps2 = abs(d2 * l2**2 + l3**2) if d == 0 else abs((d2 * l2**2 + l3 ** 2 - d) / d)
                print((eps0, eps1, eps2))
                print(eps0 + eps1 + eps2, i_l, i_d)
                if eps_min is None or eps0 + eps1 + eps2 < eps_min:
                    eps_min = eps0 + eps1 + eps2
                    eps_index = (i_l, i_d)


        print(eps_index)
        l2 = l2_candidates[eps_index[0]]
        d2 = d2_candidates[eps_index[1]]
        print(l2, d2)
