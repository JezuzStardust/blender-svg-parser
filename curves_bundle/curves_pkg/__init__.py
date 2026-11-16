from __future__ import annotations
import math
try:
    from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple, List, Dict, Self
except ImportError:
    from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple, List, Dict
    try:
        from typing_extensions import Self  # type: ignore
    except Exception:
        Self = Any

from .vector import Vector
from .matrix import Matrix
from .bezier import Bezier, QuadraticBezier, OffsetBezier
from .spline import Spline
from .curve import Curve, CurveObject
from .utils import *
__all__ = ['Vector', 'Matrix', 'Bezier', 'QuadraticBezier', 'OffsetBezier', 'Spline', 'Curve', 'CurveObject']