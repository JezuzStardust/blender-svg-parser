# ##### BEGIN GPL LICENSE BLOCK #####
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
#
# ##### END GPL LICENSE BLOCK #####
# <pep8 compliant>
#
# Based on the official Blender addon "Scalable Vector Graphics (SVG) 1.1 format" by JM Soler, Sergey Sharybin
# Additions and modifications:
# Copyright (C) 2020 Jens Zamanian, https://github.com/JezuzStardust

from mathutils import Vector, Matrix
from math import pi, sin, cos, tan, sqrt, acos

def svg_transform_translate(params):
    """
    Returns a translation matrix.
    """
    tx = float(params[0])
    ty = float(params[1]) if len(params) > 1 else 0
    m = Matrix.Translation(Vector((tx, ty, 0)))
    return m


def svg_transform_scale(params):
    """
    Returns a scale matrix.
    """
    sx = float(params[0])
    sy = float(params[1]) if len(params) > 1 else sx
    m = Matrix.Scale(sx, 4, Vector((1, 0, 0)))
    m = m @ Matrix.Scale(sy, 4, Vector((0, 1, 0)))
    return m


def svg_transform_rotate(params):
    """
    Returns a rotation matrix.
    """
    angle = float(params[0]) * pi / 180
    cx = cy = 0
    if len(params) >= 3:
        cx = float(params[1])
        cy = float(params[2])
    tm = Matrix.Translation(Vector((cx, cy, 0)))  # Translation
    rm = Matrix.Rotation(angle, 4, Vector((0, 0, 1)))  # Rotation
    # Translate (-cx, -cy), then rotate, then translate (cx, cy).
    m = tm @ rm @ tm.inverted()
    return m


def svg_transform_skewX(params):
    """
    Returns a skewX matrix.
    """
    angle = float(params[0]) * pi / 180
    m = Matrix(((1.0, tan(angle), 0), (0, 1, 0), (0, 0, 1))).to_4x4()
    return m


def svg_transform_skewY(params):
    """
    Returns a skewY matrix.
    """
    angle = float(params[0]) * pi / 180
    m = Matrix(((1.0, 0, 0), (tan(angle), 1, 0), (0, 0, 1))).to_4x4()
    return m


def svg_transform_matrix(params):
    """
    Returns a matrix transform matrix.
    """
    a = float(params[0])
    b = float(params[1])
    c = float(params[2])
    d = float(params[3])
    e = float(params[4])
    f = float(params[5])
    m = Matrix(((a, c, 0, e), (b, d, 0, f), (0, 0, 1, 0), (0, 0, 0, 1)))
    return m


SVG_TRANSFORMS = {
    "translate": svg_transform_translate,
    "scale": svg_transform_scale,
    "rotate": svg_transform_rotate,
    "skewX": svg_transform_skewX,
    "skewY": svg_transform_skewY,
    "matrix": svg_transform_matrix,
}

