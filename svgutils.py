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

import re

################################################################################
# Regular Expressions
################################################################################

# Match number e.g. -0.232E-23 or .23e1
# Breakdown:
# Optional minus sign
# One or more digits
# Optional group: Period followed by zero or more digits.
# Optional group: e or E followed by optional sign followed by one or more digits.
# The optional pattern after | is for the cases where the integer part is not present.
match_number = r"([+-]?(\d+(\.\d*)?|[+-]?(\.\d+))([eE][+-]?\d+)?)"
# match_number = r'(-?\d+(\.\d*)?([eE][-+]?\d+)?)|(-?\.\d+([eE][-+]?\d+)?)'
re_match_number = re.compile(match_number)

# Match color.
# Colors can be '#' <hex> <hex> <hex> ( <hex> <hex> <hex> ) ')'
# or 'rgb(' wsp* <int> comma <int> comma <int> wsp* ')'
# or 'rgb(' wsp* <int> '%' comma <int> '%' comma <int> '%' wsp* ')'
# or a color keyword.
# Where comma = wsp* ',' wsp*
match_rgb = r"rgb\(\s*(\d+)(%)?\s*,\s*(\d+)(%)?\s*,\s*(\d+)(%)?\s*\)"
re_match_rgb = re.compile(match_rgb)

# Match a float or a letter.
# (?:...) is a non-capturing group. We do not need the individual components.
# TODO: Perhaps make use of match_number pattern above.
match_float_or_letter = r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|\w"
re_match_float_or_letter = re.compile(match_float_or_letter)

# Match transform e.g. skewX(23)
# First match group is name of transform and
# second group is parameters.
# Breakdown:
# Zero or more spaces \s*
# Followed by one or more letters ([A-z]+), first capture group
# Followed by zero or more spaces \s*
# Followed by left parenthesis \(
# Followed by one or more (as few as possible) characters, *? means lazy, second capture group
# Followed by right parenthesis
match_transform = r"\s*([A-z]+)\s*\((.*?)\)"
re_match_transform = re.compile(match_transform)

# Match the align and meet or slice properties.  
# group(0) matches all
# group(1) matches align (either none or e.g. xMinYMax)
# group(2) matches comma + align variable.
# group(3) matches comma-wsp
# group(4) matches meetOrSlice.
# Option 'defer' is not handled.
match_align_meet_or_slice = r"\s*([A-z]+)((\s*,\s*|\s+)([A-z]+))?"
re_match_align_meet_or_slice = re.compile(match_align_meet_or_slice)

################################################################################
# End: Regular Expressions
################################################################################

################################################################################
# Reading Coordinates
################################################################################

# For 96 dpi:
# 1 in = 96 px
# 1 cm = 96 / 2.54 px
# 1 mm = 96 / 25.4 px
# 1 pt = 1 / 72 in = 96 / 72 px = 1.33... px
# 1 pc = 16 px
# Fix em an ex if SVG text support is added.
# The em and ex are relative to the font-size if present.
# E.g. if font-size="150" is used, then 1 em = 150 px.
# em units. Equivalent to the computed font-size in effect for an element.
# ex units. Equivalent to the height of a lower-case letter in the font.
# If the font doesn’t include lower-case letters, or doesn’t include the metadata about the ex-height, then 1ex = 0.5em.

SVG_UNITS = {
    "": 1.0,
    "px": 1.0,
    "in": 96.0,
    "mm": 96.0 / 25.4,
    "cm": 96.0 / 2.54,
    "pt": 96 / 72,  # 1 / 72 in = 96 / 72 px
    "pc": 15.0,
    "em": 1.0,
    "ex": 1.0,
}

def svg_parse_coord(coord, size=0):  # Perhaps the size should always be used.
    """
    Parse a coordinate component from a string.
    Converts the number to a common unit (pixels).
    The size of the surrounding dimension is used in case
    the value is given in percentage.
    """
    value_string, end_index = read_float(coord)
    value = float(value_string)
    unit = coord[end_index:].strip()  # removes extra spaces.
    if unit == "%":
        return float(size) / 100 * value
    else:
        return value * SVG_UNITS[unit]

def read_float(text, start_index=0):
    """
    Reads a float value from a string, starting from start_index.

    Returns the value as a string and the index to the first character after the value.
    """

    n = len(text)

    # Skip leading white spaces and commas.
    while start_index < n and (text[start_index].isspace() or text[start_index] == ","):
        start_index += 1

    if start_index == n:
        return "0", start_index

    text_part = text[start_index:]
    match = re_match_number.match(text_part)

    if match is None:
        raise Exception(
            "Invalid float value near " + text[start_index : start_index + 10]
        )

    value_string = match.group(0)
    end_index = start_index + match.end(0)

    return value_string, end_index

def srgb_to_linear(color):
    """
    Convert sRGB values into linear color space values.

    Input: color = single float value for one of the R, G, and B channels.
    Returns: float

    Blenders colors should be entered in linear color space if the
    Display Device setting is either 'sRGB' or 'XYZ' (i.e. if it is
    not 'None').
    In this case we need to convert the sRGB values that SVG uses
    into a linear color space.
    Ref: https://entropymine.com/imageworsener/srgbformula/
    """
    if color < 0.04045:
        return 0.0 if color < 0.0 else color / 12.92
    else:
        return (color + 0.055) ** 2.4


# CONSTANTS

SVG_EMPTY_STYLE = {
    "fill": None,
    "stroke": None,
    "stroke-width": None,
    "stroke-linecap": None,
    "stroke-linejoin": None,
    "stroke-miterlimit": None,
}

SVG_DEFAULT_STYLE = {
    "fill": "#000000",
    "stroke": "none",
    "stroke-width": "none",
    "stroke-linecap": "butt",
    "stroke-linejoin": "miter",
    "stroke-miterlimit": 4,
}

