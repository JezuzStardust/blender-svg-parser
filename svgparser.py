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
# Based on the Blender addon "IMPORT SVG CURVE?????" by JM Soler, Sergey Sharybin 
# Additions and mofications: 
# Copyright (C) 2020 Jens Zamanian, https://github.com/JezuzStardust

import xml.dom.minidom
import re
import bpy
from mathutils import Matrix, Vector
from math import tan, sin, cos, acos, sqrt, pi
import os
from svgparser import svgcolors


### Reading Coordinates ###
# For 96 dpi:  
# 1 in = 96 px 
# 1 cm = 96 / 2.54 px 
# 1 mm = 96 / 25.4 px 
# 1 pt = 1 / 72 in = 96 / 72 px = 1.33... px 
# 1 pc = 16 px 
# TODO: Fix em an ex if text handling is added.
# The em and ex are relative to the font-size if present.  
# E.g. if font-size="150" is used, then 1 em = 150 px. 
# Em units. Equivalent to the computed font-size in effect for an element.
# Ex units. Equivalent to the height of a lower-case letter in the font.
# If the font doesn’t include lower-case letters, or doesn’t include the metadata about the ex-height, then 1ex = 0.5em.
# SVG_UNITS = {'': 1.0,
#         'px': 1.0,
        # 'in': 90.0,
        # 'mm': 90.0 / 25.4,
        # 'cm': 90.0 / 2.54,
        # 'pt': 1.25, # 1 / 72 in = 90 / 72 px 
        # 'pc': 15.0,
        # 'em': 1.0,
        # 'ex': 1.0
        # }

SVG_UNITS = {'': 1.0,
        'px': 1.0,
        'in': 96.0,
        'mm': 96.0 / 25.4,
        'cm': 96.0 / 2.54,
        'pt': 96 / 72 , # 1 / 72 in = 96 / 72 px 
        'pc': 15.0,
        'em': 1.0,
        'ex': 1.0
        }

### Regexp Patterns ###
# TODO: Consider making these class variables in their specific classes. 

# Match number e.g. -0.232E-23 or .23e1 
# Breakdown:
# Optional minus sign
# One or more digits
# Optional group: . followed by zero or more digits. 
# Optional group e or E followed by optional sign followed by one or more digits. 
# The optional pattern after | is for the cases where the integer part is not present. 
match_number = r'([+-]?(\d+(\.\d*)?|[+-]?(\.\d+))([eE][+-]?\d+)?)'
# match_number = r'(-?\d+(\.\d*)?([eE][-+]?\d+)?)|(-?\.\d+([eE][-+]?\d+)?)' 
re_match_number = re.compile(match_number)  

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
match_transform = r'\s*([A-z]+)\s*\((.*?)\)' 
re_match_transform = re.compile(match_transform)

# Match color. 
# Colors can be '#' <hex> <hex> <hex> ( <hex> <hex> <hex> ) ')'
# or 'rgb(' wsp* <int> comma <int> comma <int> wsp* ')'
# or 'rgb(' wsp* <int> '%' comma <int> '%' comma <int> '%' wsp* ')'
# or a color keyword. 
# Where comma = wsp* ',' wsp*
match_rgb = r'rgb\(\s*(\d+)(%)?\s*,\s*(\d+)(%)?\s*,\s*(\d+)(%)?\s*\)' 
re_match_rgb = re.compile(match_rgb)

# Match a float or a letter. 
# (?:...) is a non-capturing group. We do not need the individual components. 
# TODO: Perhaps make use of match_number pattern above. 
match_float_or_letter = r'(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|\w'
re_match_float_or_letter = re.compile(match_float_or_letter) 

### Utility Functions ###

def read_float(text, start_index = 0):
    """ 
    Reads a float value from a string, starting from start_index. 

    Returns the value as a string and the index to the first character after the value. 
    """

    n = len(text)

    # Skip leading white spaces and commas. 
    while start_index < n and (text[start_index].isspace() or text[start_index] == ','):
        start_index += 1

    if start_index == n:
        return '0', start_index

    text_part = text[start_index:]
    match = re_match_number.match(text_part) 

    if match is None:
        raise Exception('Invalid float value near ' + text[start_index:start_index + 10])
        
    value_string = match.group(0)
    end_index = start_index + match.end(0) 

    return value_string, end_index

def svg_parse_coord(coord, size = 0): # Perhaps the size should always be used.
    """
    Parse a coordinate component from a string. 
    Converts the number to a common unit (pixels). 
    The size of the surrounding dimension is used in case 
    the value is given in percentage. 
    """

    value_string, end_index = read_float(coord) 
    value = float(value_string)
    unit = coord[end_index:].strip() # removes extra spaces. 

    if unit == '%':
        return float(size) / 100 * value
    else: 
        return value * SVG_UNITS[unit] 

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


### End: Utility Functions ###


### Constants ###

SVG_EMPTY_STYLE = {'fill': None,
                   'stroke': None,
                   'stroke-width': None,
                   'stroke-linecap': None,
                   'stroke-linejoin': None,
                   'stroke-miterlimit': None
                   }

SVG_DEFAULT_STYLE = {'fill': '#000',
                   'stroke': 'none',
                   'stroke-width': 'none',
                   'stroke-linecap': 'butt',
                   'stroke-linejoin': 'miter',
                   'stroke-miterlimit': 4 
                   }

# fill:                 Fill color. Should be initialized to black!
# stroke:               Stroke color. 
# stroke-width:         Width of the stroke. 
# stroke-linecap:       End cap of stroke (butt, round, square).
# stroke-linejoin:      Shape of path joints (miter, round, bevel)
# stroke-miterlimit:    How far, in stroke-width:s, the miter joint can stretch. 

### End: Constants ###


### Transformation Functions ###

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
    tm = Matrix.Translation(Vector((cx, cy, 0))) # Translation
    rm = Matrix.Rotation(angle, 4, Vector((0, 0, 1))) # Rotation
    # Translate (-cx, -cy), then rotate, then translate (cx, cy). 
    m = tm @ rm @ tm.inverted() 
    return m

def svg_transform_skewX(params):
    """
    Returns a skewX matrix. 
    """
    angle = float(params[0]) * pi / 180
    m = Matrix(((1.0,   tan(angle), 0),
                (0,              1, 0),
                (0,              0, 1))).to_4x4()
    return m

def svg_transform_skewY(params):
    """
    Returns a skewY matrix. 
    """
    angle = float(params[0]) * pi / 180
    m = Matrix(((1.0,        0,    0),
               (tan(angle), 1,    0),
               (0,        0,    1))).to_4x4()
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
    m = Matrix(((a, c, 0, e),
                (b, d, 0, f),
                (0, 0, 1, 0),
                (0, 0, 0, 1)))
    return m

SVG_TRANSFORMS = {'translate': svg_transform_translate, 
                 'scale': svg_transform_scale,
                 'rotate': svg_transform_rotate,
                 'skewX': svg_transform_skewX,
                 'skewY': svg_transform_skewY,
                 'matrix': svg_transform_matrix
                }

### End: Transformation Functions ###


### Classes ###

class SVGGeometry():
    """
    Geometry base class. 
    """

    __slots__ = ('_node',
                 '_transform',
                 '_style',
                 '_context',
                 '_viewport',
                 '_viewBox',
                 '_preserveAspectRatio',
                 '_name',
                 )

    def __init__(self, node, context):
        """
        Initialize the base class. 
        Has a reference to the node, the context (transformations, styles, stack).
        """
        self._node = node
        self._transform = Matrix()
        self._style = SVG_EMPTY_STYLE
        self._context = context
        self._name = None

    def parse(self):
        """
        Parses the style and transformations on the node. 
        Some nodes do not have style, fill, stroke, etc, or transform.
        However, in the _parse_transformation we always check first 
        if these attributes actually exists first. 
        """
        if type(self._node) is xml.dom.minidom.Element:
            self._style = self._parse_style()
            self._transform = self._parse_transform()
            # If the node has an id or class store reference to the instance. 
            # Also store the name. 
            for attr in ('id', 'class'):
                id_or_class = self._node.getAttribute(attr)
                if id_or_class:
                    if self._context['defs'].get('#' + id_or_class) is None:
                        self._context['defs']['#' + id_or_class] = self
                    if not self._name:
                        self._name = id_or_class # Prefer name from id. 

    def _parse_style(self):
        """
        Parse the style attributes (e.g. fill="black", stroke-width=".1cm", etc) 
        first, then parse the style-attribute (e.g. style='fill:blue;stroke:green'
        In SVG-files style="fill:blue;..." takes precedence over e.g. fill="blue".
        """
        style = SVG_EMPTY_STYLE.copy()

        for attr in SVG_EMPTY_STYLE.keys():
            val = self._node.getAttribute(attr) 
            if val:
                style[attr] = val.strip().lower()

        style_attr = self._node.getAttribute('style')
        if style_attr:
            elems = style_attr.split(';')
            for elem in elems:
                s = elem.split(':')
                if len(s) != 2:
                    continue
                name = s[0].strip().lower()
                val = s[1].strip()
                if name in SVG_EMPTY_STYLE.keys():
                    style[name] = val
        return style

    def _parse_transform(self):
        """
        Parse the transform attribute on the node. 
        Will only be called on DOM Elements.
        """
        transform = self._node.getAttribute('transform')
        m = Matrix() 
        if transform:
            for match in re_match_transform.finditer(transform):
                trans = match.group(1)
                params = match.group(2)
                params = params.replace(',', ' ').split()
                transform_function = SVG_TRANSFORMS.get(trans)
                if transform_function is None: 
                    raise Exception('Unknown transform function: ' + trans) 
                m = m @ transform_function(params) 
        return m

    def _parse_viewport(self):
        """
        Parse the x, y, width, and height attributes. 
        """
        vp_x = self._node.getAttribute('x') or '0'
        vp_y = self._node.getAttribute('y') or '0'
        vp_width = self._node.getAttribute('width') or '100%'
        vp_height = self._node.getAttribute('height') or '100%'
        return (vp_x, vp_y, vp_width, vp_height)

    def _push_transform(self, transform):
        """
        Pushes the transformation matrix onto the stack.
        """
        m = self._context['current_transform'] 
        self._context['current_transform'] = m @ transform  

    def _pop_transform(self, transform):
        """
        Pops the transformation matrix from the stack.
        """
        m = self._context['current_transform'] 
        self._context['current_transform'] = m @ transform.inverted()
    
    def _pop_viewBox(self):
        """
        """
        if self._viewBox:
            self._context['viewBox_stack'].pop()
            self._context['current_viewBox'] = self._context['viewBox_stack'][-1]

    def _transform_coord(self, co):
        """
        """
        m = self._context['current_transform']
        v = Vector((co[0], co[1], 0))

        return m @ v

    def _new_blender_curve_object(self, name):
        """
        Create new curve object and add it to the Blender collection. 
        """
        # TODO: Eliminate one of this and new_blender_curve. 
        curve = bpy.data.curves.new(name, 'CURVE')
        obj = bpy.data.objects.new(name, curve)
        self._context['blender_collection'].objects.link(obj)
        obj.data.dimensions = '2D'
        obj.data.fill_mode = 'BOTH'

        # TODO: Code repetition in new_blender_curve. 
        style = self._calculate_style_in_context()
        if style['fill'] == 'none':
            obj.data.fill_mode = 'NONE'
        else: 
            obj.data.fill_mode = 'BOTH'
            material = self._get_material_with_color(style['fill'])
            obj.data.materials.append(material)

        # Test
        m = Matrix.Translation((0, 0, .000015))
        self._push_transform(m)

        return obj.data

    def _new_spline_to_blender_curve(self, curve_object_data, is_cyclic):
        """
        Adds a new spline to an existing Blender curve object and returns 
        a reference to the spline.
        """
        # TODO: Move the style calculations elsewhere. 
        style = self._calculate_style_in_context()
        if style['fill'] != 'none':
            is_cyclic = True
        curve_object_data.splines.new('BEZIER')
        spline = curve_object_data.splines[-1]
        spline.use_cyclic_u = is_cyclic 
        return spline

    def _new_blender_curve(self, name, is_cyclic):
        """
        Create new curve object and link it to the Blender collection.
        Then adds a spline to the given curve. 
        """
        # TODO: Keep only one of this and _new_blender_curve_object. 
        curve = bpy.data.curves.new(name, 'CURVE')
        obj = bpy.data.objects.new(name, curve)
        self._context['blender_collection'].objects.link(obj)
        obj.data.dimensions = '2D'

        style = self._calculate_style_in_context()
        if style['fill'] == 'none':
            obj.data.fill_mode = 'NONE'
        else: 
            obj.data.fill_mode = 'BOTH'
            material = self._get_material_with_color(style['fill'])
            obj.data.materials.append(material)

        obj.data.splines.new('BEZIER')
        spline = obj.data.splines[-1]
        spline.use_cyclic_u = is_cyclic
        return spline

    def _add_points_to_blender(self, coords, spline):
        """
        Adds coordinate points and handles to a given spline.

        coords = list of coordinates (point, handle_left, handle_right, ...).
        spline = a reference to bpy.objects[<current curve>].data.splines[-1].
        """
        # TODO: It might be better to create the spline within the loop (for point...). 
        # In this way, we use the first point of the spline directly when it is
        # created and the remaining points are added. 
        # No need for 'first_point'.
        # In that case we might be able to get rid of the 
        # function _new_blender_curve completely.
        # Alternatively, call it from here. 
        first_point = True
        for co in coords:
            if not first_point: 
                spline.bezier_points.add(1)
            bezt = spline.bezier_points[-1]
            bezt.co = self._transform_coord(co[0])
            if co[1]:
                bezt.handle_left = self._transform_coord(co[1])
            else: 
                bezt.handle_left_type = 'VECTOR'
            if co[2]:
                bezt.handle_right = self._transform_coord(co[2])
            else:
                bezt.handle_right_type = 'VECTOR'
            first_point = False

    def _get_material_with_color(self, color):
        """
        Parse a color, creates and add a corresponding material 
        in Blender.
        """
        # Parse the color according to the specification.
        # Create a new Blender material with this color (using nodes).
        # Add the material to the material list in context.

        if color in self._context['materials']:
            return self._context['materials'][color] 

        if color.startswith('#'):
            # According the SVG 1.1 specification, if only three hexdigits
            # are given, they should each be repeated twice. 
            if len(color) == 4:
                color = color[0] + color[1] * 2 + color[2] * 2 + color[3] * 2
            diff = (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
            diffuse_color = [x / 255 for x in diff]
        elif color in svgcolors.SVG_COLORS:
            diff = svgcolors.SVG_COLORS[color] 
            diffuse_color = [x / 255 for x in diff]
        elif re_match_rgb.match(color): # Move this to a utility function. 
            diff = re_match_rgb.findall(color)[0]
            if len(c) == 6:
                diffuse_color = [c[0], c[2], c[4]] 
            else:
                diffuse_color = [x / 255 for x in diff]
        else:
            return None

        if self._context['do_colormanage']:
            diffuse_color = [srgb_to_linear(x) for x in diffuse_color]

        # TODO: Should also check if there is not already a material 
        # in Blender with the same name and reuse it. 
        mat = bpy.data.materials.new(name='SVG_' + color)
        # Set material both in default material and node tree to same color.
        # Otherwise switching to node tree eliminates the color.
        mat.diffuse_color = (*diffuse_color, 1.0)
        mat.use_nodes = True
        mat.node_tree.nodes['Principled BSDF'].inputs[0].default_value = (*diffuse_color, 1.0)
        
        # Add the material to the materials stack.
        self._context['materials'][color] = mat

        return mat

    def _calculate_style_in_context(self):
        """ 
        Starts from default material and successively overwrites 
        the different attributes for each of the parent. 
        In the end, if an attribute e.g. fill is still None, 
        the default value is used. 
        """
        style = self._style
        for sty in reversed(self._context['style_stack']):
            for key in SVG_EMPTY_STYLE.keys():
                if style[key] == None:
                    style[key] = sty[key]

        for key in SVG_DEFAULT_STYLE:
            if style[key] == None:
                style[key] = SVG_DEFAULT_STYLE[key]
        return style

    def _push_style(self, style):
        """
        Pushes the 'style' onto the style stack.
        """
        self._context['style_stack'].append(style)

    def _pop_style(self):
        """
        Pops the last style from the style stack.
        """
        self._context['style_stack'].pop()

    def _get_name_from_node(self):
        """
        Gets the id or class name from the node if present.
        """
        name = None
        name = self._node.getAttribute('id') or self._node.getAttribute('class')
        return name


class SVGGeometryContainer(SVGGeometry):
    """
    Container class for SVGGeometry. 
    Since a container has attributes such as style, and transformations, 
    it inherits from SVGGeometry. 
    """
    __slots__ = ('_geometries')

    def __init__(self, node, context): 
        """
        Initializes the container
        """
        super().__init__(node, context) 
        self._geometries = []

    def parse(self):
        """
        Initializes and parses all the children element nodes and add them 
        to the _geometries list. Also calls parse() of super in order to
        parse style and transform. 
        """
        super().parse() 
        for node in self._node.childNodes:
            if type(node) is not xml.dom.minidom.Element:
                continue
            name = node.tagName
            # Sometimes an SVG namespace (xmlns) is used. 
            if name.startswith('svg:'):
                name = name[4:]
            geometry_class = SVG_GEOMETRY_CLASSES.get(name)
            if geometry_class is not None:
                geometry_instance = geometry_class(node, self._context)
                geometry_instance.parse()
                self._geometries.append(geometry_instance)

    def create_blender_splines(self):
        """
        Make all children elements create splines in Blender.  
        Does not call the creation for SYMBOLS and DEFS, instead
        they will be created via a USE element. 
        """
        # TODO: Should transform be pushed here? 
        self._push_style(self._style)
        for geom in self._geometries:
            if geom.__class__ not in (SVGGeometrySYMBOL, SVGGeometryDEFS):
                geom.create_blender_splines()
        self._pop_style()


class SVGGeometrySVG(SVGGeometryContainer):
    """
    Corresponds to the <svg> elements. 
    """
    __slots__ = ()

    # TODO: Figure out how to set origin in Blender. This requires knowing 
    # the dimensions of the outer SVG element. 
    # This can be done by making the Loader keep a reference to 
    # the outer SVGGeometrySVG instance (a value in context that is only occupied
    # if is empty. The outer will be the first one to be parsed. 

    def parse(self):
        """
        Parse the attributes of the SVG element. 
        The viewport (x, y, width, height) cannot actually be parsed at this time
        since they require knowledge of the ancestor viewBox in case the values
        are given in percentages. 
        """
        if not self._context['outer_SVG']:
            self._context['Outer_SVG'] = self
        self._viewport = self._parse_viewport()
        self._viewBox = self._parse_viewBox()
        self._preserveAspectRatio = self._parse_preserveAspectRatio()
        super().parse()

    def create_blender_splines(self):
        """
        Adds geometry to Blender.
        """
        viewport_transform = self._view_to_transform()
        self._push_transform(viewport_transform)
        self._push_transform(self._transform)
        # If there is no viewBox we inherit it from the current viewport. 
        # Since the viewport is context dependent, this viewBox may change
        # each time the container is used (if referenced multiple times).
        # It is therefore not possible to store the viewBox in self._viewBox. 
        # Instead it is pushed onto a stack. 
        viewBox = self._viewBox
        if not viewBox:
            viewBox = self._calculate_viewBox_from_viewport() 
        self._push_viewBox(viewBox)
        super().create_blender_splines()
        self._pop_viewBox()
        self._pop_transform(self._transform)
        self._pop_transform(viewport_transform) 

    def get_viewport(self):
        """
        Return the viewport. 
        """
        # Mainly done this way since I wanted to define _viewport as pseudo private
        # and should respect that. 
        return self._viewport

    def set_viewport(self, viewport):
        self._viewport = viewport 

    def _parse_viewBox(self):
        """
        Parse the viewBox attribute. 
        """
        # TODO: Move this to SVGGeometryContainer, since this will be 
        # used by SVG and SYMBOL which are both containers. 
        viewBox = self._node.getAttribute('viewBox')
        if viewBox: 
            min_x, min_y, width, height = viewBox.replace(',', ' ').split()
            return (min_x, min_y, width, height)
        else: 
            return None

    def _push_viewBox(self, viewBox):
        """
        """
        if viewBox:
            self._context['current_viewBox'] = viewBox
            self._context['viewBox_stack'].append(viewBox)

    def _parse_preserveAspectRatio(self):
        # TODO: Handle cases where it starts with 'defer' (and ignore this case).
        # TODO: Handle 'none'. However, see _view_to_transform. Might be OK as is.
        # TODO: Move this outside of the function later. 
        preserveAspectRatio = self._node.getAttribute('preserveAspectRatio')
        # group(0) matches all
        # group(1) matches align (either none or e.g. xMinYMax)
        # group(2) matches comma + align variable. 
        # group(3) matches comma-wsp
        # group(4) matches meetOrSlice. 
        # Option 'defer' is not handled. 
        if preserveAspectRatio: 
            pattern = r'\s*([A-z]+)((\s*,\s*|\s+)([A-z]+))?'
            regex = re.compile(pattern) 
            for match in regex.finditer(preserveAspectRatio):
                align = match.group(1) 
                meetOrSlice = match.group(4)
        else: 
            align = 'xMidYMid'
            meetOrSlice = 'meet' 
        align_x = align[:4]
        align_y = align[4:]
        return (align_x, align_y, meetOrSlice)

    def _calculate_viewBox_from_viewport(self):
        """
        Inherit the viewBox from viewport, i.e. use standard coordinates.
        Used when there is not viewBox present. 
        """
        current_viewBox = self._context['current_viewBox']
        viewport = self._viewport
        viewBox_width = svg_parse_coord(viewport[2], current_viewBox[2])
        viewBox_height = svg_parse_coord(viewport[3], current_viewBox[3])
        return (0, 0, viewBox_width, viewBox_height)

    def _view_to_transform(self):
        """
        Parses the viewport and viewBox and converts them into 
        an equivalent transform.
        """
        viewBox = self._viewBox
        viewport = self._viewport
        preserveAspectRatio = self._preserveAspectRatio
        current_viewBox = self._context['current_viewBox'] # Parent's viewBox

        # First parse the viewport and (if necessary) resolve percentages 
        # using the parent's viewBox. 
        # Then parse the viewBox. In case there is no viewBox, 
        # then use the values from the rect. 
        # Parse the SVG viewport.

        # Resolve percentages to parent viewport.
        # If viewport missing, use parent viewBox. 
        if viewport:
            e_x = svg_parse_coord(viewport[0], current_viewBox[0])
            e_y = svg_parse_coord(viewport[1], current_viewBox[1])
            e_width = svg_parse_coord(viewport[2], current_viewBox[2])
            e_height = svg_parse_coord(viewport[3], current_viewBox[3])
        else:
            e_x = 0
            e_y = 0
            e_width = svg_parse_coord('100%', current_viewBox[2])
            e_height = svg_parse_coord('100%', current_viewBox[3])
        # TODO: 
        # In case the outermost SVG does not have a width and height specified
        # they are defaulted to 100%. 
        # The values will then be resolved against the default 
        # viewBox="0 0 0 0"
        # So instead we default it to 100px. 
        # This probably goes against the specification 
        # since it causes problems for internal SVG's which 
        # have width and/or height = 0 (which should not be rendered at all. 
        # However, this is a strange limit case. 
        if e_width == 0.0:
            e_width = 100.0
        if e_height == 0.0:
            e_height = 100.0
        # TODO: Handle 'none'. 
        # This might actually handled by accident by the code below.
        pARx = preserveAspectRatio[0]
        pARy = preserveAspectRatio[1]
        meetOrSlice = preserveAspectRatio[2]

        if viewBox:  
            vb_x = svg_parse_coord(viewBox[0])
            vb_y = svg_parse_coord(viewBox[1])
            vb_width = svg_parse_coord(viewBox[2])
            vb_height = svg_parse_coord(viewBox[3])
        else:
            vb_x = 0
            vb_y = 0
            vb_width = e_width 
            vb_height = e_height
        scale_x = e_width / vb_width 
        scale_y = e_height / vb_height 
        if meetOrSlice == 'meet': # Must also check that align is not none. 
            # TODO: Check how none affects this value.
            scale_x = scale_y = min(scale_x, scale_y)
        elif meetOrSlice == 'slice':
            scale_x = scale_y = max(scale_x, scale_y)
        translate_x = e_x - vb_x * scale_x 
        translate_y = e_y - vb_y * scale_y
        if pARx == 'xMid':
            translate_x += (e_width - vb_width * scale_x) / 2 
        if pARx == 'xMax':    
            translate_x += (e_width - vb_width * scale_x)
        if pARy == 'YMid':
            translate_y += (e_height - vb_height * scale_y) / 2 
        if pARy == 'YMax':
            translate_y += (e_height - vb_height * scale_y) 
        m = Matrix()
        m = m @ Matrix.Translation(Vector((translate_x, translate_y , 0)))
        m = m @ Matrix.Scale(scale_x, 4, Vector((1, 0, 0)))
        m = m @ Matrix.Scale(scale_y, 4, Vector((0, 1, 0)))
        return m


class SVGGeometryG(SVGGeometryContainer):
    """
    Same as SVGGeometryContainer, but can also have transform.
    """
    def create_blender_splines(self):
        self._push_transform(self._transform)
        super().create_blender_splines()
        self._pop_transform(self._transform)


class SVGGeometryRECT(SVGGeometry):
    """
    SVG <rect>.
    """

    __slots__ = ('_x', '_y', '_width', '_height', '_rx', '_ry') 

    def __init__(self, node, context):
        """
        Initialize a new rectangle with default values. 
        """
        super().__init__(node, context)
        self._x = '0'
        self._y = '0'
        self._width = '0'
        self._height = '0'
        self._rx = '0'
        self._ry = '0'

    def parse(self):
        """
        Parse the data from the node and store in the local variables. 
        Reads x, y, width, height, rx, ry from the node. 
        Also reads in the style. 
        Should it also read the transformation? 
        """
        super().parse()
        self._x = self._node.getAttribute('x') or '0'
        self._y = self._node.getAttribute('y') or '0'
        self._width = self._node.getAttribute('width') or '0'
        self._height = self._node.getAttribute('height') or '0'
        self._rx = self._node.getAttribute('rx') or '0'
        self._ry = self._node.getAttribute('ry') or '0'
 
    def create_blender_splines(self):
        """ 
        Create Blender geometry.
        """
        vB = self._context['current_viewBox'][2:] # width and height of viewBox.
        x = svg_parse_coord(self._x, vB[0])
        y = svg_parse_coord(self._y, vB[1])
        w = svg_parse_coord(self._width, vB[0])
        h = svg_parse_coord(self._height, vB[1]) 
        rx = ry = 0
        rad_x = self._rx 
        rad_y = self._ry
        # For radii rx and ry, resolve % values against half the width and height,
        # respectively. It is not clear from the specification which width
        # and height are considered. 
        # In SVG 2.0 it seems to indicate that it should be the width and height
        # of the rectangle. However, to be consistent with other % it is 
        # most likely the width and height of the current viewBox.  
        # If only one is given then the other one should be the same. 
        # Then clamp the values to width/2 respectively height/2.
        # 100% means half the width or height of the viewBox (or viewport).
        # https://www.w3.org/TR/SVG11/shapes.html#RectElement 
        rounded = True
        if rad_x != '0' and rad_y != '0':
            rx = min(svg_parse_coord(rad_x, vB[0]), w/2) 
            ry = min(svg_parse_coord(rad_y, vB[1]), h/2)
        elif rad_x != '0':
            rx = min(svg_parse_coord(rad_x, vB[0]), w/2)
            ry = min(rx, h/2)
        elif rad_y != '0':
            ry = min(svg_parse_coord(rad_y, vB[1]), h/2)
            rx = min(ry, w/2)
        else: 
            rounded = False
        # Approximation of elliptic curve for corner.
        # Put the handles semi minor(or major) axis radius times 
        # factor = (sqrt(7) - 1)/3 away from Bezier point.
        # http://www.spaceroots.org/documents/ellipse/elliptical-arc.pdf
        factor_x = rx * (sqrt(7) - 1)/3
        factor_y = ry * (sqrt(7) - 1)/3
        # TODO: Probably better to use a specific class for all Bezier curves. 

        if rounded:
            coords = [((x + rx, y), (x + rx - factor_x, y), None),
                      ((x + w - rx, y), None, (x + w - rx + factor_x, y)),
                      ((x + w, y + ry), (x + w, y + ry - factor_y), None),
                      ((x + w, y + h - ry), None, (x + w, y + h - ry + factor_y)),
                      ((x + w - rx, y + h), (x + w - rx + factor_x, y + h), None),
                      ((x + rx, y + h), None, (x + rx - factor_x, y + h)),
                      ((x, y + h - ry), (x, y + h - ry + factor_y), None),
                      ((x, y + ry), None, (x, y + ry - factor_y))]
        else:
            coords = [((x,y), None, None),
                      ((x + w, y), None, None),
                      ((x + w, y + h), None, None),
                      ((x, y + h), None, None)]

        # TODO: Move this to a general purpose function. 
        # Perhaps name can be defined in SVGGeometry even, since 
        # all elements can have names. 
        if not self._name:
            self._name = 'Rect'

        spline = self._new_blender_curve(self._name, True) 
        self._push_transform(self._transform)
        self._add_points_to_blender(coords, spline)
        self._pop_transform(self._transform)

    def _new_point(coordinate, handle_left = None, handle_right = None, in_type = None, out_type = None):
        return {'coordinates': coordinate, 'handle_left': handle_right, 'handle_right': handle_left, 'in_type': in_type, 'out_type': out_type}

    def _new_path(is_closed = False):
        return {'points': [], 'is_closed': is_closed} 


class SVGGeometryELLIPSE(SVGGeometry):
    """
    SVG <ellipse>. 
    """
    __slots__ = ('_cx',
                 '_cy',
                 '_rx',
                 '_ry',
                 '_is_circle')
                 

    def __init__(self, node, context):
        """ 
        Initialize the ellipse with default values (all zero). 
        """
        super().__init__(node, context)
        self._is_circle = False
        self._cx = '0'
        self._cy = '0'
        self._rx = '0'
        self._ry = '0'

    def parse(self):
        """ 
        Parses the data from the <ellipse> element.
        """
        super().parse()
        self._cx = self._node.getAttribute('cx') or '0'
        self._cy = self._node.getAttribute('cy') or '0'
        self._rx = self._node.getAttribute('rx') or '0'
        self._ry = self._node.getAttribute('ry') or '0'
        r = self._node.getAttribute('r') or '0'
        
        if r != '0':
            self._is_circle = True
            self._rx = r

    def create_blender_splines(self):
        """ 
        Create Blender geometry.
        """
        vB = self._context['current_viewBox'][2:] # width and height of viewBox.
        cx = svg_parse_coord(self._cx, vB[0])
        cy = svg_parse_coord(self._cy, vB[1])
        if self._is_circle:
            weighted_diagonal = sqrt(float(vB[0]) ** 2 + float(vB[1]) ** 2)/sqrt(2)
            rx = ry = svg_parse_coord(self._rx, weighted_diagonal)  
        else:
            rx = svg_parse_coord(self._rx, vB[0])
            ry = svg_parse_coord(self._ry, vB[1]) 
        # Approximation of elliptic curve for corner.
        # Put the handles semi minor(or major) axis radius times 
        # factor = (sqrt(7) - 1)/3 away from Bezier point.
        # http://www.spaceroots.org/documents/ellipse/elliptical-arc.pdf
        factor_x = rx * (sqrt(7) - 1)/3
        factor_y = ry * (sqrt(7) - 1)/3

        # Coordinate, first handle, second handle
        coords = [((cx - rx, cy), (cx - rx, cy + factor_y), (cx - rx, cy - factor_y)),
                  ((cx, cy - ry), (cx - factor_x, cy - ry), (cx + factor_x, cy - ry)),
                  ((cx + rx, cy), (cx + rx, cy - factor_y), (cx + rx, cy + factor_y)),
                  ((cx, cy + ry), (cx + factor_x, cy + ry), (cx - factor_x, cy + ry))]

        if not self._name:
            if self._is_circle:
                self._name = 'Circle'
            else:
                self._name = 'Ellipse'

        spline = self._new_blender_curve(self._name, True)
        self._push_transform(self._transform) 
        self._add_points_to_blender(coords, spline)
        self._pop_transform(self._transform)


class SVGGeometryCIRCLE(SVGGeometryELLIPSE):
    """
    A <circle> element with a lot of reuse of ellipse code. 
    """
    pass # Handled completely by ELLIPSE. 


class SVGGeometryLINE(SVGGeometry):
    """
    SVG <line>. 
    """
    __slots__ = ('_x1',
                 '_y1',
                 '_x2',
                 '_y2')

    def __init__(self, node, context, is_circle = False):
        """ 
        Initialize the ellipse with default values (all zero). 
        """
        super().__init__(node, context)
        self._x1 = '0'
        self._y1 = '0'
        self._x2 = '0'
        self._y2 = '0'

    def parse(self):
        """ 
        Parses the data from the <ellipse> element.
        """
        super().parse()
        self._x1 = self._node.getAttribute('x1') or '0'
        self._y1 = self._node.getAttribute('y1') or '0'
        self._x2 = self._node.getAttribute('x2') or '0'
        self._y2 = self._node.getAttribute('y2') or '0'

    def create_blender_splines(self):
        """ 
        Create Blender geometry.
        """
        vB = self._context['current_viewBox'][2:] # width and height of viewBox.
        x1 = svg_parse_coord(self._x1, vB[0])
        y1 = svg_parse_coord(self._y1, vB[1])
        x2 = svg_parse_coord(self._x2, vB[0])
        y2 = svg_parse_coord(self._y2, vB[1])
        coords = [((x1, y1), None, None), ((x2, y2), None, None)]
        if not self._name:
            self._name = 'Line'
        spline = self._new_blender_curve(self._name, False)
        self._push_transform(self._transform) 
        self._add_points_to_blender(coords, spline)
        self._pop_transform(self._transform)


class SVGGeometryPOLYLINE(SVGGeometry):
    """
    SVG <polyline>.
    """

    __slots__ = ('_points',
                 '_is_closed')

    def __init__(self, node, context):
        """
        Init the <polyline> using default values (points is empty).
        """
        super().__init__(node, context)
        self._is_closed = False
        self._points = []

    def parse(self):
        """
        Parse the node data.
        In this case coordinates cannot be %, so this parsing 
        does not have to be done at time of coordinate creation. 
        """
        super().parse()
        points = self._node.getAttribute('points')
        # TODO: Check if this should be done in a separate function. 
        previous = None
        for p in re_match_number.findall(points):
            # This will skip the last one if an odd number of numbers.
            if previous is None: 
                previous = float(p[0])
            else: 
                self._points.append((previous, float(p[0])))
                previous = None

    def create_blender_splines(self):
        """
        Creates the splines in Blender. 
        """
        if not self._name:
            if self._is_closed: 
                self._name = 'Polygon' # Polygons defaults to closed...
            else:
                self._name = 'Polyline' # ...and Polylines are open.

        if self._style['fill'] != 'none': # But if we have a fill, we must close it anyway.
            self._is_closed = True
        spline = self._new_blender_curve(self._name, self._is_closed)
        self._push_transform(self._transform) 
        # TODO: Rethink this so that it can use the function
        # _add_points_to_blender. 
        # Alternative 1: Remake this code. 
        # Alternative 2: If this is similar for how to handle 
        # paths, then create a new function for these two. 
        # Alternative 3: Keep this code separate as is. 
        first_point = True
        for point in self._points:
            if not first_point:
                spline.bezier_points.add(1)
            bezt = spline.bezier_points[-1]
            bezt.co = self._transform_coord(point)
            bezt.handle_left_type = 'VECTOR'
            bezt.handle_right_type = 'VECTOR'
            first_point = False
        self._pop_transform(self._transform)


class SVGGeometryPOLYGON(SVGGeometryPOLYLINE):
    """
    SVG <polygon>.
    """

    def __init__(self, node, context):
        """
        Init the <polyline> using default values (points is empty).
        """
        super().__init__(node, context)
        self._is_closed = True


class SVGGeometryPATH(SVGGeometry):
    """
    SVG <path>.
    """
    __slots__ = ('_splines')

    def __init__(self, node, context):
        """
        Inits the path to an empty path. 
        """
        super().__init__(node, context)
        self._splines = None

    def parse(self):
        """
        Parses the path data. 
        """
        super().parse()
        d = self._node.getAttribute('d')
        parser = SVGPATHParser(d)
        parser.parse()
        self._splines = parser.get_data()

    def create_blender_splines(self):
        """
        Create the Blender curves needed to draw out the path. 
        """
        # TODO: Refactor this code. 
        # TODO: Change points to a dictionary or something else.  
        # TODO: Move this to SVGGeometry.parse instead. 
        if not self._name:
            self._name = 'Path'
        curve_object_data = self._new_blender_curve_object(self._name)

        self._push_style(self._style)
        self._push_transform(self._transform)
        for spline in self._splines:
            if spline[-1] == 'closed':
                is_cyclic = True
                spline = spline[:-1]
            else:
                is_cyclic = False
            blender_spline = self._new_spline_to_blender_curve(curve_object_data, is_cyclic)
            self._add_points_to_blender(spline, blender_spline)
        self._pop_transform(self._transform)
        self._pop_style()


class SVGPATHParser:
    """
    Helper for parsing path. Used only by SVGGeometryPATH. 

    Input: the d attribute from a path element. 
    Output: a list with points for the spline. 
    """
    # TODO: Make this nicer. 
    # TODO: Make all geometry classes use this? 
    # Perhaps this makes them less explicit...
    # In that case the data should be a string which is then passed
    # to the PATHDataSupplier. Perhaps this is not so nice. 
    # Perhaps geometry classes can prepare the data the same way the 
    # data supplier does and make the use of that optional? 

    # TODO: Consider using a dict for points. 
    # It is hard to remember which entry is what without it.
    # In this case, we should also consider using this for all classes
    # that create geometry. 
    # TODO: It might be useful to have a separate variable for 
    # the last point, since it is used so frequently. 
    # In that case there can be a function which adds the point both
    # to the current spline and updates the last point. 
    # TODO: Refactor the code. A lot of repetition (e.g. the previous 
    # point is updated in a similar manner in all cases. 
    __slots__ = ('_data',
                 '_index',
                 '_current_spline',
                 '_splines', # List containing all finished curves. 
                 '_commands')

    def __init__(self, d): 
        """
        """
        self._data = SVGPATHDataSupplier(d)
        self._index = 0
        self._current_spline = None
        self._splines = []
        self._commands = {'M': self._path_move_to, 
                          'L': self._path_line_to,
                          'H': self._path_line_to,
                          'V': self._path_line_to,
                          'C': self._path_curve_to,
                          'S': self._path_curve_to,
                          'Q': self._path_curve_to,
                          'T': self._path_curve_to,
                          'A': self._path_elliptical_arc_to,
                          'Z': self._path_close,
                          }

    def parse(self):
        # Split out data handler into separate class. 
        # This class should have a dict with names of 
        # the functions in question. 
        while not self._data.eof():
            cmd = self._data.next()
            parse_function = self._commands.get(cmd.upper())
            parse_function(cmd)
        self._splines.append(self._current_spline)

    def get_data(self):
        return self._splines 

    def _get_next_coord_pair(self, relative, previous_point):
        """
        Extract the next two coordinates from the data. 
        """
        x = self._data.next_coord()
        y = self._data.next_coord()
        if relative and previous_point is not None:
            x += previous_point[0]
            y += previous_point[1]
        return x, y

    def _get_next_coord(self, relative, previous_point):
        """
        Extract the next coordinate from the data. 
        """
        p = self._data.next_coord()
        if relative and previous_point is not None:
            p += previous_point
        return p

    def _get_last_point(self):
        """
        Return the last point. 
        We need to know the coordinates if the next point has relative coordinates.
        But we also need to update it when we add the next point (specifically the 
        handles). 
        """
        if self._current_spline is None:
            return None
        else:
            return self._current_spline[-1]

    def _path_move_to(self, command):
        """
        The SVG Path M and m commands. 
        Moves to a new point and creates a new subpath. 
        """
        # TODO: Fix bug! A move-to should not produce a new Blender object. 
        # Instead it should just produce a new spline within the current curve
        # object. 
        if self._current_spline is not None:
            self._splines.append(self._current_spline)
        self._current_spline = []
        # Check if m is used and move the counter forward in the data. 
        relative = command.islower() 
        x, y = self._get_next_coord_pair(relative, None) # The first move does not care if relative...
        # Should I wait with appending the points until 
        # I know what the handle will be? 
        # On the other hand, using [] for points,
        # makes it possible to go back and change that 
        # later. 
        self._current_spline.append([(x,y), None, None, 'M', None])
        while not self._data.eof() and not self._data.check_next().isalpha():
            last_point = self._get_last_point()
            last_point[2] = None # Update right handle of last point. 
            last_point[4] = 'L' # Outgoing type of last point. 
            # Remaining points are implicit line-to commands. 
            x, y = self._get_next_coord_pair(relative, last_point[0]) # ...but the remaining ones do. 
            self._current_spline.append([(x,y), None, None, 'L', None])

    def _path_line_to(self, command):
        """
        The SVG <path> L, l, H, h, and V, v commands in the d attribute. 
        Draws a line from the current point to the next. 
        """
        # According to the specification, a path must start 
        # with a move-to command. 
        # So there must already be a _current_spline. 
        relative = command.islower() 
        command = command.lower()
        # A line-to can be followed by more than one coordinate pair, 
        # i.e. implicit line-to's. 
        while not self._data.eof() and not self._data.check_next().isalpha():
            last_point = self._get_last_point()
            last_point[2] = None # Right handle of last point. TODO: Is this needed?
            last_point[4] = 'L' # Outgoing type of last point. 
            if command == 'l':
                x, y = self._get_next_coord_pair(relative, last_point[0]) 
                self._current_spline.append([(x,y), None, None, 'L', None])
            elif command == 'h':
                x = self._get_next_coord(relative, last_point[0][0])
                y = last_point[0][1]
                self._current_spline.append([(x,y), None, None, 'L', None])
            elif command == 'v':
                x = last_point[0][0]
                y = self._get_next_coord(relative, last_point[0][1])
                self._current_spline.append([(x,y), None, None, 'L', None])
                # TODO: Needed?
                self._current_spline[-1][2] = None # None means 'VECTOR' type. 
            else: 
                # TODO: Raise exception?
                print('ERROR')

    def _path_curve_to(self, command):
        """
        This handles the SVG <path> commands C, c, S, s, Q, q, T, and t. 
        """
        relative = command.islower()
        command = command.lower()
        while not self._data.eof() and not self._data.check_next().isalpha():
            last_point = self._get_last_point()
            last_point[4] = 'C' # Outgoing type from last point.
            if command == 'c':
                # Update right handle of previous point. 
                previous_handle_x, previous_handle_y = self._get_next_coord_pair(relative, last_point[0])
                last_point[2] = (previous_handle_x, previous_handle_y)
                handle_x, handle_y = self._get_next_coord_pair(relative, last_point[0])
                x, y = self._get_next_coord_pair(relative, last_point[0])
                self._current_spline.append([(x, y), (handle_x, handle_y), None, 'C', None])
            elif command == 's':
                # Update right handle of previous point. 
                handle_x, handle_y = self._get_next_coord_pair(relative, last_point[0]) 
                last_handle_left = last_point[1] 
                last_point_type = last_point[3]
                # TODO: The first check might be redundant.
                if last_handle_left is None or last_point_type != 'C': 
                    last_handle_right = last_point[0]
                else:
                    last_handle_right = self._flip_handle(last_handle_left) # Flip this! 
                last_point[2] = last_handle_right 
                x, y = self._get_next_coord_pair(relative, last_point[0]) 
                self._current_spline.append([(x,y), (handle_x, handle_y), None, 'C', None])
            elif command == 'q':
                q_handle_x, q_handle_y = self._get_next_coord_pair(relative, last_point[0])
                last_handle_right_x = 2 * q_handle_x / 3 + last_point[0][0] / 3
                last_handle_right_y = 2 * q_handle_y / 3 + last_point[0][1] / 3
                last_point[2] = (last_handle_right_x, last_handle_right_y)
                x, y = self._get_next_coord_pair(relative, last_point[0])
                # Creating an equivalent quadratic Bézier curve using a cubic.
                # See https://pomax.github.io/bezierinfo/#reordering 
                handle_x = 2 * q_handle_x / 3 + x / 3
                handle_y = 2 * q_handle_y / 3 + y / 3
                self._current_spline.append([(x,y), (handle_x, handle_y), None, 'C', None])
            elif command == 't':
                # First check if the previous point was really a Q or a T. 
                # We can infer the quadratic Bézier handle of the previous point. 
                # last_handle_left = 2 * q_handle / 3 + last_point / 3 => 
                # q_handle = 3 * last_handle_left / 2 - last_point / 2
                # Now we flip it and convert back to a cubic Bezier handle. 
                # However, this is eqivalent to just flipping it. 
                # So instead, we first flip and then infer what the 
                # new q_handle is and then calculate the left handle of the next point 
                x, y = self._get_next_coord_pair(relative, last_point[0]) # Update so that last_point is automatic. 
                last_point_coord = last_point[0]
                last_handle_left = last_point[1] 
                if last_point[3] == 'C': 
                    last_handle_right = self._flip_handle(last_handle_left) 
                else:
                    last_handle_right = last_point[0]
                last_point[2] = last_handle_right 
                q_handle_x = 3 * last_handle_right[0] / 2 - last_point_coord[0] / 2
                q_handle_y = 3 * last_handle_right[1] / 2 - last_point_coord[1] / 2
                handle_left_x = 2 * q_handle_x /3 + x / 3 
                handle_left_y = 2 * q_handle_y /3 + y / 3 
                self._current_spline.append([(x,y), (handle_left_x, handle_left_y), None, 'C', None])
            # TODO: Make this more consistent. 
            # The _last_point should probably include everything, i.e. 
            # both point and handles since the handles sometimes need to 
            # be updated. 
            # Figure out how to deal with closed curves. 
            # Perhaps a dict for points is not such a bad idea. 
            # Dicts improve readability. 
            # They can also include info about the incoming and outgoing 
            # curve types. 
            # However, that prevents us from reusing path creation code,
            # unless all other classes are also rewritten. 
            # Perhaps 'closed' or 'open' can just be added to the end of the list. 
            """
            point = {'x': x_coord, 
                     'y': y_coord,
                     'handle_left': handle_left_coord,
                     'handle_right': handle_right_coord,
                     'incoming_type': None|Ll|Vv|Hh|Cc|Ss|Qq|Tt|Aa,
                     'outgoing_type': See above}
            Only two handle types relevant: VECTOR and FREE.
            For all straight lines we can use VECTOR. 
            Since these are automatically placed, we just need to 
            make the coordinates for them to be None, to know. 
            If the coordinates are not None, then the handle type should be
            FREE.
            point = [(x,y), (hlx, hly), (hrx, hry), type_l, type_r]
            type_l = incomming curve type
            type_r = outgoing curve type
            """

    def _path_elliptical_arc_to(self, command):
        """
        Parses the SVG <path> A and a commands.
        """
        # A|a rx ry x_axis_rotation large_arc_flag sweep_flag x y
        # This is complicated for two reasons. 
        # First: The arc is a bit hard to infer from the data (with the flags etc).
        # Second: An ellipse can only be approximated with a Bézier curve. 
        # References:
        # For the first problem: https://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
        # For the second problem: http://www.spaceroots.org/documents/ellipse/elliptical-arc.pdf 
        relative = command.islower()
        command = command.lower()
        while not self._data.eof() and not self._data.check_next().isalpha():
            last_point = self._get_last_point() 
            x1, y1 = last_point[0]
            rx, ry = self._get_next_coord_pair(False, None)
            angle = self._get_next_coord(False, None)
            fA = int(self._data.next())     # Flags are best kept as integers. 
            fS = int(self._data.next())
            x2, y2 = self._get_next_coord_pair(relative, last_point[0])
            rx, ry, cx, cy, theta1, dtheta = self._calculate_arc(rx, ry, angle, fA, fS, x1, y1, x2, y2)
            # TODO: Edge-case: dtheta = 0 (in case start and endpoints are the same!) 
            # TODO: Fix case where there should be a straight line, i.e. when 
            # one of rx or ry is zero. 
            dang = dtheta / 4
            angle *= pi / 180
            alpha = sin(dang) * ( sqrt(4 + 3 * tan( dang / 2)**2 ) - 1 ) / 3
            for i in range(1,5):
                last_point = self._get_last_point()
                last_point[4] = 'A' # Update handle. 
            # Ex Ey same as last point. 
                x = cx + rx * cos(angle) * cos(theta1 + i * dang) - ry * sin(angle) * sin(theta1 + i * dang) 
                y = cy + rx * sin(angle) * cos(theta1 + i * dang) + ry * cos(angle) * sin(theta1 + i * dang)
                Epx = - rx * cos(angle) * sin(theta1 + i * dang) - ry * sin(angle) * cos(theta1 + i * dang)
                Epy = - rx * sin(angle) * sin(theta1 + i * dang) + ry * cos(angle) * cos(theta1 + i * dang)
                Eppx = - rx * cos(angle) * sin(theta1 + (i - 1) * dang) - ry * sin(angle) * cos(theta1 + (i - 1) * dang)
                Eppy = - rx * sin(angle) * sin(theta1 + (i - 1) * dang) + ry * cos(angle) * cos(theta1 + (i - 1) * dang)
                last_handle_r_x = last_point[0][0] + alpha * Eppx 
                last_handle_r_y = last_point[0][1] + alpha * Eppy 
                last_point[2] = (last_handle_r_x, last_handle_r_y) 
                next_handle_l_x = x - alpha * Epx 
                next_handle_l_y = y - alpha * Epy
                self._current_spline.append([(x, y), (next_handle_l_x, next_handle_l_y) , None, 'A', None])

    def _calculate_arc(self, rx, ry, angle, fA, fS, x1, y1, x2, y2):
        """
        Helper function for _path_elliptical_arc_to. 
        Calculates the center point parametrization of the elliptical arc.
        """
        # Ref: https://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
        # TODO: Consider using numpy for this. 
        # There is a lot of vector algebra going on. 
        # Many things can be simplified a lot (and perhaps speed up). 
        angle *= pi / 180
        xp = cos(angle) * (x1 - x2) / 2 + sin(angle) * (y1 - y2) / 2 
        yp = - sin(angle) * (x1 - x2) / 2 + cos(angle) * (y1 - y2) / 2 
        # TODO: Fix this! 
        if rx == 0 or ry == 0: 
            print("DRAW A LINE INSTEAD!", "Called from _calculate_arc in SVGPATHParser")
        # TODO: Consider making rx, ry an instance variable. 
        # In that case it does not have to be passed back by 
        # the function. 
        rx, ry = self._correct_radii(rx, ry, xp, yp)
        fac = sqrt(abs(rx**2 * ry**2 - rx**2 * yp**2 - ry**2 * xp**2)) / sqrt(rx**2 * yp**2 + ry**2 * xp**2)
        if fA == fS:
            fac *= -1
        cpx = fac * rx * yp / ry
        cpy = - fac * ry * xp / rx
        cx = cos(angle) * cpx - sin(angle) * cpy + (x1 + x2) / 2
        cy = sin(angle) * cpx + cos(angle) * cpy + (y1 + y2) / 2
        theta_1 = acos( (xp - cpx) / rx / sqrt( (xp - cpx)**2 / rx**2 + (yp - cpy)**2 / ry**2 ))
        if yp < cpy:
            theta_1 *= -1 
        num = ( cpx ** 2 - xp ** 2) / rx ** 2 + (cpy ** 2 - yp ** 2) / ry ** 2 
        denom1 = sqrt( (xp - cpx) ** 2 / rx ** 2 + (yp - cpy) ** 2 / ry ** 2)
        denom2 = sqrt( (xp + cpx) ** 2 / rx ** 2 + (yp + cpy) ** 2 / ry ** 2)
        delta_theta = acos( num / (denom1 * denom2))
        if (xp - cpx) * (-yp - cpy) / (rx * ry) - (yp - cpy) * (-xp - cpx) / (rx * ry) < 0:
            delta_theta *= -1
        if fS == 0 and delta_theta > 0: 
            delta_theta -= 2 * pi 
        if fS == 1 and delta_theta < 0:
            delta_theta += 2 * pi
        return rx, ry, cx, cy, theta_1, delta_theta 

    def _correct_radii(self, rx, ry, xp, yp):
        """
        Helper function to _calculate_arc.
        Corrects the radii in case the ellipse is not large enough to reach
        from one point to the other. In this case, it uniformly scales the 
        ellipse until it exactly touches the two endpoints. 
        Ref: https://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
        """
        # 1. Check that they are nonzero. If rx = 0 or ry = 0, draw a straight line. 
        # 2. Take the absolute value of rx, ry. 
        # 3. Make sure they are large enough. If gamma = xp**2/rx**2 + yp**2/ry**2 <= 1.0 then ok. 
        #    Otherwise rx = sqrt(gamma) * rx, ry = sqrt(gamma) * ry. 
        # 4. Continue with the calculations
        if rx == 0 or ry == 0:
            return None # Set a flag somewhere. 
        rx = abs(rx)
        ry = abs(ry)
        gamma = xp ** 2 / rx ** 2 + yp ** 2 / ry ** 2
        if gamma > 1.0:
            rx *= sqrt(gamma)
            ry *= sqrt(gamma)
        return rx, ry

    def _flip_handle(self, handle):
        """
        Flips an handle around the last point. 

        Used for the smooth versions of the curves. 
        Uses the _last_point from the _current_spline.
        """
        last_point = self._get_last_point()[0]
        h_x = 2 * last_point[0] - handle[0]
        h_y = 2 * last_point[1] - handle[1]
        return (h_x, h_y)

    def _path_close(self, command):
        """
        Closes the path.
        """
        # TODO: Use dict for points and make this nicer.
        self._current_spline.append('closed')


class SVGPATHDataSupplier:
    """
    Supplies the data from the d attribute, one slot at the time. 
    Used by SVGPATHParser.
    """
    __slots__ = ('_data',
                 '_index',
                 '_len')
    
    def __init__(self, d):

        self._data = []
        for entry in re_match_float_or_letter.findall(d):
            # Each entry is either a number (integer or float) or a letter.
            self._data.append(entry)
        self._index = 0
        self._len = len(self._data)

    def eof(self):
        return self._index >= self._len

    def check_next(self):
        if self.eof():
            return None
        else:
            return self._data[self._index] 

    def next(self):
        if self.eof():
            return None
        else:
            self._index += 1
            return self._data[self._index - 1]

    def next_coord(self):
        token = self.next()

        if token is  None:
            return None
        else:
            return float(token)


class SVGGeometrySYMBOL(SVGGeometrySVG):
    """
    Handles the <symbol> element. 
    
    Symbols can only be referenced by a USE element to be rendered. 
    In this case they are replaced with an SVG element internally. 
    Since this is the case, they function exactly as SVG elements. 
    However, the container super class has to check if itself is
    a SYMBOL element. In that case, it will not create any geometry.
    The geometry will only be created when referenced by a USE element. 
    """
    pass


class SVGGeometryDEFS(SVGGeometryContainer):
    """
    Handles the <defs> element. 
    """
    pass


class SVGGeometryUSE(SVGGeometry):
    """
    <use> element.
    """
    __slots__ = ()
    # TODO: Think about where the name for the geometry should come from. 
    # Right now the name comes from the shape, but in case symbol has a separate
    # name and that name is the one actually being used, then that name 
    # should probably be the one that is used. 
    # E.g. if a symbol is named 'hej' and it creates a circle and a rect, 
    # then the circle and the rect should be called hej and hej.001

    def parse(self):
        super().parse()
        # self._viewport = self._parse_viewport() 
        vp_x = self._node.getAttribute('x') or '0'
        vp_y = self._node.getAttribute('y') or '0'
        vp_width = self._node.getAttribute('width') or None
        vp_height = self._node.getAttribute('height') or None

        self._viewport = (vp_x, vp_y, vp_width, vp_height)

    def create_blender_splines(self):
        """
        Create Blender curves objects for the referenced element. 
        """
        # Get the current viewBox. 
        current_viewBox = self._context['current_viewBox']
        # ...and parse the coordinates with respect to the width and height of the vB.  
        x = svg_parse_coord(self._viewport[0], current_viewBox[2])
        y = svg_parse_coord(self._viewport[1], current_viewBox[3])
        # Then add a translation transformation corresponding to the placement
        # attributes x, and y. 
        translation = Matrix.Translation((x,y,0))
        self._push_transform(translation)

        # Push remaining transforms. 
        self._push_transform(self._transform)
        # Push the style onto the stack. 
        self._push_style(self._style)

        # Get a reference to the instance of the referenced geometry. 
        ref = self._node.getAttribute('xlink:href')
        geom = self._context['defs'].get(ref)

        if geom is not None:
            geom_class = geom.__class__
            # If the referenced element is an SVG or a SYMBOL element... 
            if geom_class == SVGGeometrySVG: # SVGGeometrySYMBOL
                #...then save the current viewport of that class...
                old_viewport = geom.get_viewport()
                # ...and replace the width and height with the corresponding 
                #values from the use element.
                # Only replaces the values if they are defined. 
                width = self._viewport[2] or old_viewport[2]
                height = self._viewport[3] or old_viewport[3]
                geom.set_viewport((old_viewport[0], old_viewport[1], width, height))
                geom.create_blender_splines()
                # Reset the old viewport. 
                geom._viewport = old_viewport # TODO: Make a setter for this. 
            elif geom_class == SVGGeometrySYMBOL:
                old_viewport = geom.get_viewport()
                width = self._viewport[2] or '100%'
                height = self._viewport[3] or '100%'
                geom.set_viewport((old_viewport[0], old_viewport[1], width, height))
                geom.create_blender_splines()
                geom._viewport = old_viewport # TODO: Use setter. 
            elif geom_class is SVGGeometryDEFS:
                pass
            else:
                # If it anything else, then we should simply create that geometry.
                # In this case, the width and height attributes are ignored.
                geom.create_blender_splines()

        # Pop styles and transforms. 
        self._pop_style()
        self._pop_transform(self._transform)
        self._pop_transform(translation)


SVG_GEOMETRY_CLASSES = {'svg': SVGGeometrySVG,
                        'g':   SVGGeometryG,
                        'rect': SVGGeometryRECT,
                        'ellipse': SVGGeometryELLIPSE, 
                        'circle': SVGGeometryCIRCLE, 
                        'line': SVGGeometryLINE,
                        'polyline': SVGGeometryPOLYLINE,
                        'polygon': SVGGeometryPOLYGON,
                        'path': SVGGeometryPATH,
                        'symbol': SVGGeometrySYMBOL,
                        'use': SVGGeometryUSE,
                        'defs': SVGGeometryDEFS,
                        }

### End: Classes ###


class SVGLoader(SVGGeometryContainer):
    """
    Parses an SVG file and creates curve objects in Blender.
    """
    # TODO: Fix so that this is done like in the original plugin (e.g. do_colormanage)
    def __init__(self, blender_context, svg_filepath):
        """ 
        Initializes the loader.
        All geometries will be contained by this instance.
        """
        svg_name = os.path.basename(svg_filepath)
        scene = blender_context.scene
        # Create new collection data block in Blender, name from SVG-file.
        collection = bpy.data.collections.new(name=svg_name)
        # Link this to the current scene. 
        scene.collection.children.link(collection) 
        node = xml.dom.minidom.parse(svg_filepath)
        # 96 pixels/inch, 0.3048 meter/feet, 12 inches per feet. 
        scale = 1 / 96 * 0.3048 / 12 
        # SVG y-axis points downwards, but Blender's y-axis points upwards. 
        # So the y-transformation needs a minus sign. 
        m = Matrix()
        m = m @ Matrix.Scale(scale, 4, Vector((1, 0, 0)))
        m = m @ Matrix.Scale(-scale, 4, Vector((0, 1, 0))) 
        # Note: The context is not the same as the Blender context, 
        # but a dictionary to store the stack of transforms, etc, at different
        # elements in the tree. 
        context = {'current_viewBox': (0, 0, 0, 0), # Same as viewBox_stack[-1].
                   'viewport_stack': [(0, 0, 0, 0)], 
                   'viewBox_stack': [(0, 0, 0, 0)], 
                   'current_transform': m, # The full transformation to get from the current node to Blender coordinate space. 
                   'style_stack': [], # Stores the styles of all parents at the time of curve creations.
                   'defs': {}, # Reference to elements by id or class.
                   'blender_collection': collection,
                   'materials': {}, # A dictionary with all defined materials.
                   'do_colormanage': True, # TODO: Calculate this instead by checking if Bl has display device. 
                   'outer_SVG': None, # Keep track of the outermost SVG. Will be used to 
                   # find an overall transformation to move around the origin of the geometry. 
                   }
        super().__init__(node, context)
