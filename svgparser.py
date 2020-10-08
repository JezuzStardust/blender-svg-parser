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
import math
from math import pi, tan
import os
BLENDER = True # Debug flag 
### Reading Coordinates ###

# For 96 dpi:  
# 1 in = 96 px # 1 cm = 96 / 2.54 px # 1 mm = 96 / 25.4 px # 1 pt = 1 / 72 in = 96 / 72 px = 1.33... px # 1 pc = 16 px # The em and ex are relative to the font-size if present.  # If support is added for font-sizes, we should base em and ex on those 
# instead.  # E.g. if font-size="150" is used, then 1 em = 150 px. 
# Em units. Equivalent to the computed font-size in effect for an element.
# Ex units. Equivalent to the height of a lower-case letter in the font (and font-size) in effect for an element. If the font doesn’t include lower-case letters, or doesn’t include the metadata about the ex-height, then 1ex = 0.5em.
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

# RE-breakdown
# (-?\d+(\.\d*)?([eE][-+]?\d+)?) 
# Optional minus sign
# One or more digits
# Optional group: . followed by zero or more digits. 
# Optional group e or E followed by optional sign followed by one or more digits. 
# The optional pattern after | is for the cases where the integer part is not present. 
match_number_part = r'(-?\d+(\.\d*)?([eE][-+]?\d+)?)|(-?\.\d+([eE][-+]?\d+)?)' 
re_match_number_part = re.compile(match_number_part)  
# E.g. for '-1.232e+2cm' the match group(0) will be '-1.232e+2' and end(0) 
# will be 9, the first index after the numerical part. 

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
    match = re_match_number_part.match(text_part) 

    if match is None:
        raise Exception('Invalide float value near ' + text[start_index:start_index + 10])
        
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

### End: Reading Coordinate ###


### Constants ###
# Put this in the end. 

SVG_EMPTY_STYLE = {'fill': None, # TODO: Initialize to black.
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

    m = Matrix(((a, c, e, 0),
                (b, d, f, 0),
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
                 '_material', # This can be baked into context. 
                 '_context')


    def __init__(self, node, context):
        """
        Initialize the base class. 
        Has a reference to the node, the context (transformations, styles, stack).
        """
        self._node = node
        self._transform = Matrix()
        self._style = SVG_EMPTY_STYLE
        self._context = context


    def parse(self):
        """
        Parses the style and transformations on the node. 
        Some nodes do not have style, fill, stroke, etc, or transform.
        However, in the _parse_transformation we always check first 
        if these attributes actually exists first. 
        """
        # TODO: Move calls for transformation parsing and style parsing to
        # classes that actually can have these attributes. 
        # But keep the functions in this class since they are shared between 
        # many different classes with that function. 
        
        if type(self._node) is xml.dom.minidom.Element:
            self._style = self._parse_style()
            self._transform = self._parse_transform()


    def _parse_style(self):
        """
        Parse the style attributes (e.g. fill="black", stroke-width=".1cm", etc) 
        first, then parse the style-attribute (e.g. style="fill:blue;stroke:green.."
        In SVG-files style="fill:blue;..." takes precedence over e.g. fill="blue".
        """
        
        for attr in SVG_EMPTY_STYLE.keys():
            val = self._node.getAttribute(attr) 
            if val:
                self._style[attr] = val.strip().lower()


        style = self._node.getAttribute('style')
        if style:
            elems = style.split(';')
            for elem in elems:
                s = elem.split(':')

                if len(s) != 2:
                    continue

                name = s[0].strip().lower()
                val = s[1].strip()

                if name in SVG_EMPTY_STYLE.keys():
                    self._style[name] = val

        # TODO: Initialize the empty style to have a default fill of black. 
        # Only if fill="none" or style="...;fill:none;..." is specified 
        # we should ignore the fill. 


    def _parse_transform(self):
        """
        Parse the transform attribute on the node. 
        Will only be called on DOM Elements.
        """
        
        transform = self._node.getAttribute('transform')

        m = Matrix() 

        if transform:
            # RE-breakdown. 
            # Zero or more spaces \s*
            # Followed by one or more letters ([A-z]+), first capture group
            # Followed by zero or more spaces \s*
            # Followed by left parenthesis \(
            # Followed by one or more (as few as possible) characters, *? means lazy, second capture group
            # Followed by right parenthesis 
            pattern = r'\s*([A-z]+)\s*\((.*?)\)' 
            matcher = re.compile(pattern)

            for match in matcher.finditer(transform):
                trans = match.group(1)
                params = match.group(2)
                params = params.replace(',', ' ').split()

                transform_function = SVG_TRANSFORMS.get(trans)
                if transform_function is None: 
                    raise Exception('Unknown transform function: ' + trans) 

                m = m @ transform_function(params) 

        return m


    def _parse_preserveAspectRatio(self):
        # TODO: Move to specific class. 
        # TODO: Handle cases where it starts with 'defer' (and ignore this case).
        # TODO: Handle 'none'.

        # Can exist in SVG and SYMBOL. Move to container. 

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


    def _view_to_transform(self): # Remove pAR, add self. 
        """
        Parses the viewBox into equivalent transformations.
        """
        # TODO: Figure out what happens for % variables. 
        # TODO: Figure out what happens for nested viewports and viewBoxes. 

        viewBox = self._viewBox
        viewport = self._viewport
        preserveAspectRatio = self._preserveAspectRatio

        current_viewBox = self._context['current_viewBox'] # Parent's viewBox

        # First parse the rect resolving against the current viewBox. 
        # The parse the viewBox. In case there is no viewBox, 
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

        # TODO: Handle 'none'. 
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


    def _inherit_viewBox_from_viewport(self):
        """
        Inherit the viewBox from viewport, i.e. use standard coordinates.
        Used when there is not viewBox present. 
        """
        current_viewBox = self._context['current_viewBox']
        viewport = self._viewport
        viewBox_width = svg_parse_coord(viewport[2], current_viewBox[2])
        viewBox_height = svg_parse_coord(viewport[3], current_viewBox[3])

        return (0, 0, viewBox_width, viewBox_height)


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


    def _push_viewBox(self, viewBox):
        """
        """
        if viewBox:
            self._context['current_viewBox'] = viewBox
            self._context['viewBox_stack'].append(viewBox)

    
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


    def _new_blender_curve(self, name, is_cyclic):
        """
        Create new curve object in Blender. 
        """
        # Create Blender curve object. 
        curve = bpy.data.curves.new(name, 'CURVE')
        obj = bpy.data.objects.new(name, curve)
        self._context['blender_collection'].objects.link(obj)
        cu = obj.data 

        cu.dimensions = '2D'
        cu.fill_mode = 'BOTH'
        #cu.materials.append(...)

        cu.splines.new('BEZIER')

        spline = cu.splines[-1]
        spline.use_cyclic_u = is_cyclic

        return spline


    def _add_points_to_blender(self, coords, spline):
        """
        Adds points from coords to the spline. 

        coords = list of coordinates (point, handle_left, handle_right).
        spline = a reference to bpy.objects[...].data.splines[-1].
        """
        # TODO: It is better to create the spline within the loop (for point...). 
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


class SVGGeometryContainer(SVGGeometry):
    """
    Container class for SVGGeometry. 
    Since a container has attributes such as style, and transformations, 
    it inherits from SVGGeometry. 
    """
    __slots__ = ('_geometries')


    def __init__(self, node, context): 
        """
        Inits the container. 
        """

        super().__init__(node, context) 
        self._geometries = []


    def parse(self):

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
        Create 
        """

        for geo in self._geometries:
            geo.create_blender_splines()


class SVGGeometrySVG(SVGGeometryContainer):
    """
    Corresponds to the <svg> elements. 
    """
    
    __slots__ = ('_viewBox',
                 '_preserveAspectRatio',
                 '_viewport')


    def parse(self):

        self._viewport = self._parse_viewport()
        self._viewBox = self._parse_viewBox()
        self._preserveAspectRatio = self._parse_preserveAspectRatio()

        super().parse()


    def create_blender_splines(self):
        """
        Adds geometry to Blender.
        """

        if self._viewBox:
            viewBox = self._viewBox
        else:
            viewBox = self._inherit_viewBox_from_viewport()

        viewport_transform = self._view_to_transform()
        self._push_transform(viewport_transform)
        self._push_transform(self._transform)
        self._push_viewBox(viewBox)
        super().create_blender_splines()
        self._pop_viewBox()
        self._pop_transform(self._transform)
        self._pop_transform(viewport_transform) 


    def _parse_viewport(self):
        """
        Parse the x, y, width, and height attributes. 
        """
        # TODO: Move this to SVGGeometryContainer or SVGGeometry.
        # Can be used by SVG, USE, and perhaps (says so in SVG 2.0) SYMBOL. 
        # Does not work with SYMBOL in Inkscape. 
        # x, y, width and/or height of SYMBOL might be overridden by USE. 
        # Must return it and store it somewhere. 

        vp_x = x = self._node.getAttribute('x') or '0'
        vp_y = self._node.getAttribute('y') or '0'

        vp_width = self._node.getAttribute('width') or '100%'
        vp_height = self._node.getAttribute('height') or '100%'

        return (vp_x, vp_y, vp_width, vp_height)


    def _parse_viewBox(self):
        """
        Parse the viewBox attribute. 
        """
        # TODO: Move this to SVGGeometryContainer, since this will be 
        # used by SVG and SYMBOL which are both containers. 

        view_box = self._node.getAttribute('viewBox')
        if view_box: 
            vb_min_x, vb_min_y, vb_width, vb_height = view_box.replace(',', ' ').split()
            return (vb_min_x, vb_min_y, vb_width, vb_height)
        else: 
            return None


class SVGGeometryG(SVGGeometryContainer):
    pass


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
        factor_x = rx * (math.sqrt(7) - 1)/3
        factor_y = ry * (math.sqrt(7) - 1)/3

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

        name = self._node.getAttribute('id') or self._node.getAttribute('class') 
        if not name:
            name = 'Rect'

        spline = self._new_blender_curve(name, True) 

        self._push_transform(self._transform)

        self._add_points_to_blender(coords, spline)

        self._pop_transform(self._transform)


class SVGGeometryELLIPSE(SVGGeometry):
    """
    SVG <ellipse>. 
    """
    __slots__ = ('_cx',
                 '_cy',
                 '_rx',
                 '_ry',
                 '_is_circle')
                 

    def __init__(self, node, context, is_circle = False):
        """ 
        Initialize the ellipse with default values (all zero). 
        """

        super().__init__(node, context)

        self._is_circle = is_circle
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
        # TODO: Can this be made more nice?

        vB = self._context['current_viewBox'][2:] # width and height of viewBox.

        cx = svg_parse_coord(self._cx, vB[0])
        cy = svg_parse_coord(self._cy, vB[1])

        if self._is_circle:
            weighted_diagonal = math.sqrt(float(vB[0]) ** 2 + float(vB[1]) ** 2)/math.sqrt(2)
            rx = ry = svg_parse_coord(self._rx, weighted_diagonal)  
        else:
            rx = svg_parse_coord(self._rx, vB[0])
            ry = svg_parse_coord(self._ry, vB[1]) 

        # Approximation of elliptic curve for corner.
        # Put the handles semi minor(or major) axis radius times 
        # factor = (sqrt(7) - 1)/3 away from Bezier point.
        # http://www.spaceroots.org/documents/ellipse/elliptical-arc.pdf
        factor_x = rx * (math.sqrt(7) - 1)/3
        factor_y = ry * (math.sqrt(7) - 1)/3

        # Coordinate, first handle, second handle
        coords = [((cx - rx, cy), (cx - rx, cy + factor_y), (cx - rx, cy - factor_y)),
                  ((cx, cy - ry), (cx - factor_x, cy - ry), (cx + factor_x, cy - ry)),
                  ((cx + rx, cy), (cx + rx, cy - factor_y), (cx + rx, cy + factor_y)),
                  ((cx, cy + ry), (cx + factor_x, cy + ry), (cx - factor_x, cy + ry))]

        name = self._node.getAttribute('id') or self._node.getAttribute('class')
        if not name:
            if self._is_circle:
                name = 'Circle'
            else:
                name = 'Ellipse'

        spline = self._new_blender_curve(name, True)

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
        # TODO: Can this be made more nice?

        vB = self._context['current_viewBox'][2:] # width and height of viewBox.

        x1 = svg_parse_coord(self._x1, vB[0])
        y1 = svg_parse_coord(self._y1, vB[1])
        x2 = svg_parse_coord(self._x2, vB[0])
        y2 = svg_parse_coord(self._y2, vB[1])

        coords = [((x1, y1), None, None), ((x2, y2), None, None)]

        name = self._node.getAttribute('id') or self._node.getAttribute('class') 
        if not name:
            name = 'Line'

        spline = self._new_blender_curve(name, False)

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
        points = self._node.getAttribute('points')

        # TODO: Check if this should be done in a separate function. 
        match_number = r'([+-]?(\d+(\.\d*)?|(\.\d+))([eE][+-]?\d+)?)'

        number_matcher = re.compile(match_number) 

        previous = None
        
        # TODO: If also useful for path: 
        # append ((previous, float(p[0])), None, None)
        # In this way, we can reuse _add_points_to_blender. 
        for p in number_matcher.findall(points):
            if previous is None: # Skips last number if number of points is odd.
                previous = float(p[0])
            else: 
                self._points.append((previous, float(p[0])))
                previous = None


    def create_blender_splines(self):
        """
        Creates the splines in Blender. 
        """


        name = self._node.getAttribute('id') or self._node.getAttribute('class')

        if not name:
            if self._is_closed: 
                name = 'Polygon'
            else:
                name = 'Polyline'

        spline = self._new_blender_curve(name, self._is_closed)

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


SVG_GEOMETRY_CLASSES = {'svg': SVGGeometrySVG,
                        'g':   SVGGeometryG,
                        'rect': SVGGeometryRECT,
                        'ellipse': SVGGeometryELLIPSE, 
                        'circle': SVGGeometryCIRCLE, 
                        'line': SVGGeometryLINE,
                        'polyline': SVGGeometryPOLYLINE,
                        'polygon': SVGGeometryPOLYGON,
                        }

### End: Classes ###

class SVGLoader(SVGGeometryContainer):
    """
    Parses an SVG file and creates curve objects in Blender.
    """

    # TODO: Fix so that this is done like in the original plugin (e.g. do_colormanage)
    def __init__(self, blender_context, svg_filepath):
        """ 
        Inits the loader.
        All geometries will be contained by this instance and the containers it contains. 
        """

        # BLENDER = False # Debug flag. 
        if BLENDER: 
            svg_name = os.path.basename(svg_filepath)
            scene = blender_context.scene
            # Create new collection data block in Blender, name
            # from SVG-file.
            collection = bpy.data.collections.new(name=svg_name)
            # Link this to the current scene. 
            scene.collection.children.link(collection) 
        else:  
            collection = None

        node = xml.dom.minidom.parse(svg_filepath)
        
        # 96 pixels/inch, 0.3048 meter/feet, 12 inches per feet. 
        scale = 1 / 96 * 0.3048 / 12 

        # SVG y-axis points downwards, but Blender's y-axis points upwards. 
        m = Matrix()
        m = m @ Matrix.Scale(scale, 4, Vector((1, 0, 0)))
        m = m @ Matrix.Scale(-scale, 4, Vector((0, 1, 0))) 
        
        context = {'current_viewport': (0, 0), # Not used so far. 
                   'current_viewBox': (0, 0, 0, 0), # Same as viewBox_stack[-1].
                   'viewport_stack': [(0, 0, 0, 0)], # Not used so far.  
                   'viewBox_stack': [(0, 0, 0, 0)], # Used. 
                   'current_transform': m,       # Used
                   'current_style': SVG_EMPTY_STYLE, # Will be used. Probably also a stack is needed. This stack should have a special push method that considers the previous 
                   'defs': {}, # DEFS (and maybe SYMBOL) references. 
                   'blender_collection': collection # Used.
                  }

        super().__init__(node, context)
