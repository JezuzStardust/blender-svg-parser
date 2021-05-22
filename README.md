# Blender SVG-Parser

This is a slightly improved SVG parser based on SVG importer addon from Blender.
The original source code can be found e.g. [here](https://github.com/sobotka/blender-addons/tree/master/io_curve_svg). 


**Work in progress.** 

## How to test it
1. Make a folder e.g. `~/Blender-Scipts/modules/svgparser` and clone the project there.
2. In Blender Preferences choose `File Paths` and set the `Scripts` to the path above.
3. Open a Scripting tab and enter e.g.
```python
import bpy
import svgparser.svgparser 
ld = svgparser.svgparser.SVGLoader(bpy.context, <path-to-svg-file>)
ld.parse()
ld.create_blender_splines()
```
where `<path-to-svg-file>` is the absolut path to an SVG file on your computer. 
If you want to edit the source, you need to make sure that Blender always uses 
the latest version. In that case, add
```python
import importlib
importlib.reload(svgparser.svgparser)
```

## Current work: 
Currently I am working on a module for Bézier curves. This will handle all the calculations for e.g. strokes.
To test this, you can add a poly-Bézier curve (add a Bézier curve and extrude one of the ends a few times) and then do the following:
```python
import svgparser.bezier as sb
c = sb.spline_from_Blender('BezierCurve')
c.stroke()
```
This will add a stroke around the path. The encaps of this will depend on what I was working on when the code was cloned, but it can manually be changed in the code (in the Spline.stroke method).

You can also find intersections between curves, and curve self intersections. These will be used later when I will be trying to figure out how to reduce an intersecting stroke shape so that it does not overlap with itself. Blender tends to handle the filling of such curves in a strange way. 


## Comparisons
![tiger.svg comparison](https://github.com/JezuzStardust/blender-svg-parser/blob/main/Comparisons/Tiger%20SVG%20comparison.png)
Left: File imported with standard SVG importer. Right: Imported with this plugin. Each new curve is automatically offset slightly in the z-direction. This simulates the drawing order of the SVG file. (Note, this feature is currently turned off. I will consider adding it back later.)
Original SVG-file: [tiger.svg](https://commons.wikimedia.org/wiki/File:Ghostscript_Tiger.svg)

## Current status
- [x] Handles `<svg>` and nested `<svg>`.
- [x] Handles basic shapes: `<rect>`, `<circle>`, `<ellipse>`, `<polyline>`, and `<polygon>`. 
- [x] Handles `<path>`. 
- [x] `<g>` 
- [x] `<symbol>`
- [x] `<defs>`
- [x] `<use>`
- [x] Color handling
- [ ] Extensive testing.
- [ ] To handle style and strokes: Convert non-zero stroke-widths to two 'parallel' paths (similar to Inkscape's "Stroke to path". 
- [ ] Make this into an actual Blender plug-in that can be installed. 
- [ ] Add settings for choosing scale on import.
- [ ] Add settings for choosing dpi on import.
- [ ] Add settings for choosing where origin is positioned, e.g. upper/mid/lower left/mid/right.
- [ ] Make the imported geometry offset in the z-direction according to the drawing order. E.g. offset by .001 B.U. in the z direction everytime a new object is drawn.
- [ ] Possibly: Make it possible to import as grease pencil object instead. Or a mixed object (curves for the closed shapes and grease pencil for all strokes). 
- [ ] Possibly: Make geometry clip outside of main SVG viewport. 
- [ ] Possibly: Add support for clipping paths. 
- [ ] Possibly: Add support for text. 
- [ ] Possibly: Add support for gradients. 
- [ ] Possibly: Add support for transparency.
- [ ] Possibly: Add support for some filters. 
- [ ] Possibly: Add support for embedded images (might be hard to extract). But perhaps not: https://gist.github.com/jeromerobert/ff34f504acd7feb0306a 


## Improvements/changes
* Uses a different (perhaps improved) way of approximating elliptic curves for the rounded corners of `<rect>`. Based on [this](http://www.spaceroots.org/documents/ellipse/elliptical-arc.pdf).
* Handles `viewBox` attributes more consistently with the SVG specification. 
* Handles nested SVG elements according to the specification. 
* Takes into account the `preserveAspectRatio`.
* Scales the radius of a circle correctly when given in relative terms (percentages). 
* The imported image will have the correct size (e.g. a 2cm-by-2cm SVG image will produce a 2cm-by-2cm "image" in Blender)
* Handles smooth quadratic and cubic Bézier curves according the specification. Specifically, a smooth cubic curve (S or s) should only be smooth in case the previous curve is either smooth (S or s) or a cubic Bézier (C or c). 
* Changed so that the default is 96 dpi, but I am planning to make this into a separate setting.
* Handles percentages in color specified as rgb.
* Sets the color both as a default color and also in the node tree. In the standard plugin you loose the color if you switch to using nodes. 
* Handles both the style attribute and a standalone fill attribute. 

## References 
Below is a list of references and what I have used them for. 
1. [Drawing an elliptical arc using polylines, quadratic or cubic Bézier curves by L. Maisonobe](http://www.spaceroots.org/documents/ellipse/elliptical-arc.pdf) - Approximation of an elliptical arc using cubic Bézier curves. 
2. [The SVG 1.1 recommendation](https://www.w3.org/TR/SVG11/Overview.html)
3. [The SVG 2.0 candidate recommendation](https://www.w3.org/TR/SVG2/Overview.html) - In particular Appendix B.2 for handling of elliptical arcs, and 8.2 for converting a viewport and viewBox into an equivalent transform.
4. [A Primer on Bézier Curves by "Pomax"](https://pomax.github.io/bezierinfo/) - How to get a quadratic curve using a cubic curve. Also includes information on how to approximate parallel Bézier curves. 
