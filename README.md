# Blender SVG-Parser

This is a slightly imporved SVG parser based on SVG importer addon from Blender.
The original source code can be found e.g. [here](https://github.com/sobotka/blender-addons/tree/master/io_curve_svg). 

**Work in progress.** 

## Current status:
* Can import SVG files which contains only nested `<svg>`, `<rect>`, `<circle>`, and `<ellipse>`. 

## Improvements/changes:
* Uses a different (perhaps improved) way of approximating elliptic curves for the rounded corners of `<rect>`.
* Handles `viewBox` attributes more consistently, e.g. for nested SVG-elements. 
* Takes into account the `preserveAspectRatio` (but does not yet handle 'none').
* Scales the radius of a circle correctly when given in relative terms (percentages). 
* The imported image will have the correct size (e.g. a 2cm-by-2cm SVG image will produce a 2cm-by-2cm "image" in Blender)
* Changed so that the default is 96 dpi. 

## Planned work:
* Add so that it handles remaining geometry objects (path, line, polyline, etc).
* Add handling of `<defs>`, `<g>` and `<symbol>`.
* Add handling of style attributes. 
* Long term goal: Convert non-zero stroke-widths to two parallel curves (similar to "Object to path/Stroke to path" in Inkscape.)
* Add possibility of automatically scale image on import. 
* Add settings for different dpi. 
* Make it into an actual Blender addon that can be installed. 
* Add comparisons images between this, the orignal plugin and Firefox renderings.
* Possibly, take into account clipping paths and clipping of objects partly outside the `viewBox`.
