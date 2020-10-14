# Blender SVG-Parser

This is a slightly imporved SVG parser based on SVG importer addon from Blender.
The original source code can be found e.g. [here](https://github.com/sobotka/blender-addons/tree/master/io_curve_svg). 

**Work in progress.** 

## Current status:
- [x] Handles `<svg>` and nested `<svg>`.
- [x] Handles basic shapes: `<rect>`, `<circle>`, `<ellipse>`, `<polyline>`, and `<polygon>`. 
- [x] Handles `<path>`. 
- [ ] `<g>` 
- [ ] `<symbol>`
- [ ] `<defs>`
- [ ] `<use>`
- [ ] Color handling
- [ ] Style and line-width.

## Improvements/changes:
* Uses a different (perhaps improved) way of approximating elliptic curves for the rounded corners of `<rect>`. Based on [this](http://www.spaceroots.org/documents/ellipse/elliptical-arc.pdf).
* Handles `viewBox` attributes more consistently, e.g. for nested SVG-elements. 
* Takes into account the `preserveAspectRatio` (but does not yet handle 'none').
* Scales the radius of a circle correctly when given in relative terms (percentages). 
* The imported image will have the correct size (e.g. a 2cm-by-2cm SVG image will produce a 2cm-by-2cm "image" in Blender)
* Changed so that the default is 96 dpi. 

## Planned work:
* Add handling of `<defs>`, `<g>`, `<symbol>`, and `<use>`.
* Add handling of style attributes and colors.
* Long term goal: Convert non-zero stroke-widths to two parallel curves (similar to "Object to path/Stroke to path" in Inkscape.)
* Add settings for choosing scale image on import. 
* Add settings for different dpi. 
* Make it into an actual Blender addon that can be installed. 
* Add comparisons images between this, the orignal addon and Firefox renderings.
* Possibly, take into account clipping paths and clipping of objects partly outside the `viewBox`.

## References 
The implementation is based on the following references:
1. http://www.spaceroots.org/documents/ellipse/elliptical-arc.pdf - Approximation of an elliptical arc using cubic Bézier curves. 
2. https://www.w3.org/TR/SVG11/Overview.html - The SVG 1.1 Recommendation. 
3. https://www.w3.org/TR/SVG2/Overview.html - The SVG 2.0 Candidate Recommendation. In particular Appendix B.2 for handling of elliptical arcs, and 8.2 for converting a viewport and viewBox into an equivalent transform.
