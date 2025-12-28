#ifndef IMAGETOSVG_H
#define IMAGETOSVG_H

#include "exported.h"
#include <string>
#include <cstdint>

/// Convert an RGBA pixel buffer (row-major) into an SVG string preserving region shapes.
/// - pixels: pointer to width*height*4 bytes (RGBA)
/// - width, height: image dimensions
/// - minArea: optional; components smaller than minArea are ignored in the output
EXPORTED void imageToSVG(const uint8_t* pixels, int width, int height, int minArea,
                              char* outSvg, int outSvgSizeLimit);

#endif // IMAGETOSVG_H
