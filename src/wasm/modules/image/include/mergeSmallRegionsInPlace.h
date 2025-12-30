#ifndef MERGESMALLREGIONSINPLACE_H
#define MERGESMALLREGIONSINPLACE_H

#include "exported.h" // EXPORTED macro
#include "Region.h"
#include <cstdint>
#include <utility>
#include <vector>

// pixels: RGBA uint8_t* buffer
// width, height: dimensions
// minArea: minimum area to preserve (smaller regions get merged)
std::pair<std::vector<int>, std::vector<Region>> mergeSmallRegionsInPlace(uint8_t *pixels, int width, int height, int minArea, int minWidth, int minHeight);
#endif
