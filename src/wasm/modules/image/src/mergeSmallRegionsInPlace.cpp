#include "mergeSmallRegionsInPlace.h"
#include "floodFillLabel.h"
#include <queue>

struct Pixel {
  int x, y;
};

// Helper to get 1D index from x,y
static inline int idx(int x, int y, int width) { return y * width + x; }

// TODO: check for gaps inside regions - its possible their dimensions are fine,
//          but inner gaps reduce effective width and height and distort quadrilateral area
std::pair<std::vector<int>, std::vector<Region>>  mergeSmallRegionsInPlace(uint8_t *pixels, int width, int height,
                              int minArea, int minWidth, int minHeight) {
  auto [labels, regions] = floodFillLabel(pixels, width, height);

  // Merge small regions
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int l = labels[idx(x, y, width)];
      if (regions[l].isBigEnough(minArea, minWidth, minHeight))
        continue;

      // Check immediate neighbors
      // TODO: make sure pixels on edges are not compared to non-neighbors
      //        e.g., pixel at [width, 5] must not be compared to pixel at [0, 6] (it exists on the next line of pixels)
      int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
      for (auto &d : dirs) {
        int nx = x + d[0], ny = y + d[1];
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          int nl = labels[idx(nx, ny, width)];
          if (nl != l && regions[nl].isBigEnough(minArea, minWidth, minHeight)) {
            // Copy color
            for (int c = 0; c < 4; c++) {
              pixels[idx(x, y, width) * 4 + c] = pixels[idx(nx, ny, width) * 4 + c];
            }

            // update label
            labels[idx(x, y, width)] = nl;

            // update regions (don't update small region metadata - it will be filtered out in next step)
            regions[nl].add(x, y);
            regions[l].size--;
            break;
          }
        }
      }
    }
  }

  std::vector<Region> existingRegions;
  for (Region &r : regions) {
    if (r.size > 0) {
      existingRegions.push_back(r);
    }
  }

  return std::make_pair(labels, existingRegions);
}
