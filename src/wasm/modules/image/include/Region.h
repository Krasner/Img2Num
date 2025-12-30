#ifndef REGION_H
#define REGION_H

#include "Color.h"
#include <cstdint>
#include <climits>

struct Region {
  int size = 0;
  int minX = INT_MAX, maxX = INT_MIN;
  int minY = INT_MAX, maxY = INT_MIN;

  Color color;
  bool colorSet = false;

  void add(int x, int y) {
    size++;
    if (x < minX)
      minX = x;
    if (x > maxX)
      maxX = x;
    if (y < minY)
      minY = y;
    if (y > maxY)
      maxY = y;
  }

  void add(int x, int y, const uint8_t* image, int width) {
    if (!colorSet) {
      color = getColor(image, width, x, y);
      colorSet = true;
    }
    add(x, y);
  }

  int width() const { return maxX - minX + 1; }
  int height() const { return maxY - minY + 1; }

  bool isBigEnough(int minArea, int minWidth, int minHeight) const {
    return size >= minArea && width() >= minWidth && height() >= minHeight;
  }
};

#endif // REGION_H
