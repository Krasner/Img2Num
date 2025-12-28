#ifndef REGION_H
#define REGION_H

#include <climits>

struct Region {
  int size = 0;
  int minX = INT_MAX, maxX = INT_MIN;
  int minY = INT_MAX, maxY = INT_MIN;

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

  int width() const { return maxX - minX + 1; }
  int height() const { return maxY - minY + 1; }

  bool isBigEnough(int minArea, int minWidth, int minHeight) const {
    return size >= minArea && width() >= minWidth && height() >= minHeight;
  }
};

#endif // REGION_H
