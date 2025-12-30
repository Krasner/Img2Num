#ifndef COLOR_H
#define COLOR_H

#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>

struct Color {
  uint8_t r, g, b, a;
};

static Color getColor(const uint8_t* pixels, int w, int x, int y) {
  int i = (y * w + x) * 4;
  return { pixels[i], pixels[i+1], pixels[i+2], pixels[i+3] };
}

static inline bool sameColor(const uint8_t* pixels, int w, int x1, int y1, int x2, int y2) {
  Color c1 = getColor(pixels, w, x1, y1);
  Color c2 = getColor(pixels, w, x2, y2);
  return c1.r == c2.r && c1.g == c2.g && c1.b == c2.b && c1.a == c2.a;
}

/// Convert RGBA to hex string like "#rrggbb"
static std::string colorToHex(const Color &c) {
  std::ostringstream ss;
  ss << '#' << std::hex << std::setfill('0')
     << std::setw(2) << (int)c.r
     << std::setw(2) << (int)c.g
     << std::setw(2) << (int)c.b;
  return ss.str();
}

#endif // COLOR_H
