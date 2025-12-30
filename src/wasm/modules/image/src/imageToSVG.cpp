#include "imageToSVG.h"
#include "mergeSmallRegionsInPlace.h"
#include "image_utils.h"
#include "kmeans.h"
#include <vector>
#include <queue>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cstring>
#include <cmath>
#include <emscripten/console.h>

static inline int idx(int x, int y, int w) { return y * w + x; }

/// Trace outer contour of a connected component labeled with `lab` using Moore neighbor tracing.
/// Returns vector of (x,y) points in order. If it fails, returns empty vector.
static std::vector<std::pair<int,int>> traceContour(const std::vector<int>& labels, int lab, int width, int height) {
  // 8-neighbor offsets in clockwise order (starting West to match Moore style)
  const int dx[8] = {-1,-1,0,1,1,1,0,-1};
  const int dy[8] = {0,-1,-1,-1,0,1,1,1};

  // Find start pixel: topmost-leftmost pixel of the label that touches background
  int sx=-1, sy=-1;
  for (int y = 0; y < height && sx == -1; ++y) {
    for (int x = 0; x < width; ++x) {
      if (labels[idx(x,y,width)] != lab) continue;
      // check 8-neighbor if any neighbor is not same label or out of bounds
      bool boundary = false;
      for (int k = 0; k < 8; ++k) {
        int nx = x + dx[k], ny = y + dy[k];
        if (nx < 0 || nx >= width || ny < 0 || ny >= height || labels[idx(nx,ny,width)] != lab) {
          boundary = true;
          break;
        }
      }
      if (boundary) { sx = x; sy = y; break; }
    }
  }

  if (sx == -1) return {}; // no boundary found

  std::vector<std::pair<int,int>> contour;
  contour.emplace_back(sx, sy);

  int px = sx, py = sy;
  int bx = sx + 1, by = sy; // previous pixel (outside the region)
  int iterLimit = width * height * 10; // safety cap
  int iter = 0;

  auto neighborIndex = [&](int nx, int ny)->int {
    for (int i = 0; i < 8; ++i) if (px + ( (i==0||i==1||i==7)?-1: (i==3||i==4||i==5)?1:0 ) /*not used*/ == nx && py == ny) return i;
    // The simple neighborIndex used earlier assumed px/py; we'll fallback below if needed
    for (int i = 0; i < 8; ++i) if (px + ( (i==0||i==1||i==7)?-1: (i==3||i==4||i==5)?1:0 ) == nx && py == ny) return i;
    return -1;
  };

  // We'll implement Moore tracing loop similarly to earlier approach but without relying on neighborIndex heavily:
  // Use the canonical 8 neighbors relative to current p
  const int ndx[8] = {-1,-1,0,1,1,1,0,-1};
  const int ndy[8] = {0,-1,-1,-1,0,1,1,1};

  // for the first iteration set previous b to east (sx+1,sy) so we start search from W side
  bx = sx + 1; by = sy;

  while (iter++ < iterLimit) {
    // find index of b relative to current p
    int bIdx = -1;
    for (int i = 0; i < 8; ++i) {
      if (px + ndx[i] == bx && py + ndy[i] == by) { bIdx = i; break; }
    }
    int startK = (bIdx == -1) ? 0 : ((bIdx + 1) % 8);

    int foundK = -1;
    int foundX = 0, foundY = 0;
    for (int t = 0; t < 8; ++t) {
      int k = (startK + t) % 8;
      int nx = px + ndx[k], ny = py + ndy[k];
      if (nx >= 0 && nx < width && ny >= 0 && ny < height && labels[idx(nx,ny,width)] == lab) {
        foundK = k;
        foundX = nx; foundY = ny;
        break;
      }
    }

    if (foundK == -1) {
      // isolated pixel — stop
      break;
    }

    // move: set b = previous p, p = found
    bx = px; by = py;
    px = foundX; py = foundY;

    if (px == sx && py == sy && contour.size() > 1) {
      // closed
      break;
    }
    contour.emplace_back(px, py);
  }

  return contour;
}

void imageToSVG(const uint8_t* pixels, int width, int height, int minArea,
                char* outSvg, int outSvgSizeLimit) {
  if (!pixels || width <= 0 || height <= 0 || !outSvg || outSvgSizeLimit <= 0) {
    if (outSvg && outSvgSizeLimit > 0) outSvg[0] = '\0';
    return;
  }

  const int area{width * height};

  // Make a modifiable copy of the pixel buffer so we can call mergeSmallRegionsInPlace.
  // If minArea <= 0 we skip merging and use the original buffer.
  std::vector<uint8_t> modPixels{pixels, pixels + (area * 4)};

  gaussian_blur_fft(modPixels.data(), width, height, width * 0.005);

  black_threshold_image(modPixels.data(), width, height, 8);

  kmeans_clustering(modPixels.data(), width, height, 8, 100);

  // choose minWidth/minHeight as sqrt(minArea) (at least 1)
  int minDimension = std::max(1, (int)std::floor(std::sqrt((double)std::max(1, minArea))));
  // Call merge: this mutates modPixels in-place
  auto [labels, regions] = mergeSmallRegionsInPlace(modPixels.data(), width, height, minArea, minDimension, minDimension);
  const uint8_t* srcPixels{modPixels.data()};

  // Build SVG
  std::ostringstream ss;
  ss << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
  ss << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
     << "width=\"" << width << "\" height=\"" << height << "\" "
     << "viewBox=\"0 0 " << width << " " << height << "\" "
     << "shape-rendering=\"crispEdges\">\n";

  // For each region produce a path from its contour
  for (int lab = 0; lab < regions.size(); ++lab) {
    if (regions[lab].size < minArea) continue;

    auto contour = traceContour(labels, lab, width, height);
    if (contour.empty()) {
      // fallback: produce rect bounding box scanned from pixels if tracing failed
      int minX = width, minY = height, maxX = 0, maxY = 0;
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          if (labels[idx(x,y,width)] == lab) {
            minX = std::min(minX, x);
            minY = std::min(minY, y);
            maxX = std::max(maxX, x);
            maxY = std::max(maxY, y);
          }
        }
      }
      if (minX <= maxX && minY <= maxY) {
        const Color &c = regions[lab].color;
        std::string hex = colorToHex(c);
        double opacity = (double)c.a / 255.0;
        ss << "<rect x=\"" << minX << "\" y=\"" << minY
           << "\" width=\"" << (maxX - minX + 1) << "\" height=\"" << (maxY - minY + 1)
           << "\" fill=\"" << hex << "\" fill-opacity=\"" << std::fixed << std::setprecision(3) << opacity << "\" />\n";
      }
      continue;
    }

    // Build path "M x y L x y ... Z"
    std::ostringstream path;
    for (size_t i = 0; i < contour.size(); ++i) {
      int x = contour[i].first;
      int y = contour[i].second;
      if (i == 0) path << "M " << x << " " << y;
      else path << " L " << x << " " << y;
    }
    path << " Z";

    const Color &c = regions[lab].color;
    std::string hex = colorToHex(c);
    double opacity = (double)c.a / 255.0;
    ss << "<path d=\"" << path.str() << "\" fill=\"" << hex << "\"";
    if (c.a < 255) ss << " fill-opacity=\"" << std::fixed << std::setprecision(3) << opacity << "\"";
    ss << " stroke=\"none\" />\n";
  }

  ss << "</svg>\n";
  std::string svgString{ss.str()};
  emscripten_console_log(svgString.c_str());
  size_t len = svgString.size();
  if ((int)len > outSvgSizeLimit) {
    // Not enough space
    len = outSvgSizeLimit - 1; // leave space for null terminator
    emscripten_console_warn("SVG truncated because buffer is too small!");
  }

  std::memcpy(outSvg, svgString.c_str(), len);
  outSvg[len] = '\0'; // null-terminate
}
