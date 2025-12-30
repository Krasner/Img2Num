#include "floodFillLabel.h"

struct Pixel { int x, y; };

// Helper to get 1D index from x,y
static inline int idx(int x, int y, int width) { return y * width + x; }

// Compare two RGBA image
static inline bool sameColor(const uint8_t* img, int w, int h, int x1, int y1, int x2, int y2) {
    int i1 = idx(x1, y1, w) * 4;
    int i2 = idx(x2, y2, w) * 4;
    return img[i1] == img[i2] && img[i1 + 1] == img[i2 + 1] &&
           img[i1 + 2] == img[i2 + 2] && img[i1 + 3] == img[i2 + 3];
}

std::pair<std::vector<int>, std::vector<Region>>
floodFillLabel(const uint8_t* const image, const int width, const int height) {
  std::vector<int> labels(width * height, -1);
  std::vector<Region> regions;
  int nextLabel = 0;

  // Flood-fill labeling
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (labels[idx(x, y, width)] != -1)
        continue;

      Region r;
      std::queue<Pixel> q;
      q.push({x, y});
      labels[idx(x, y, width)] = nextLabel;
      r.add(x, y, image, width);

      while (!q.empty()) {
        Pixel p = q.front();
        q.pop();
        int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (auto &d : dirs) {
          int nx = p.x + d[0], ny = p.y + d[1];
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            if (labels[idx(nx, ny, width)] == -1 &&
                sameColor(image, width, height, p.x, p.y, nx, ny)) {
              labels[idx(nx, ny, width)] = nextLabel;
              r.add(nx, ny, image, width);
              q.push({nx, ny});
            }
          }
        }
      }

      regions.push_back(r);
      nextLabel++;
    }
  }

  return std::make_pair(labels, regions);
}
