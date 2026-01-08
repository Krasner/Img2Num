#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "slicpp.h"
#include "cielab.h"
#include "exported.h"

static inline int clampi(int x, int lo, int hi) {
    return std::max(lo, std::min(hi, x));
}

static inline float lab_dist2(const Lab& p, const Lab& q) {
    float dL = p.L - q.L;
    float da = p.a - q.a;
    float db = p.b - q.b;
    return dL * dL + da * da + db * db;
}

static std::vector<float> gaussian_kernel_1d(float sigma) {
    // predefined gaussian kernel
    if (sigma <= 0.f) return {1.f};
    int radius = std::max(1, int(std::ceil(3.f * sigma)));
    std::vector<float> k(2 * radius + 1);
    float sum = 0.f;
    float inv2s2 = 1.f / (2.f * sigma * sigma);
    for (int i = -radius; i <= radius; ++i) {
        float v = std::exp(-(float(i * i)) * inv2s2);
        k[size_t(i + radius)] = v;
        sum += v;
    }
    for (float& v : k) v /= sum;
    return k;
}

static void gaussian_blur(const std::vector<float>& src, std::vector<float>& dst, int w, int h, float sigma) {
  auto k = gaussian_kernel_1d(sigma);
  int radius = int((k.size() - 1) / 2);

  std::vector<float> tmp(w * h, 0.f);

  // horizontal
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float acc = 0.f;
      for (int dx = -radius; dx <= radius; ++dx) {
        int xx = clampi(x + dx, 0, w - 1);
        acc += src[size_t(y) * w + xx] * k[size_t(dx + radius)];
      }
      tmp[size_t(y) * w + x] = acc;
    }
  }

  // vertical
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float acc = 0.f;
      for (int dy = -radius; dy <= radius; ++dy) {
        int yy = clampi(y + dy, 0, h - 1);
        acc += tmp[size_t(yy) * w + x] * k[size_t(dy + radius)];
      }
      dst[size_t(y) * w + x] = acc;
    }
  }
}

// -----------------------------------------------------------------------------
// SLIC++
// -----------------------------------------------------------------------------

SLICPP::SLICPP(
    int desiredK, 
    float m, 
    int iters, 
    float sigma/*= 1.0f*/, 
    float gamma/*= 1e-3f*/, 
    float nu/*= 1.0f*/, 
    bool use_diagonal/* = true*/
) : SLIC(desiredK, m, iters),
    sigma_(sigma),
    gamma_(gamma),
    nu_(nu),
    use_diagonal_(use_diagonal) 
{}

SLICPP::~SLICPP(){}

// Returns label image (size w*h), labels are contiguous after connectivity.
//std::vector<int> SLICPP::segment(const ImageRGB& img, const Params& p) {
void SLICPP::segment(const uint8_t* image, const size_t height, const size_t width, std::vector<int>& labels, int& num_labels){
    
    std::cout << "SLICPP segment" << std::endl;

    const int N = (int)height * (int)width;

    // short hand for index
    auto indexing = [width](int x, int y) { return y * width + x; };

    // Convert to Lab
    std::vector<Lab> cie_image;
    cie_image.resize(width * height);
    for (int y{0}; y < height; y++) {
        for (int x{0}; x < width; x++) {
            int index = (y * static_cast<int>(width) + x);
            int center_idx{index* 4};
            uint8_t r0{image[center_idx]};
            uint8_t g0{image[center_idx + 1]};
            uint8_t b0{image[center_idx + 2]};
            uint8_t a0{image[center_idx + 3]};
            double L0, A0, B0;
            rgb_to_lab(r0, g0, b0, L0, A0, B0);

            cie_image[index] = Lab{.L=L0, .a=A0, .b=B0};
        }
    }

    // Precompute gradient magnitude (Eq. 11-like) on Lab (edge strength)
    std::vector<float> grad(N, 0.f);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int xm1 = clampi(x - 1, 0, int(width) - 1);
        int xp1 = clampi(x + 1, 0, int(width) - 1);
        int ym1 = clampi(y - 1, 0, int(height) - 1);
        int yp1 = clampi(y + 1, 0, int(height) - 1);

        const Lab& Lxp = cie_image[size_t(indexing(xp1, y))];
        const Lab& Lxm = cie_image[size_t(indexing(xm1, y))];
        const Lab& Lyp = cie_image[size_t(indexing(x, yp1))];
        const Lab& Lym = cie_image[size_t(indexing(x, ym1))];

        float dx2 = lab_dist2(Lxp, Lxm);
        float dy2 = lab_dist2(Lyp, Lym);
        float g = std::sqrt(std::max(0.f, dx2 + dy2));
        grad[size_t(indexing(x, y))] = g;
      }
    }

    // Smoothed gradient magnitude (G_sigma * ||∇I||)
    std::vector<float> grad_blur(N, 0.f); 
    gaussian_blur(grad, grad_blur, width, height, sigma_);

    // Density D(x) = exp(E(x)/nu), E(x) = g / (g_blur + gamma)
    std::vector<float> density(N, 1.f);
    for (int i = 0; i < N; ++i) {
      float g = grad[size_t(i)];
      float gb = grad_blur[size_t(i)];
      float E = g / (gb + gamma_);
      E = std::min(E, 20.f); // avoid overflow
      density[size_t(i)] = std::exp(E / std::max(1e-6f, nu_));
    }

    // Compute S (grid interval)
    float S = std::sqrt(float(N) / float(K_));
    if (S < 1.f) S = 1.f;

    // Initialize centers on grid; move to local minimum gradient in 3x3 (SLIC)
    std::vector<Center> centers;
    centers.reserve(size_t(K_));

    int step = int(std::round(S));
    if (step < 1) step = 1;

    for (int y = step / 2; y < height; y += step) {
      for (int x = step / 2; x < width; x += step) {
        int bestx = x, besty = y;
        float bestg = grad[size_t(indexing(x, y))];

        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            int xx = clampi(x + dx, 0, int(width) - 1);
            int yy = clampi(y + dy, 0, int(height) - 1);
            float gg = grad[size_t(indexing(xx, yy))];
            if (gg < bestg) {
              bestg = gg;
              bestx = xx;
              besty = yy;
            }
          }
        }

        int ii = indexing(bestx, besty);
        const Lab& P = cie_image[size_t(ii)];
        centers.push_back(Center{P.L, P.a, P.b, float(bestx), float(besty)});
      }
    }

    const int K_eff = int(centers.size());
    if (K_eff == 0) throw std::runtime_error("No centers initialized; image too small?");

    // label + distance buffers
    // std::vector<int> labels(N, -1);
    labels.resize(N);
    std::vector<float> bestD(N, std::numeric_limits<float>::infinity());

    // Dijkstra helper (local window)
    struct PQItem {
      float dist;
      int lx, ly; // local coordinates in window
      bool operator<(const PQItem& o) const { return dist > o.dist; } // min-heap
    };

    auto dijkstra_window = [&](int cx, int cy, int x0, int y0, int x1, int y1,
                               std::vector<float>& geo_spatial,
                               std::vector<float>& geo_color) {
      int ww = x1 - x0 + 1;
      int hh = y1 - y0 + 1;
      int sz = ww * hh;
      const float INF = std::numeric_limits<float>::infinity();
      geo_spatial.assign(size_t(sz), INF);
      geo_color.assign(size_t(sz), INF);

      auto lidx = [ww](int x, int y) { return y * ww + x; };

      int scx = cx - x0;
      int scy = cy - y0;
      if (scx < 0 || scx >= ww || scy < 0 || scy >= hh) return;

      std::priority_queue<PQItem> pq;
      geo_spatial[size_t(lidx(scx, scy))] = 0.f;
      geo_color[size_t(lidx(scx, scy))] = 0.f;
      pq.push(PQItem{0.f, scx, scy});

      const std::array<int, 8> dx8{{1, -1, 0, 0, 1, 1, -1, -1}};
      const std::array<int, 8> dy8{{0, 0, 1, -1, 1, -1, 1, -1}};
      int nbh = use_diagonal_ ? 8 : 4;

      while (!pq.empty()) {
        PQItem cur = pq.top();
        pq.pop();
        int x = cur.lx;
        int y = cur.ly;
        int li = lidx(x, y);
        float dcur = geo_spatial[size_t(li)];
        if (cur.dist != dcur) continue;

        int gx = x0 + x;
        int gy = y0 + y;
        int gi = indexing(gx, gy);

        for (int k = 0; k < nbh; ++k) {
          int nx = x + dx8[size_t(k)];
          int ny = y + dy8[size_t(k)];
          if (nx < 0 || nx >= ww || ny < 0 || ny >= hh) continue;

          int ngx = x0 + nx;
          int ngy = y0 + ny;
          int ngi = indexing(ngx, ngy);

          float step_len = (k < 4) ? 1.f : 1.41421356237f;

          // Spatial geodesic edge weight: density-weighted step length
          float w_sp = 0.5f * (density[size_t(gi)] + density[size_t(ngi)]) * step_len;

          // "Geodesic color" accumulation (practical approximation):
          // accumulate Lab neighbor-to-neighbor difference along the shortest (w_sp) path.
          float w_col = std::sqrt(lab_dist2(cie_image[size_t(gi)], cie_image[size_t(ngi)]));

          float nd = dcur + w_sp;
          float nc = geo_color[size_t(li)] + w_col;

          int nli = lidx(nx, ny);
          float& best_sp = geo_spatial[size_t(nli)];
          float& best_col = geo_color[size_t(nli)];

          // Standard Dijkstra update + tie-break by lower color cost
          if (nd < best_sp - 1e-6f || (std::abs(nd - best_sp) <= 1e-6f && nc < best_col)) {
            best_sp = nd;
            best_col = nc;
            pq.push(PQItem{nd, nx, ny});
          }
        }
      }
    };

    // Main iterations (SLIC loop)
    std::vector<float> geo_spatial;
    std::vector<float> geo_color;

    for (int it = 0; it < iters_; ++it) {
      std::fill(bestD.begin(), bestD.end(), std::numeric_limits<float>::infinity());
      std::fill(labels.begin(), labels.end(), -1);

      // Assignment
      for (int k = 0; k < K_eff; ++k) {
        const Center& C = centers[size_t(k)];
        int cx = clampi(int(std::round(C.x)), 0, int(width) - 1);
        int cy = clampi(int(std::round(C.y)), 0, int(height) - 1);

        int rad = int(std::round(S)); // window half-size ~ S (=> ~2S x 2S)
        int x0 = std::max(0, cx - rad);
        int x1 = std::min(int(width) - 1, cx + rad);
        int y0 = std::max(0, cy - rad);
        int y1 = std::min(int(height) - 1, cy + rad);

        // Geodesic distances from center in local window
        dijkstra_window(cx, cy, x0, y0, x1, y1, geo_spatial, geo_color);

        int ww = x1 - x0 + 1;

        for (int y = y0; y <= y1; ++y) {
          for (int x = x0; x <= x1; ++x) {
            int i = indexing(x, y);
            const Lab& P = cie_image[size_t(i)];

            // Euclidean SLIC terms
            float dc = std::sqrt((P.L - C.L) * (P.L - C.L) +
                                 (P.a - C.a) * (P.a - C.a) +
                                 (P.b - C.b) * (P.b - C.b));
            float ds = std::sqrt((x - C.x) * (x - C.x) + (y - C.y) * (y - C.y));

            // Geodesic terms
            int lx = x - x0;
            int ly = y - y0;
            int li = ly * ww + lx;
            float d4 = geo_spatial[size_t(li)];
            float d3 = geo_color[size_t(li)];

            // Fallback (shouldn't happen)
            if (!std::isfinite(d4) || !std::isfinite(d3)) {
              d4 = ds;
              d3 = dc;
            }

            float De = dc + (m_ / S) * ds;      // Eq. 3 style
            float Dg = d3 + (m_ / S) * d4;      // geodesic analogue
            float D  = w1_ * De + w2_ * Dg;    // Eq. 10 style

            if (D < bestD[size_t(i)]) {
              bestD[size_t(i)] = D;
              labels[size_t(i)] = k;
            }
          }
        }
      }

      // Update centers
      std::vector<double> sumL(size_t(K_eff), 0.0);
      std::vector<double> suma(size_t(K_eff), 0.0);
      std::vector<double> sumb(size_t(K_eff), 0.0);
      std::vector<double> sumx(size_t(K_eff), 0.0);
      std::vector<double> sumy(size_t(K_eff), 0.0);
      std::vector<int> count(size_t(K_eff), 0);

      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          int i = indexing(x, y);
          int k = labels[size_t(i)];
          if (k < 0) continue;
          const Lab& P = cie_image[size_t(i)];
          sumL[size_t(k)] += P.L;
          suma[size_t(k)] += P.a;
          sumb[size_t(k)] += P.b;
          sumx[size_t(k)] += x;
          sumy[size_t(k)] += y;
          count[size_t(k)] += 1;
        }
      }

      float max_shift = 0.f;
      for (int k = 0; k < K_eff; ++k) {
        if (count[size_t(k)] == 0) continue;
        Center old = centers[size_t(k)];
        double inv = 1.0 / double(count[size_t(k)]);
        centers[size_t(k)].L = float(sumL[size_t(k)] * inv);
        centers[size_t(k)].a = float(suma[size_t(k)] * inv);
        centers[size_t(k)].b = float(sumb[size_t(k)] * inv);
        centers[size_t(k)].x = float(sumx[size_t(k)] * inv);
        centers[size_t(k)].y = float(sumy[size_t(k)] * inv);

        float dx = centers[size_t(k)].x - old.x;
        float dy = centers[size_t(k)].y - old.y;
        max_shift = std::max(max_shift, std::sqrt(dx * dx + dy * dy));
      }

      // Early stop
      if (max_shift < 0.5f) break;
    }

    // Connectivity enforcement (standard SLIC post-processing)
    // enforce_connectivity(labels, lab, W, H, p.K);
    enforceConnectivity_(labels, num_labels, width, height, S);
}

EXPORTED void slicpp_segmentation(uint8_t* image, size_t width, size_t height) {
    SLICPP slic(400, 5.0, 20);
    std::vector<int> labels;
    int num_labels = 0;
    slic.segment(image, height, width, labels, num_labels);
    std::cout << "Number of segments: " << num_labels << std::endl;
    slic.meanColorImage(image, height, width, labels, num_labels);
}