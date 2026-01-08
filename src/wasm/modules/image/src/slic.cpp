#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <string>
#include <vector>
#include "slic.h"
#include "cielab.h"
#include "exported.h"


SLIC::SLIC(
    int desiredK, 
    float m, 
    int iters
) : K_(std::max(1, desiredK)), m_(std::max(0.0f, m)), iters_(std::max(1, iters)) {}

SLIC::~SLIC() {}

void SLIC::segment(const uint8_t* image, const size_t height, const size_t width, std::vector<int>& labels, int& num_labels) {

    const int N = (int)height * (int)width;

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

    // S ~ sqrt(N / K)
    const float S_f = std::sqrt(float(N) / float(K_));
    const int S = std::max(1, int(std::lround(S_f)));

    // Initialize centers on grid; optionally shift to lowest gradient in 3x3
    std::vector<SLIC::Center> centers = initCenters_(cie_image, width, height, S);
    num_labels = (int)centers.size();
    if (num_labels <= 0) {
        labels.assign(size_t(N), 0);
        num_labels = 1;
        return;
    }

    labels.assign(size_t(N), -1);
    std::vector<float> dist(size_t(N), std::numeric_limits<float>::max());

    const float invS2 = 1.0f / (S_f * S_f);
    const float m2 = m_ * m_;

    for (int iter = 0; iter < iters_; ++iter) {
        std::fill(dist.begin(), dist.end(), std::numeric_limits<float>::max());

        // Assignment: only visit pixels in 2S x 2S neighborhood per center
        for (int ci = 0; ci < num_labels; ++ci) {
            const SLIC::Center& c = centers[ci];

            const int x0 = std::max(0,   (int)std::floor(c.x - S_f));
            const int x1 = std::min((int)width-1, (int)std::ceil (c.x + S_f));
            const int y0 = std::max(0,   (int)std::floor(c.y - S_f));
            const int y1 = std::min((int)height-1, (int)std::ceil (c.y + S_f));

            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    const int idx = y * width + x;
                    const Lab& p = cie_image[idx];

                    const double dL = p.L - c.L;
                    const double da = p.a - c.a;
                    const double db = p.b - c.b;
                    const double dc2 = dL*dL + da*da + db*db;

                    const double dx = x - c.x;
                    const double dy = y - c.y;
                    const double ds2 = dx*dx + dy*dy;

                    // D^2 = dc^2 + (m/S)^2 * ds^2
                    const double D2 = dc2 + (m2 * invS2) * ds2;

                    if (D2 < dist[idx]) {
                        dist[idx] = D2;
                        labels[idx] = ci;
                    }
                }
            }
        }

        // Update centers
        std::vector<double> sumL(num_labels, 0.0), suma(num_labels, 0.0), sumb(num_labels, 0.0);
        std::vector<double> sumx(num_labels, 0.0), sumy(num_labels, 0.0);
        std::vector<double> count(num_labels, 0.0);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const int idx = y * width + x;
                const int id = labels[idx];
                if (id < 0) continue;

                const Lab& p = cie_image[idx];
                sumL[id] += p.L;  suma[id] += p.a;  sumb[id] += p.b;
                sumx[id] += x;    sumy[id] += y;
                count[id] += 1.0;
            }
        }

        for (int i = 0; i < num_labels; ++i) {
            if (count[i] <= 0.0) continue;
            centers[i].L = sumL[i] / count[i];
            centers[i].a = suma[i] / count[i];
            centers[i].b = sumb[i] / count[i];
            centers[i].x = sumx[i] / count[i];
            centers[i].y = sumy[i] / count[i];
        }
    }

    // Enforce connectivity
    enforceConnectivity_(labels, num_labels, width, height, S);

    // Optional: relabel to [0..num_labels-1] already guaranteed by connectivity stage.
}

void SLIC::meanColorImage(uint8_t* image, const size_t height, const size_t width, const std::vector<int>& labels, int num_labels) {
    
    std::vector<uint64_t> sumR(num_labels, 0), sumG(num_labels, 0), sumB(num_labels, 0);
    std::vector<uint32_t> cnt (num_labels, 0);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int index = y * width + x;
            const int id = labels[index];
            if (id < 0 || id >= num_labels) continue;
            int center_idx{index * 4};
            sumR[id] += image[center_idx];
            sumG[id] += image[center_idx + 1];
            sumB[id] += image[center_idx + 2];
            cnt[id]  += 1;
        }
    }

    std::vector<std::array<uint8_t,3>> mean(num_labels, {0,0,0});
    for (int i = 0; i < num_labels; ++i) {
        if (cnt[i] == 0) continue;
        mean[i][0] = (uint8_t)std::clamp<uint64_t>(sumR[i] / cnt[i], 0, 255);
        mean[i][1] = (uint8_t)std::clamp<uint64_t>(sumG[i] / cnt[i], 0, 255);
        mean[i][2] = (uint8_t)std::clamp<uint64_t>(sumB[i] / cnt[i], 0, 255);
    }

    std::vector<uint8_t> result(width * height * 4);

    for (int y{0}; y < height; y++) {
        for (int x{0}; x < width; x++) {
            int index = (y * static_cast<int>(width) + x);
            const int id = labels[index];
            const auto c = (id >= 0 && id < num_labels) ? mean[id] : std::array<uint8_t,3>{0,0,0};

            int center_idx{index * 4};
            result[center_idx] = c[0];
            result[center_idx + 1] = c[1];
            result[center_idx + 2] = c[2];
            result[center_idx + 3] = image[center_idx + 3];
        }
    }

    std::memcpy(image, result.data(), result.size());
}
    
std::vector<SLIC::Center> SLIC::initCenters_(const std::vector<Lab>& lab, int W, int H, int S) {
    std::vector<SLIC::Center> centers;
    if (W <= 0 || H <= 0) return centers;

    const bool canGradient = (W >= 3 && H >= 3);

    for (int y = S/2; y < H; y += S) {
        for (int x = S/2; x < W; x += S) {
            int bestx = x;
            int besty = y;

            if (canGradient) {
                const int cx = std::clamp(x, 1, W-2);
                const int cy = std::clamp(y, 1, H-2);

                double bestg = std::numeric_limits<float>::max();
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = std::clamp(cx + dx, 1, W-2);
                        const int ny = std::clamp(cy + dy, 1, H-2);
                        const double g = gradL2_(lab, W, H, nx, ny);
                        if (g < bestg) {
                            bestg = g;
                            bestx = nx;
                            besty = ny;
                        }
                    }
                }
            }

            const Lab& v = lab[besty*W + bestx];
            centers.push_back(SLIC::Center{v.L, v.a, v.b, (double)bestx, (double)besty});
        }
    }

    return centers;
}

void SLIC::enforceConnectivity_(std::vector<int>& labels, int& num_labels, int W, int H, int S) {
    const int N = W * H;
    if ((int)labels.size() != N) throw std::runtime_error("enforceConnectivity_: label size mismatch");

    std::vector<int> newLabels(size_t(N), -1);

    // Expected superpixel area ~ S^2; merge anything smaller than ~ S^2/4
    const int expectedSize = std::max(1, S * S);
    const int minSize = std::max(1, expectedSize / 4);

    const int dx4[4] = { 1, -1, 0, 0 };
    const int dy4[4] = { 0, 0, 1, -1 };

    std::vector<int> component;
    component.reserve(size_t(expectedSize) * 2);

    int nextLabel = 0;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int start = y*W + x;
            if (newLabels[start] != -1) continue;

            const int oldLabel = labels[start];

            std::queue<int> q;
            q.push(start);
            newLabels[start] = nextLabel;
            component.clear();
            component.push_back(start);

            int adjacent = -1; // merge target if component too small

            while (!q.empty()) {
                const int idx = q.front();
                q.pop();

                const int cx = idx % W;
                const int cy = idx / W;

                for (int k = 0; k < 4; ++k) {
                    const int nx = cx + dx4[k];
                    const int ny = cy + dy4[k];
                    if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

                    const int nidx = ny*W + nx;

                    if (newLabels[nidx] != -1) {
                        if (newLabels[nidx] != nextLabel) adjacent = newLabels[nidx];
                        continue;
                    }

                    if (labels[nidx] == oldLabel) {
                        newLabels[nidx] = nextLabel;
                        q.push(nidx);
                        component.push_back(nidx);
                    }
                }
            }

            if ((int)component.size() < minSize && adjacent != -1) {
                for (int idx : component) newLabels[idx] = adjacent;
                // do not increment nextLabel
            } else {
                ++nextLabel;
            }
        }
    }

    labels.swap(newLabels);
    num_labels = nextLabel;
}

EXPORTED void slic_segmentation(uint8_t* image, size_t width, size_t height) {
    SLIC slic(400, 5.0, 20);
    std::vector<int> labels;
    int num_labels = 0;
    slic.segment(image, height, width, labels, num_labels);
    std::cout << "Number of segments: " << num_labels << std::endl;
    slic.meanColorImage(image, height, width, labels, num_labels);
}