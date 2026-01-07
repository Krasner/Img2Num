#ifndef SLIC_H
#define SLIC_H

#include <cstdint>

struct Lab {
    double L, a, b;
};

class SLIC {
    public:
        SLIC(int desiredK, float m, int iters);
        ~SLIC();

        void segment(const uint8_t* image, const size_t height, const size_t width, std::vector<int>& labels, int& num_labels);
        void meanColorImage(uint8_t* image, const size_t height, const size_t width, const std::vector<int>& labels, int num_labels);


    private:
        struct Center {
            double L, a, b;
            double x, y;
        };

        static inline double gradL2_(const std::vector<Lab>& lab, int W, int H, int x, int y) {
            // Caller ensures 1 <= x <= W-2 and 1 <= y <= H-2
            const double Lxm1 = lab[y*W + (x-1)].L;
            const double Lxp1 = lab[y*W + (x+1)].L;
            const double Lym1 = lab[(y-1)*W + x].L;
            const double Lyp1 = lab[(y+1)*W + x].L;

            const double dx = Lxp1 - Lxm1;
            const double dy = Lyp1 - Lym1;
            return dx*dx + dy*dy;
        }

        int K_;
        float m_;
        int iters_;

        std::vector<Center> initCenters_(const std::vector<Lab>& lab, int W, int H, int S);
        void enforceConnectivity_(std::vector<int>& labels, int& num_labels, int W, int H, int S);
};

#endif