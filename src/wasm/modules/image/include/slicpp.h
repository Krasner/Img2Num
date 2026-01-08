#ifndef SLICPP_H
#define SLICPP_H

#include "slic.h"

class SLICPP: public SLIC {
    public:
        SLICPP(
            int desiredK, 
            float m, 
            int iters, 
            float sigma = 1.0f, 
            float gamma = 1e-3f, 
            float nu = 1.0f, 
            bool use_diagonal = true
        );
        ~SLICPP();

        void segment(const uint8_t* image, const size_t height, const size_t width, std::vector<int>& labels, int& num_labels);

    private:
        
        float w1_ = 0.3175f;     // Euclidean weight
        float w2_ = 0.6825f;     // Geodesic weight

        // Density parameters (edge-aware geodesic weighting)
        float sigma_;     // Gaussian smoothing sigma for gradient magnitude
        float gamma_;    // small constant to suppress weak edges
        float nu_;        // scaling in exp(E/nu)

        bool use_diagonal_; // 8-connected geodesic (true) vs 4-connected (false)

};

#endif