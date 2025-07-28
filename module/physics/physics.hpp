#pragma once

#include <cmath>

#include <cuda_runtime.h>

#include "constants.hpp"


namespace physics {

    double coulombCorrection(int z);  // PEGS4 FCOULCP
    double xsifp(int z);              // PEGS4 XSIF


    inline int getZnumberFromZA(int za) {
        return za / 1000;
    }


    inline int getAnumberFromZA(int za) {
        return za % 1000;
    }


    inline int2 splitZA(int za) {
        return { za / 1000, za % 1000 };
    }


    class DensityEffect {
    private:
        static double _GAS_DENSITY_THRES;
        double        _m;
        double        _c;
        double        _a;
        double2       _x;
    public:
        DensityEffect(
            double mean_ie,
            double density,
            double plasma_frequency
        );
        double get(double energy);
    };

}