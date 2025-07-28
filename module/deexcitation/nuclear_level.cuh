/**
 * @file    module/deexcitation/nuclear_level.cuh
 * @brief   Deexcitation nuclear level data
 * @author  CM Lee
 * @date    07/09/2024
 */

#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "device/memory.cuh"
#include "physics/constants.cuh"

#include "utils.cuh"


namespace deexcitation {


    constexpr float FIXED_LEVEL_DENSITY = 0.075f;  // Simple level density [1/MeV]


    extern __constant__ bool USE_SIMPLE_LEVEL_DENSITY;


    __inline__ __device__ float getLevelDensityParameter(int a) {
        if (USE_SIMPLE_LEVEL_DENSITY)
            return (float)a * FIXED_LEVEL_DENSITY;
        else
            return 0.058025f * (float)a * (1.f + 5.9059f * powf((float)a, -constants::ONE_OVER_THREE));  // Mengoni, A., & Nakajima, Y. (1994)
    }


    __inline__ __device__ float getFissionLevelDensityParameter(int z, int a) {
        float ldp = getLevelDensityParameter(a);
        if (z >= 89)
            ldp *= 1.05f;
        else if (z <= 85)
            ldp *= 1.03f;
        else
            ldp *= 1.03f + 0.005f * (float)(z - 85);
        return ldp;
    }


    __host__ cudaError_t setSimpleLevelDensity(bool use_it);


}