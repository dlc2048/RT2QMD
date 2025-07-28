/**
 * @file    module/deexcitation/channel_photon.cuh
 * @brief   Photon evaporation channel
 * @author  CM Lee
 * @date    07/09/2024
 */

#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "device/memory.cuh"
#include "hadron/nucleus.cuh"

#include "auxiliary.cuh"
#include "nuclear_level.cuh"
#include "utils.cuh"


namespace deexcitation {
    namespace photon {


        constexpr float GDRE_FACTOR = 2.5f;

        // photo evaporation
        constexpr int   PHOTON_EVAP_MAX_DE_POINT = 10;
        constexpr float PHOTON_EVAP_NORM = (float)(
            1.25 / constants::MILLIBARN / (constants::FP64_PI_SQ * constants::HBARC * constants::HBARC));


        /**
        * @brief Calculate the empirical parameter of the giant diple resonance
        * @param a Mass number of nuclei
        *
        * @return GDR parameter [MeV]
        */
        __inline__ __device__ float GDREnergy(int a);


        __inline__ __device__ float GDRWidth(int a);


        __device__ float emissionProbability(int z, int a, float mass, float exc_energy, bool store_shared_mem);


        __device__ float sampleContinuumEnergy(curandState* state, uchar4 zaev, float mass, float exc_energy);


        __device__ void boostAndWriteGamma(float m0);


        __global__ void __kernel__continuumEvaporationStep();
        __host__ void continuumEvaporationStep(int block, int thread);


#ifdef __CUDACC__


        __device__ float GDREnergy(int a) {
            return 40.3f * powf((float)a, -0.2f);
        }


        __device__ float GDRWidth(int a) {
            return 0.3f * GDREnergy(a);
        }


#endif


    }
}