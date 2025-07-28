/**
 * @file    module/deexcitation/channel_evaporation.cuh
 * @brief   Evaporation channel
 * @author  CM Lee
 * @date    07/11/2024
 */

#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "device/memory.cuh"
#include "hadron/nucleus.cuh"

#include "auxiliary.cuh"
#include "nuclear_level.cuh"
#include "channel_fission.cuh"


namespace deexcitation {
    namespace evaporation {

        constexpr float SSQR3      = 1.5f * 1.7320508f;  // 1.5 x sqrt(3)

        constexpr float HBAR       = 6.58212e-13f;  // [MeV*ns]
        constexpr float XS_R0      = 1.5e-12f;      // [m]
        constexpr float PROB_COEFF = XS_R0 * XS_R0 / (constants::FP32_TWO_PI * HBAR * HBAR);


        __inline__ __device__ float getAlphaParamHydrogen(int rz) {
            return rz <= 70
                ? 0.1f
                : ((((0.15417e-06f * rz) - 0.29875e-04f) * rz + 0.21071e-02f) * rz - 0.66612e-01f) * rz + 0.98375f;
        }


        __inline__ __device__ float getAlphaParamHelium(int rz) {
            float c;
            if (rz <= 30)
                c = 0.1f;
            else if (rz <= 50)
                c = 0.1f - (rz - 30) * 0.001f;
            else if (rz < 70)
                c = 0.08f - (rz - 50) * 0.001f;
            else
                c = 0.06f;
            return c;
        }


        __inline__ __device__ float getAlphaParamNeutron(int ra) {
            return 0.76f + 2.2f * powf((float)ra, -constants::ONE_OVER_THREE);
        }


        __inline__ __device__ float getBetaParamNeutron(int ra) {
            return (2.12f * powf((float)ra, -constants::TWO_OVER_THREE) - 0.05f) / getAlphaParamNeutron(ra);
        }


        __device__ float getAlphaParam(CHANNEL channel, int rz, int ra);


        __device__ float getBetaParam(CHANNEL channel, int rz, int ra);


        __device__ float emissionProbability(CHANNEL channel, int z, int a, float mass, float exc_energy);


        /**
        * @brief Sample kinetic energy of emitted particle
        * @param state      CUDA XORWOW rand state
        * @param channel    Evaporation channel
        * @param res_z      Atomic number of remnant nucleus
        * @param res_a      Mass number of remnant nucleus
        * @param exc_energy Excitation energy of nucleus [MeV]
        * 
        * @return Kinetic energy of emitted particle [MeV]
        */
        __device__ float emitParticleEnergy(curandState* state, CHANNEL channel, int res_z, int res_a, float exc_energy);


        __device__ void emitParticle(curandState* state, CHANNEL channel);


    }
}