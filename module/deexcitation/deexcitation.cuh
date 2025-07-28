/**
 * @file    module/deexcitation/deexcitation.cuh
 * @brief   Deexcitation kernel
 * @author  CM Lee
 * @date    07/15/2024
 */

#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "device/memory.cuh"

#include "auxiliary.cuh"
#include "channel_evaporation.cuh"
#include "channel_fission.cuh"
#include "channel_photon.cuh"
#include "channel_unstable_breakup.cuh"


namespace deexcitation {


    __device__ void boostAndWrite(mcutil::BUFFER_TYPE origin, mcutil::UNION_FLAGS flags, float m0, bool is_secondary);


    __global__ void __kernel__deexcitationStep();
    __host__ void deexcitationStep(int block, int thread);


}