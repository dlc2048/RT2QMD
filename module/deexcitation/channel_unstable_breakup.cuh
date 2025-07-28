/**
 * @file    module/deexcitation/channel_unstable_breakup.cuh
 * @brief   Excessive energy breakup, for unstable & zero excitation energy nuclei
 * @author  CM Lee
 * @date    07/25/2024
 */

#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "device/memory.cuh"
#include "hadron/nucleus.cuh"

#include "auxiliary.cuh"


namespace deexcitation {
    namespace breakup {


        __device__ bool emitParticle(curandState* state);


    }
}