/**
 * @file    module/hadron/auxiliary.cuh
 * @brief   NN collision data
 * @author  CM Lee
 * @date    06/11/2025
 */

#pragma once

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "physics/constants.cuh"


namespace Hadron {


    typedef struct NNScatteringTable {
        int    nenergy[2];
        int    nangle[2];
        float* elab[2];
        float* sig[2];


        __host__ void free();


        __inline__ __device__ float sampleMu(curandState* state, bool isospin, float elab);


    } NNScatteringTable;


#ifdef __CUDACC__


    __device__ float NNScatteringTable::sampleMu(curandState* state, bool isospin, float elab) {

    }


#endif


}