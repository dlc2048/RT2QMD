/**
 * @file    module/qmd/reaction.cuh
 * @brief   QMD reaction (G4QMDReaction.hh)
 * @author  CM Lee
 * @date    02/15/2024
 */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <assert.h>

#include "device/tuning.cuh"
#include "hadron/xs_dev.cuh"


namespace RT2QMD {


    // timer
    extern __constant__ bool           USE_CLOCK;
    extern __device__   long long int* timer;


    __global__ void __kernel__fieldDispatcher(int field_target);


    __host__ cudaError_t __host__fieldDispatcher(int block, int thread, int field_target);


    __global__ void __kernel__pullEligibleField();


    __host__ cudaError_t __host__pullEligibleField(int block, int thread);


    __device__ void prepareModel(int field_target);


    __device__ void returnModel(int field_target);


    __device__ void prepareImpact();


    __global__ void __kernel__prepareModel(int field_target);


    __host__ cudaError_t __host__prepareModel(int block, int thread, int field_target);


    __global__ void __kernel__prepareProjectile();


    __host__ cudaError_t __host__prepareProjectile(int block, int thread);


    __global__ void __kernel__prepareTarget();


    __host__ cudaError_t __host__prepareTarget(int block, int thread);


    __global__ void __kernel__propagate();


    __host__ cudaError_t __host__propagate(int block, int thread);


    __global__ void __kernel__finalize(int field_target);


    __host__ cudaError_t __host__finalize(int block, int thread, int field_target);


    __host__ void __host__deviceResetModelBuffer(int block, int thread, bool return_data);


    __global__ void __device__deviceResetModelBuffer(bool return_data);


    __host__ cudaError_t setPtrTimer(bool use_clock, long long int* clock_ptr);


#ifndef NDEBUG


    __global__ void __kernel__testG4QMDSampleSystem();


    __host__ void testG4QMDSampleMeanFieldSystem(int block, int thread);


#endif


}