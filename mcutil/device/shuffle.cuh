//
// Copyright (C) 2025 CM Lee, SJ Ye, Seoul Sational University
//
// Licensed to the Apache Software Foundation(ASF) under one
// or more contributor license agreements.See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// 	"License"); you may not use this file except in compliance
// 	with the License.You may obtain a copy of the License at
// 
// 	http ://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.See the License for the
// specific language governing permissionsand limitations
// under the License.

/**
 * @file    mcutil/device/shuffle.cuh
 * @brief   warp-level shuffle algorithm for the 
 *          reduction, sort, random permutation, and cumulative sum
 * @author  CM Lee
 * @date    03/13/2024
 */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>


constexpr int CUDA_WARP_SHIFT = 5;
constexpr int CUDA_WARP_SIZE  = 1 << CUDA_WARP_SHIFT;


#ifdef __CUDACC__


__device__ __forceinline__ int pow2_ceil_(int x) {
    if (!x) return 1;

    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;

    return x;
}


__device__ __forceinline__ unsigned int bfe_(unsigned int i, unsigned int k) {
    unsigned int y;
    asm(" bfe.u32 %0, %1, %2, %3;"
        : "=r"(y) : "r" (i), "r" (k), "r" (0x01u));
    return y;
}


__device__ __forceinline__ float swap_(float x, int mask, int dir) {
    float y = __shfl_xor_sync(0xffffffff, x, mask);
    return x < y == dir ? y : x;
}


__device__ __forceinline__ int swap_(int x, int mask, int dir) {
    int y = __shfl_xor_sync(0xffffffff, x, mask);
    return x < y == dir ? y : x;
}


__device__ __forceinline__ double rsum64_(double x) {
    x += __shfl_xor_sync(0xffffffff, x, 16);
    x += __shfl_xor_sync(0xffffffff, x, 8);
    x += __shfl_xor_sync(0xffffffff, x, 4);
    x += __shfl_xor_sync(0xffffffff, x, 2);
    x += __shfl_xor_sync(0xffffffff, x, 1);
    return x;
}


__device__ __forceinline__ float rsum32_(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 16);
    x += __shfl_xor_sync(0xffffffff, x, 8);
    x += __shfl_xor_sync(0xffffffff, x, 4);
    x += __shfl_xor_sync(0xffffffff, x, 2);
    x += __shfl_xor_sync(0xffffffff, x, 1);
    return x;
}


__device__ __forceinline__ int rsum32_(int x) {
    x += __shfl_xor_sync(0xffffffff, x, 16);
    x += __shfl_xor_sync(0xffffffff, x, 8);
    x += __shfl_xor_sync(0xffffffff, x, 4);
    x += __shfl_xor_sync(0xffffffff, x, 2);
    x += __shfl_xor_sync(0xffffffff, x, 1);
    return x;
}


__device__ __forceinline__ int rmax32_(int x) {
    x = max(x, __shfl_xor_sync(0xffffffff, x, 16));
    x = max(x, __shfl_xor_sync(0xffffffff, x, 8));
    x = max(x, __shfl_xor_sync(0xffffffff, x, 4));
    x = max(x, __shfl_xor_sync(0xffffffff, x, 2));
    x = max(x, __shfl_xor_sync(0xffffffff, x, 1));
    return x;
}


__device__ __forceinline__ float upSweep_(float x, int mask, int t1) {
    float y = __shfl_xor_sync(0xffffffff, x, mask);
    return t1 ? y + x : x;
}


__device__ __forceinline__ int upSweep_(int x, int mask, int t1) {
    int y = __shfl_xor_sync(0xffffffff, x, mask);
    return t1 ? y + x : x;
}


__device__ __forceinline__ float downSweep_(float x, int mask, int t1, int t2) {
    float y = __shfl_xor_sync(0xffffffff, x, mask);
    return t1 ? (t2 ? y + x : y) : x;
}


__device__ __forceinline__ int downSweep_(int x, int mask, int t1, int t2) {
    int y = __shfl_xor_sync(0xffffffff, x, mask);
    return t1 ? (t2 ? y + x : y) : x;
}


__device__ __forceinline__ float cumsum32_(float x) {
    int lane_idx = threadIdx.x % CUDA_WARP_SIZE;
    x = upSweep_(x, 0x01, bfe_(0xaaaaaaaau, lane_idx));
    x = upSweep_(x, 0x02, bfe_(0x88888888u, lane_idx));
    x = upSweep_(x, 0x04, bfe_(0x80808080u, lane_idx));
    x = upSweep_(x, 0x08, bfe_(0x80008000u, lane_idx));
    x = upSweep_(x, 0x10, bfe_(0x80000000u, lane_idx));
    x = lane_idx == 31 ? 0.f : x;
    x = downSweep_(x, 0x10, bfe_(0x80008000u, lane_idx), bfe_(0x80000000u, lane_idx));
    x = downSweep_(x, 0x08, bfe_(0x80808080u, lane_idx), bfe_(0x80008000u, lane_idx));
    x = downSweep_(x, 0x04, bfe_(0x88888888u, lane_idx), bfe_(0x80808080u, lane_idx));
    x = downSweep_(x, 0x02, bfe_(0xaaaaaaaau, lane_idx), bfe_(0x88888888u, lane_idx));
    x = downSweep_(x, 0x01, 1                          , bfe_(0xaaaaaaaau, lane_idx));
    return x;
}


__device__ __forceinline__ int cumsum32_(int x) {
    int lane_idx = threadIdx.x % CUDA_WARP_SIZE;
    x = upSweep_(x, 0x01, bfe_(0xaaaaaaaau, lane_idx));
    x = upSweep_(x, 0x02, bfe_(0x88888888u, lane_idx));
    x = upSweep_(x, 0x04, bfe_(0x80808080u, lane_idx));
    x = upSweep_(x, 0x08, bfe_(0x80008000u, lane_idx));
    x = upSweep_(x, 0x10, bfe_(0x80000000u, lane_idx));
    x = lane_idx == 31 ? 0 : x;
    x = downSweep_(x, 0x10, bfe_(0x80008000u, lane_idx), bfe_(0x80000000u, lane_idx));
    x = downSweep_(x, 0x08, bfe_(0x80808080u, lane_idx), bfe_(0x80008000u, lane_idx));
    x = downSweep_(x, 0x04, bfe_(0x88888888u, lane_idx), bfe_(0x80808080u, lane_idx));
    x = downSweep_(x, 0x02, bfe_(0xaaaaaaaau, lane_idx), bfe_(0x88888888u, lane_idx));
    x = downSweep_(x, 0x01, 1                          , bfe_(0xaaaaaaaau, lane_idx));
    return x;
}


__device__ __forceinline__ float bsort32_(float x) {
    x = swap_(x, 0x01, bfe_(threadIdx.x, 1) ^ bfe_(threadIdx.x, 0));
    x = swap_(x, 0x02, bfe_(threadIdx.x, 2) ^ bfe_(threadIdx.x, 1));
    x = swap_(x, 0x01, bfe_(threadIdx.x, 2) ^ bfe_(threadIdx.x, 0));
    x = swap_(x, 0x04, bfe_(threadIdx.x, 3) ^ bfe_(threadIdx.x, 2));
    x = swap_(x, 0x02, bfe_(threadIdx.x, 3) ^ bfe_(threadIdx.x, 1));
    x = swap_(x, 0x01, bfe_(threadIdx.x, 3) ^ bfe_(threadIdx.x, 0));
    x = swap_(x, 0x08, bfe_(threadIdx.x, 4) ^ bfe_(threadIdx.x, 3));
    x = swap_(x, 0x04, bfe_(threadIdx.x, 4) ^ bfe_(threadIdx.x, 2));
    x = swap_(x, 0x02, bfe_(threadIdx.x, 4) ^ bfe_(threadIdx.x, 1));
    x = swap_(x, 0x01, bfe_(threadIdx.x, 4) ^ bfe_(threadIdx.x, 0));
    x = swap_(x, 0x10, bfe_(threadIdx.x, 4));
    x = swap_(x, 0x08, bfe_(threadIdx.x, 3));
    x = swap_(x, 0x04, bfe_(threadIdx.x, 2));
    x = swap_(x, 0x02, bfe_(threadIdx.x, 1));
    x = swap_(x, 0x01, bfe_(threadIdx.x, 0));
    return x;
}


__device__ __forceinline__ int bsort32_(int x) {
    x = swap_(x, 0x01, bfe_(threadIdx.x, 1) ^ bfe_(threadIdx.x, 0));
    x = swap_(x, 0x02, bfe_(threadIdx.x, 2) ^ bfe_(threadIdx.x, 1));
    x = swap_(x, 0x01, bfe_(threadIdx.x, 2) ^ bfe_(threadIdx.x, 0));
    x = swap_(x, 0x04, bfe_(threadIdx.x, 3) ^ bfe_(threadIdx.x, 2));
    x = swap_(x, 0x02, bfe_(threadIdx.x, 3) ^ bfe_(threadIdx.x, 1));
    x = swap_(x, 0x01, bfe_(threadIdx.x, 3) ^ bfe_(threadIdx.x, 0));
    x = swap_(x, 0x08, bfe_(threadIdx.x, 4) ^ bfe_(threadIdx.x, 3));
    x = swap_(x, 0x04, bfe_(threadIdx.x, 4) ^ bfe_(threadIdx.x, 2));
    x = swap_(x, 0x02, bfe_(threadIdx.x, 4) ^ bfe_(threadIdx.x, 1));
    x = swap_(x, 0x01, bfe_(threadIdx.x, 4) ^ bfe_(threadIdx.x, 0));
    x = swap_(x, 0x10, bfe_(threadIdx.x, 4));
    x = swap_(x, 0x08, bfe_(threadIdx.x, 3));
    x = swap_(x, 0x04, bfe_(threadIdx.x, 2));
    x = swap_(x, 0x02, bfe_(threadIdx.x, 1));
    x = swap_(x, 0x01, bfe_(threadIdx.x, 0));
    return x;
}



__device__ __inline__ int shuffle32_(curandState* state) {
    int idx1 = threadIdx.x;
    int idx2;
    int rand;

    for (int shift = 1; shift < 32; shift <<= 1) {
        idx2  = idx1;
        rand  = curand(state);
        rand += __shfl_xor_sync(0xffffffff, rand, shift);
        idx2  = __shfl_xor_sync(0xffffffff, idx1, shift);
        if (rand & 1)
            idx1 = idx2;
        __syncthreads();
    }
    return idx1;
}


#endif
