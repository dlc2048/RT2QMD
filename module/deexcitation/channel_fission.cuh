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
 * @file    module/deexcitation/channel_fission.cuh
 * @brief   Competitive fission channel
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


namespace deexcitation {
    namespace fission {

        // Cameron corrections

        constexpr float PAIRING_CONSTANT    = 12.f;

        constexpr int   CAMERON_Z_TABLE_MIN = 11;
        constexpr int   CAMERON_Z_TABLE_MAX = 98;
        constexpr int   CAMERON_N_TABLE_MIN = 11;
        constexpr int   CAMERON_N_TABLE_MAX = 150;

        extern __device__ float* cameron_spz;  // spin + pairing, proton
        extern __device__ float* cameron_spn;  // spin + pairing, neutron
        extern __device__ float* cameron_sz;   // spin, proton
        extern __device__ float* cameron_sn;   // spin, neutron
        extern __device__ float* cameron_pz;   // pairing, proton
        extern __device__ float* cameron_pn;   // pairing, neutron

        // fission barrier constants [Barasenkov 1977]

        constexpr float BARASHENKOV_SURFACE = 17.9439f;
        constexpr float BARASHENKOV_COULOMB = 0.7053f;
        constexpr float BARASHENKOV_K       = 1.7826f;
        constexpr float BARASHENKOV_DELTA   = 1.248f;

        // fission parameter

        constexpr int   FISSION_PARAM_A1 = 134;
        constexpr int   FISSION_PARAM_A2 = 141;
        constexpr float FISSION_PARAM_A3 = (float)(FISSION_PARAM_A1 + FISSION_PARAM_A2) * 0.5f;
        constexpr float FISSION_Z_SIGMA  = 0.6f;


        __inline__ __device__ float CameronSpinPairingCorrection(int z, int n);


        __device__ float BarashenkovFissionBarrier(int z, int a);


        __inline__ __device__ float fissionBarrier(int z, int a, float exc_energy);


        __device__ float emissionProbability(int z, int a, float exc_energy);


        __inline__ __device__ float fissionPairingCorrection(int z, int n);


        __inline__ __device__ float pairingCorrection(int z, int n);


        __device__ void emitParticle(curandState* state, uchar4 zaev, float m0, float exc_energy);

        
        __device__ float fissionWeight(uchar4 zaev, float u, float fb, float sigma2, float sigmas);


        __inline__ __device__ float localExp(float x);


        __device__ int sampleMassNumber(curandState* state, int a, float as, float sigma2, float sigmas, float w);


        __device__ int sampleAtomicNumber(curandState* state, int a, int z, int af);


        __device__ float massDistribution(float x, int a, float as, float sigma2, float sigmas, float w);


        __device__ float fissionKineticEnergy(curandState* state, int a, int z, int af1, int af2, float tmax, float as, float sigma2, float sigmas, float w);


        __inline__ __device__ float ratio(float a, float a11, float b1, float a00);


        __inline__ __device__ float asymmetricRatio(int a, float a11);


        __inline__ __device__ float symmetricRatio(int a, float a11);


        __global__ void __kernel__fissionStep();
        __host__ void fissionStep(int block, int thread);


        __host__ cudaError_t setCameronSpinPairingCorrections(float* ptr_spz, float* ptr_spn);


        __host__ cudaError_t setCameronSpinCorrections(float* ptr_sz, float* ptr_sn);


        __host__ cudaError_t setCameronPairingCorrections(float* ptr_pz, float* ptr_pn);


#ifdef __CUDACC__


        __device__ float CameronSpinPairingCorrection(int z, int n) {
            return cameron_spz[z - 1] + cameron_spn[n - 1];
        }


        __device__ float fissionBarrier(int z, int a, float exc_energy) {
            return a >= 65 
                ? BarashenkovFissionBarrier(z, a) / (1.f + sqrtf(exc_energy * 0.5f / (float)a)) 
                : 1e5f;
        }


        __device__ float fissionPairingCorrection(int z, int n) {
            return ((1 - z + 2 * (z / 2)) + (1 - n + 2 * (n / 2))) * PAIRING_CONSTANT * rsqrtf((float)(z + n));
        }


        __device__ float pairingCorrection(int z, int n) {
            float res;
            if (z >= CAMERON_Z_TABLE_MIN && z <= CAMERON_Z_TABLE_MAX && n >= CAMERON_N_TABLE_MIN && n <= CAMERON_N_TABLE_MAX)
                res = cameron_pz[z - 1] + cameron_pn[n - 1];
            else
                res = fissionPairingCorrection(z, n);
            return res;
        }


        __device__ float localExp(float x) {
            return fabsf(x) < 8.f ? expf(-0.5f * x * x) : 0.f;
        }


        __device__ float ratio(float a, float a11, float b1, float a00) {
            float x,res;
            if (a11 >= a * 0.5f && a11 <= a00 + 10.f) {
                x   = (a11 - a00) / a;
                res = 1.f - b1 * x * x;
            }
            else {
                x   = 10.f / a;
                res = 1.f - b1 * x * x - 2.f * x * b1 * (a11 - a00 - 10.f) / a;
            }
            return res;
        }


        __device__ float asymmetricRatio(int a, float a11) {
            return ratio((float)a, a11, 23.5f, 134.f);
        }
        

        __device__ float symmetricRatio(int a, float a11) {
            return ratio((float)a, a11, 5.32f, (float)a * 0.5f);
        }


#endif


    }
}