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
 * @file    module/deexcitation/channel_unstable_breakup.cu
 * @brief   Excessive energy breakup, for unstable & zero excitation energy nuclei
 * @author  CM Lee
 * @date    07/25/2024
 */


#include "channel_unstable_breakup.cuh"
#include "device/shuffle.cuh"

#include <stdio.h>


namespace deexcitation {
    namespace breakup {


        __device__ bool emitParticle(curandState* state) {
            uchar4* cache_zaev = reinterpret_cast<uchar4*>(mcutil::cache_univ + CUDA_WARP_SIZE);
            uchar4  zaev       = cache_zaev[threadIdx.x];
            float   m0         = mcutil::cache_univ[CUDA_WARP_SIZE +     blockDim.x + threadIdx.x];
            float   exc_energy = mcutil::cache_univ[CUDA_WARP_SIZE + 3 * blockDim.x + threadIdx.x];

            m0 += exc_energy;  // invariant mass

            float   exca = -1000.f;
            float   m1, m2;
            bool    found_channel = false;
            for (int i = CHANNEL::CHANNEL_NEUTRON; i < CHANNEL::CHANNEL_UNKNWON; i++) {
                int zres = (int)zaev.x - PROJ_Z[i];
                int ares = (int)zaev.y - PROJ_A[i];
                if (zres >= 0 && ares >= zres && ares >= PROJ_A[i]) {  // physically allowed channel
                    if (ares <= 4) {  // simple channel
                        for (int j = CHANNEL::CHANNEL_NEUTRON; j < CHANNEL::CHANNEL_UNKNWON; ++j) {
                            if (zres == PROJ_Z[j] && ares == PROJ_A[j]) {
                                float delm = m0 - PROJ_M[i] - PROJ_M[j];
                                if (delm > exca) {
                                    m2      = PROJ_M[i];
                                    m1      = PROJ_M[j];
                                    exca    = delm;
                                    if (delm > 0.f) {
                                        exca          = 0.f;
                                        found_channel = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if (found_channel) {
                        zaev.x = (unsigned char)zres;
                        zaev.y = (unsigned char)ares;
                        zaev.z = (unsigned char)PROJ_Z[i];
                        zaev.w = (unsigned char)PROJ_A[i];
                        break;
                    }
                    // no simple channel
                    float mres = mass_table[zres].get(ares);
                    float e    = m0 - mres - PROJ_M[i];

                    if (e >= exca) {
                        m2      = PROJ_M[i];
                        m1      = (ares > 4 && e > 0.f) ? mres + e * curand_uniform(state) : mres;
                        exca    = e;
                        if (e > 0.f) {
                            exca   = m1 - mres;
                            zaev.x = (unsigned char)zres;
                            zaev.y = (unsigned char)ares;
                            zaev.z = (unsigned char)PROJ_Z[i];
                            zaev.w = (unsigned char)PROJ_A[i];
                            found_channel = true;
                            break;
                        }
                    }
                }
            }

            if (!found_channel || m0 < m1 + m2) {
                if (m0 + 1e-1f < m1 + m2) {
                    zaev.w = 0;
                    exca   = 0.f;
                }
                else
                    m0 = m1 + m2;
            }

            // ZA cache
            cache_zaev[threadIdx.x] = zaev;

            // mass
            mcutil::cache_univ[32 + 1 * blockDim.x + threadIdx.x] = m1;
            mcutil::cache_univ[32 + 2 * blockDim.x + threadIdx.x] = m2;

            // excitation energy of residual
            mcutil::cache_univ[32 + 3 * blockDim.x + threadIdx.x] = exca;
            mcutil::cache_univ[32 + 4 * blockDim.x + threadIdx.x] = 0.f;

            if (zaev.w == 0)
                return false;  // fail

            // momentum
            float etot1    = 0.5f * ((m0 - m2) * (m0 + m2) + m1 * m1) / m0;
            etot1 = fmaxf(etot1, m1);
            float momentum = sqrtf((etot1 - m1) * (etot1 + m1));

            // random isotropic direction
            float cost, sint;
            float cosp, sinp;
            float angle;

            // polar
            cost  = 1.f - 2.f * curand_uniform(state);
            sint  = sqrtf(fmaxf(0.f, 1.f - cost * cost));
            // azimuthal
            angle = constants::FP32_TWO_PI * curand_uniform(state);
            __sincosf(angle, &sinp, &cosp);
            
            mcutil::cache_univ[CUDA_WARP_SIZE + 5 * blockDim.x + threadIdx.x] = momentum * sint * cosp;  // X
            mcutil::cache_univ[CUDA_WARP_SIZE + 6 * blockDim.x + threadIdx.x] = momentum * sint * sinp;  // Y
            mcutil::cache_univ[CUDA_WARP_SIZE + 7 * blockDim.x + threadIdx.x] = momentum * cost;         // Z

            return true;
        }


    }
}