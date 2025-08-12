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
 * @file    module/deexcitation/cross_section.cuh
 * @brief   Inverse cross-section for evaporation
 * @author  CM Lee
 * @date    08/11/2025
 */


#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "auxiliary.cuh"


namespace deexcitation {


    /**
    * @brief Calculate Kalbach power parameter (thread-wised)
    * @param res_a   Mass number of residual nuclei
    * @param channel Evaporation channel
    */
    __inline__ __device__ float powerParameter(int res_a, int channel);


    /**
    * @brief Set Chatterjee shared memory parameters, block-wised
    */
    __device__ void setChatterjeeSharedParameters(int channel, int z, int a, float mass, float exc_energy);


    /**
    * @brief Set Chatterjee shared memory parameters, thread-wised
    */
    __device__ void setChatterjeeParameters(int channel, float cb, int res_a);


    /**
    * @brief Calculate Chatterjee inverse cross-section (block-wised)
    */
    __device__ float crossSectionChatterjeeShared(float k);


    /**
    * @brief Calculate Chatterjee inverse cross-section (thread-wised)
    * @param k       Kinetic energy of emitted particle [MeV]
    * 
    * @return Chatterjee inverse cross-section [mb]
    */
    __device__ float crossSectionChatterjee(int channel, float k);



#ifdef __CUDACC__


    __inline__ __device__ float powerParameter(int res_a, int channel) {
        return powf((float)res_a, KMXS_PARAM[6][channel]);
    }


    /*
    __device__ float crossSectionChatterjee(
        float k,
        float cb,
        float res_a13,
        CHANNEL channel,
        int z,
        int res_a
    ) {
        k = fmaxf(k, MAX_ENERGY_CJXS);

        float p     = CJXS_PARAM[0][channel] + (CJXS_PARAM[1][channel] + CJXS_PARAM[2][channel] / cb) / cb;
        float landa = CJXS_PARAM[4][channel];
        float mu    = CJXS_PARAM[5][channel];
        float nu    = CJXS_PARAM[7][channel];
        float amu   = powerParameter(res_a, channel);
        
        float sig, q, r, ji;
        if (channel == CHANNEL::CHANNEL_NEUTRON) {
            landa += CJXS_PARAM[3][channel] / res_a13;
            mu     = (mu + CJXS_PARAM[6][channel] * res_a13) * res_a13;
            nu     = fabsf((nu * (float)res_a + CJXS_PARAM[8][channel] * res_a13)
                * res_a13 + CJXS_PARAM[9][channel]);
            sig    = landa * k + mu + nu / k;
        }
        else {
            landa += CJXS_PARAM[3][channel] * (float)res_a;
            mu    *= amu;
            nu     = amu * (nu + (CJXS_PARAM[8][channel] + CJXS_PARAM[9][channel] * cb) * cb);
            q      = landa - nu / cb / cb - 2.f * p * cb;
            r      = mu + 2.f * nu / cb + p * cb * cb;
            ji     = fmaxf(k, cb);
            if (k < cb)
                sig = (p * k + q) * k + r;
            else
                sig = p * (k - ji) * (k - ji) + landa * k + mu + nu * (2.f - k / ji) / ji;
        }
        return fmaxf(sig, 0.f);
    }
    */


#endif


}