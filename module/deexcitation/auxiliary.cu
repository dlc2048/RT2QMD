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
 * @file    module/deexcitation/auxiliary.cu
 * @brief   Data structure for de-excitation channel
 * @author  CM Lee
 * @date    07/08/2024
 */


#include "auxiliary.cuh"


namespace deexcitation {


    __device__ float PROJ_M[CHANNEL::CHANNEL_UNKNWON];
    __device__ float PROJ_M2[CHANNEL::CHANNEL_UNKNWON];
    __device__ float PROJ_CB_RHO[CHANNEL::CHANNEL_UNKNWON];

    __constant__ bool BUFFER_HAS_HID;
    __device__ mcutil::RingBuffer* buffer_catalog;

    __device__ curandState* rand_state;
    __device__ Nucleus::MassTable* mass_table;
    __device__ Nucleus::LongLivedNucleiTable long_lived_table;

    __constant__ float COULOMB_RATIO;
    __device__ float* coulomb_r0;

    __device__ float CJXS_PARAM[DIM_XS][CHANNEL::CHANNEL_UNKNWON];
    __device__ float KMXS_PARAM[DIM_XS][CHANNEL::CHANNEL_UNKNWON]; 

    __constant__ XS_MODEL XS_TYPE;
    __constant__ bool     DO_FISSION;
    __constant__ bool     USE_DISCRETE_LEVEL;


    __device__ float coulombBarrierRadius(int z, int a) {
        float r = Nucleus::explicitNuclearRadius({ (unsigned char)z, (unsigned char)a });
        if (r <= 0.f) {
            z = min(z, 92);
            r = coulomb_r0[z] * powf((float)a, constants::ONE_OVER_THREE);
        }
        return r;
    }


    __device__ float coulombBarrier(CHANNEL channel, int rz, int ra, float exc_energy) {
        float cb = constants::FP32_FSC_HBARC_MEV * (float)(PROJ_Z[channel] * rz)
            / (coulombBarrierRadius(rz, ra) + PROJ_CB_RHO[channel]);
        if (exc_energy > 0.f)
            cb /= 1.f + sqrtf(exc_energy / (float)ra * 0.5f);
        return cb;
    }


    __host__ cudaError_t setBufferHandle(CUdeviceptr handle, bool has_hid) {
        M_SOASymbolMapper(mcutil::RingBuffer*, handle, buffer_catalog);
        M_SOAPtrMapper(bool, has_hid, BUFFER_HAS_HID);
        return cudaSuccess;
    }


    __host__ cudaError_t setPrngHandle(CUdeviceptr handle) {
        M_SOASymbolMapper(curandState*, handle, rand_state);
        return cudaSuccess;
    }


    __host__ cudaError_t setMassTableHandle(CUdeviceptr handle) {
        M_SOASymbolMapper(Nucleus::MassTable*, handle, mass_table);
        return cudaSuccess;
    }


    __host__ cudaError_t setStableTable(const Nucleus::LongLivedNucleiTable& table_host) {
        return cudaMemcpyToSymbol(long_lived_table, &table_host, sizeof(Nucleus::LongLivedNucleiTable));
    }


    __host__ cudaError_t setCoulombBarrierRadius(float* cr_arr) {
        M_SOAPtrMapper(float*, cr_arr, coulomb_r0);
        return cudaSuccess;
    }


    __host__ cudaError_t setCoulombRatio(float coulomb_penetration_ratio) {
        float coulomb_ratio = 1.f - coulomb_penetration_ratio;
        M_SOAPtrMapper(float, coulomb_ratio, COULOMB_RATIO);
        return cudaSuccess;
    }


    __host__ cudaError_t setChatterjeeXS(float xs_host[][(int)CHANNEL::CHANNEL_UNKNWON]) {
        return cudaMemcpyToSymbol(CJXS_PARAM[0][0], xs_host,
            sizeof(float) * DIM_XS * (int)CHANNEL::CHANNEL_UNKNWON);
    }


    __host__ cudaError_t setKalbackXS(float xs_host[][(int)CHANNEL::CHANNEL_UNKNWON]) {
        return cudaMemcpyToSymbol(KMXS_PARAM[0][0], xs_host,
            sizeof(float) * DIM_XS * (int)CHANNEL::CHANNEL_UNKNWON);
    }


    __host__ cudaError_t setEmittedParticleMass(float* mass_arr, float* mass2_arr) {
        cudaError_t res;
        res = cudaMemcpyToSymbol(PROJ_M[0],  mass_arr,  
            sizeof(float) * CHANNEL::CHANNEL_UNKNWON);
        if (res != cudaSuccess) return res;
        res = cudaMemcpyToSymbol(PROJ_M2[0], mass2_arr, 
            sizeof(float) * CHANNEL::CHANNEL_UNKNWON);
        if (res != cudaSuccess) return res;
        return res;
    }


    __host__ cudaError_t setEmittedParticleCBRho(float* rho_arr) {
        return cudaMemcpyToSymbol(PROJ_CB_RHO[0], rho_arr,
            sizeof(float) * CHANNEL::CHANNEL_UNKNWON);
    }


    __host__ cudaError_t setFissionFlag(bool flag) {
        M_SOAPtrMapper(bool, flag, DO_FISSION);
        return cudaSuccess;
    }


}