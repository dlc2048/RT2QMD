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
 * @file    module/nuc_secondary/secondary.cuh
 * @brief   Post-process secondary phase-space from nuclear inelastic reactions
 * @author  CM Lee
 * @date    07/30/2024
 */


#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "device/memory.cuh"

#include "particles/define_struct.hpp"

#include "transport/buffer.cuh"
#include "physics/constants.cuh"
#include "scoring/tally.cuh"
#include "hadron/nucleus.cuh"
#include "hadron/kinematics.cuh"
#include "hadron/projectile.cuh"


namespace Hadron {


    __global__ void __kernel__secondaryStep();
    __host__ void secondaryStep(int block, int thread);


    // external memory

    // 1 dimensional SOA data corresponded to material
    namespace MATSOA1D {

        // material
        extern __device__ float* density;   //! @brief Material density [g/cm3]


        __host__ cudaError_t setDensity(CUdeviceptr deviceptr);


    }


    // 1 dimensional SOA data corresponded to projectile
    namespace PROJSOA1D {

        extern __constant__ int PROJECTILE_TABLE_SIZE;

        // projectile
        extern __device__ int     offset[Hadron::Projectile::TABLE_MAX_Z];
        extern __device__ uchar2* za;                    // ZA lists
        extern __device__ float*  mass;                  // Mass [MeV]
        extern __device__ float*  mass_u;                // Mass per nucleon [MeV/u]
        extern __device__ float*  mass_ratio;            // Mass per nucleon ratio against reference projectile
        extern __device__ int*    spin;                  // Spin of projectile nucleus [hbar/2]


        __inline__ __device__ int getIonIndex(int z, int a) {
            int i;
            for (i = 0; i < PROJECTILE_TABLE_SIZE; ++i) {
                uchar2 za_this = za[i];
                if (za_this.x == z && za_this.y == a)
                    return i;
            }
            return -1;
        }


        __host__ cudaError_t setTableSize(int size);


        __host__ cudaError_t setOffset(int* offset_ptr);


        __host__ cudaError_t setTable(uchar2* za_ptr, float* mass_ptr, float* mass_u_ptr, float* mass_ratio_ptr, int* spin_ptr);


    }


    extern __device__ int* region_mat_table;                    // Region to material table

    extern __device__ tally::DeviceHandle tally_catalog;        // tally

    // buffer
    extern __constant__ bool BUFFER_HAS_HID;
    extern __device__ mcutil::RingBuffer* buffer_catalog;       // Particle buffer list

    // prng
    extern __device__ curandState* rand_state;

    // mass table
    extern __device__ Nucleus::MassTable* mass_table;


    __host__ cudaError_t setRegionMaterialTable(CUdeviceptr table);
    __host__ cudaError_t setTallyHandle(const tally::DeviceHandle& handle);
    __host__ cudaError_t setBufferHandle(CUdeviceptr handle, bool has_hid);
    __host__ cudaError_t setPrngHandle(CUdeviceptr handle);
    __host__ cudaError_t setMassTableHandle(CUdeviceptr handle);


}