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
 * @file    module/qmd/collision.cuh
 * @brief   QMD binary collision (G4QMDCollision.hh)
 * @author  CM Lee
 * @date    02/23/2024
 */

#pragma once

#include <cuda_runtime.h>
#include <assert.h>
#include <curand_kernel.h>

#include "constants.cuh"
#include "buffer.cuh"

#include "device/shuffle.cuh"
#include "hadron/xs_dev.cuh"
#include "hadron/auxiliary.cuh"


namespace RT2QMD {
    namespace Collision {


        constexpr int BINARY_CANDIDATE_MAX   = 512;                              // binary mask
        constexpr int BINARY_MAX_TRIAL_SHIFT = 2;
        constexpr int BINARY_MAX_TRIAL       = (1 << BINARY_MAX_TRIAL_SHIFT);    // G4QMDCollision (iitry < 4) in line 589
        constexpr int BINARY_MAX_TRIAL_2     = BINARY_MAX_TRIAL << 1;

        constexpr float ENERGY_CONSERVATION_VIOLATION_THRES = 1.2e-4f;


        extern __constant__ bool USING_INCL_NN_SCATTERING;
        extern __device__   Hadron::NNScatteringTable* g4_nn_table;


        /**
        * @brief for caching memory in collision phase
        */
        typedef struct CollisionSharedMem {
            float  __offset_shared[Buffer::MODEL_CACHING_OFFSET];
            unsigned int is_hit[MAX_DIMENSION_CLUSTER_B];           //! @brief participant hit mask
            int    pauli_blocked;                                   //! @brief for broadcast
            float  pot_ini[2];                                      //! @brief potential before collision (for candidate i and j)
            float  pot_fin[2];                                      //! @brief potential after collision (for candidate i and j)
            int    n_binary_candidate;                              //! @brief number of collision candidates
            int    exit_condition;                                  //! @brief exit condition broadcaster
            uchar2 binary_candidate[BINARY_CANDIDATE_MAX];
            float  boostx[CUDA_WARP_SIZE];
            float  boosty[CUDA_WARP_SIZE];
            float  boostz[CUDA_WARP_SIZE];
            int    flag[CUDA_WARP_SIZE];
            float  mass[CUDA_WARP_SIZE];
            float  rx[CUDA_WARP_SIZE];
            float  ry[CUDA_WARP_SIZE];
            float  rz[CUDA_WARP_SIZE];
            float  px[CUDA_WARP_SIZE];
            float  py[CUDA_WARP_SIZE];
            float  pz[CUDA_WARP_SIZE];

        } CollisionSharedMem;


        constexpr int COLLISION_SHARED_MEM_OFFSET
            = (sizeof(CollisionSharedMem) - sizeof(float) * CUDA_WARP_SIZE * 8) / 4;


        __host__ cudaError_t setNNTable(CUdeviceptr table_handle, bool use_incl_model);


        __device__ void boostNN();


        __device__ void sortCollisionCandidates();


        __device__ void elasticScatteringNN();


        __device__ void calPotentialSegBeforeBinaryCollision(int prid, int ip);


        __device__ void calPotentialSegAfterBinaryCollision(int prid, int ip);


        __device__ void calKinematicsOfBinaryCollisions();


        __device__ bool isPauliBlocked(int candidate_id);


    }
}