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
 * @file    module/hadron/projectile.cuh
 * @brief   Aligned projectile mass table for efficient access
 * @author  CM Lee
 * @date    04/02/2024
 */

#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "physics/constants.cuh"


namespace Hadron {
    namespace Projectile {


        constexpr int ZA_SCORING_MASK_DIM    = 128;
        constexpr int ZA_SCORING_MASK_HION   = ZA_SCORING_MASK_DIM - 1;  // heavy ion (which cannot be filtered)
        constexpr int ZA_SCORING_MASK_STRIDE = 8 * sizeof(uint32_t);
        constexpr int ZA_SCORING_MASK_SIZE   = ZA_SCORING_MASK_DIM / ZA_SCORING_MASK_STRIDE;
        

        constexpr int TABLE_MAX_Z = 8;   // up to oxygen
        constexpr int TABLE_MAX_A = 18;

        constexpr int REFERENCE_ZA[TABLE_MAX_Z]   = { 1001, 2004, 3007, 4009, 5010, 6012, 7014, 8016 };
        constexpr int REFERENCE_SPIN[TABLE_MAX_Z] = { 1,    0,    -3,   -3,   6,    0,    2,    0    };


    }
}