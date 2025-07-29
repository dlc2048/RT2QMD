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
 * @file    module/deexcitation/nuclear_level.cuh
 * @brief   Deexcitation nuclear level data
 * @author  CM Lee
 * @date    07/09/2024
 */


#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "device/memory.cuh"
#include "physics/constants.cuh"


namespace deexcitation {


    constexpr float FIXED_LEVEL_DENSITY = 0.075f;  // Simple level density [1/MeV]


    extern __constant__ bool USE_SIMPLE_LEVEL_DENSITY;


    __inline__ __device__ float getLevelDensityParameter(int a) {
        if (USE_SIMPLE_LEVEL_DENSITY)
            return (float)a * FIXED_LEVEL_DENSITY;
        else
            return 0.058025f * (float)a * (1.f + 5.9059f * powf((float)a, -constants::ONE_OVER_THREE));  // Mengoni, A., & Nakajima, Y. (1994)
    }


    __inline__ __device__ float getFissionLevelDensityParameter(int z, int a) {
        float ldp = getLevelDensityParameter(a);
        if (z >= 89)
            ldp *= 1.05f;
        else if (z <= 85)
            ldp *= 1.03f;
        else
            ldp *= 1.03f + 0.005f * (float)(z - 85);
        return ldp;
    }


    __host__ cudaError_t setSimpleLevelDensity(bool use_it);


}