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
 * @file    module/deexcitation/deexcitation.cuh
 * @brief   Deexcitation kernel
 * @author  CM Lee
 * @date    07/15/2024
 */

#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "device/memory.cuh"

#include "auxiliary.cuh"
#include "channel_evaporation.cuh"
#include "channel_fission.cuh"
#include "channel_photon.cuh"
#include "channel_unstable_breakup.cuh"
#include "cross_section.cuh"


namespace deexcitation {


    __device__ void boostAndWrite(mcutil::BUFFER_TYPE origin, mcutil::UNION_FLAGS flags, float m0, bool is_secondary);


    /**
    * Dostrovsky inverse cross-section
    */
    namespace Dostrovsky {


        __global__ void __kernel__deexcitationStep();
        __host__ void deexcitationStep(int block, int thread);


    }


    /**
    * Chaterjee inverse cross-section
    */
    namespace Chatterjee {


        __global__ void __kernel__deexcitationStep();
        __host__ void deexcitationStep(int block, int thread);


    }


    /**
    * Kalbach inverse cross-section
    */
    namespace Kalbach {


        __global__ void __kernel__deexcitationStep();
        __host__ void deexcitationStep(int block, int thread);


    }


}