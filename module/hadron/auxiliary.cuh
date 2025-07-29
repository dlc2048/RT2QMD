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
 * @file    module/hadron/auxiliary.cuh
 * @brief   NN collision data
 * @author  CM Lee
 * @date    06/11/2025
 */

#pragma once

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "physics/constants.cuh"


namespace Hadron {


    typedef struct NNScatteringTable {
        int    nenergy[2];
        int    nangle[2];
        float* elab[2];
        float* sig[2];


        __host__ void free();


        __inline__ __device__ float sampleMu(curandState* state, bool isospin, float elab);


    } NNScatteringTable;


#ifdef __CUDACC__


    __device__ float NNScatteringTable::sampleMu(curandState* state, bool isospin, float elab) {
        // not used
    }


#endif


}