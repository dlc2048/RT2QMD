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
 * @file    module/scoring/tally_interface.cuh
 * @brief   Sparse implementation of the mesh tally
 * @author  CM Lee
 * @date    04/23/2025
 */


#pragma once

#include <stdio.h>

#include <cuda_runtime.h>
#include <assert.h>


namespace tally {

    
    // build sparse uncertainty matrix from data sparse coo index
    __global__ void __kernel__buildUncertaintyCOOSparse(
        float* __restrict__ dense_unc_ptr,
        int*   __restrict__ coo_index_ptr,
        float* __restrict__ coo_unc_ptr,
        int    nnz
    );


    __host__ void __host__buildUncertaintyCOOSparse(
        float* dense_unc_ptr,
        int*   coo_index_ptr,
        float* coo_unc_ptr,
        int    nnz,
        int    block,
        int    thread
    );


}