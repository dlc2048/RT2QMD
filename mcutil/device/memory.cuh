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
 * @file    mcutil/device/memory.hpp
 * @brief   host to device memcpy macro
 * @author  CM Lee
 * @date    05/23/2023
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda.h>


#define M_SOASymbolMapper(type, deviceptr, member_alias) {                               \
    type sym_member_alias = reinterpret_cast<type>(deviceptr);                            \
    cudaError_t err = cudaMemcpyToSymbol(member_alias, &sym_member_alias, sizeof(type));  \
    if (err != cudaSuccess) return err;                                                   \
}


#define M_SOAPtrMapper(type, ptr, member_alias) {                           \
    cudaError_t err = cudaMemcpyToSymbol(member_alias, &ptr, sizeof(type));  \
    if (err != cudaSuccess) return err;                                      \
}


namespace mcutil {


    // shared memory size
    constexpr int SIZE_SHARED_MEMORY_GLOBAL = 6144;
    constexpr int SIZE_SHARED_MEMORY_QMD    = 8192;
    constexpr int SIZE_SHARED_MEMORY_INCL   = 6144;
    constexpr int SIZE_SHARED_MEMORY_PEVAP  = 8192;


    extern __shared__ float cache_univ[];  //! @brief Universal shared memory (offset necessary)


}