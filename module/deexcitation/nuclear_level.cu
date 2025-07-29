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
 * @file    module/deexcitation/nuclear_level.cu
 * @brief   Deexcitation nuclear level data
 * @author  CM Lee
 * @date    07/09/2024
 */


#include "nuclear_level.cuh"


namespace deexcitation {


    __constant__ bool USE_SIMPLE_LEVEL_DENSITY = true;


    __host__ cudaError_t setSimpleLevelDensity(bool use_it) {
        return cudaSuccess;
    }


}