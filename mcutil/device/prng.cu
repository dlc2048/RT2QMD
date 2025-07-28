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
 * @file    mcutil/device/prng.hpp
 * @brief   Device-side random number generator
 * @author  CM Lee
 * @date    05/23/2023
 */

#include "prng.hpp"


namespace mcutil {

	// Initialize state memory of the random number generate, device kernel
	__global__ void __kernel__initialize(curandState* state, int* seed, int* offset) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		curand_init(*seed, idx, *offset,
			&state[idx]);
	}

	// Launch kernel
	void __host__initialize(int block, int thread, curandState* state, int* seed, int* offset) {
		__kernel__initialize <<< block, thread >>> (state, seed, offset);
	} 

}