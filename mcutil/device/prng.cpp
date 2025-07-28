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
 * @file    mcutil/device/prng.cpp
 * @brief   Device-side random number generator
 * @author  CM Lee
 * @date    05/23/2023
 */

#include <iomanip>
#include <iostream>
#include <string>

#ifdef RT2QMD_STANDALONE
#include "exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "prng.hpp"


namespace mcutil {

	RandState::RandState(uint2 dimension, int seed, int offset) {
		this->_dim.x  = dimension.x;
		this->_dim.y  = dimension.y;
		this->_seed   = seed;
		this->_offset = offset;
		this->_initialize(dimension.x, dimension.y);
	}


	RandState::~RandState() {
		CUDA_CHECK(cudaFree(this->_rand_state));
	}


	// Initialize state memory of the random number generate
	void RandState::_initialize(int block, int thread) {
		// Initialize rand state memory
		this->_memsize = sizeof(curandState) * block * thread;
		CUDA_CHECK(cudaMalloc((void**)&this->_rand_state,
			this->_memsize));

		// Copy seed and offset to device memory
		int* seed_d;
		int* offset_d;
		CUDA_CHECK(cudaMalloc((void**)&seed_d,
			sizeof(int)));
		CUDA_CHECK(cudaMemcpy(seed_d, &this->_seed,
			sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMalloc((void**)&offset_d,
			sizeof(int)));
		CUDA_CHECK(cudaMemcpy(offset_d, &this->_offset,
			sizeof(int), cudaMemcpyHostToDevice));

		__host__initialize(block, thread, this->_rand_state, seed_d, offset_d);

		// Free seed_d and offset_d
		CUDA_CHECK(cudaFree(seed_d));
		CUDA_CHECK(cudaFree(offset_d));
	}

	// Get state memory size
	size_t RandState::memoryUsage() {
		return this->_memsize;
	}

	// Get state memory device pointer
	CUdeviceptr RandState::deviceHandle() {
		return reinterpret_cast<CUdeviceptr>(this->_rand_state);
	}

}