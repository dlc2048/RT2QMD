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

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


namespace mcutil {


	__global__ void __kernel__initialize(curandState* state, int* seed, int* offset);


	/**
	* @brief Allocate memory and initialize rand state by seed and offset
	* @param block  Dimension of device kernel block
	* @param thread Dimension of device kernel thread 
	* @param state  XORWOW PRNG state
	* @param seed   XORWOW PRNG seed
	* @param offset Absolute offset of sequence
	*/
	void __host__initialize(int block, int thread, curandState* state, int* seed, int* offset);


	/**
	* @brief Device XORWOW PRNG memory handler
	*/
	class RandState {
	private:
		uint2        _dim;         //!< @brief Dimension of device kernel {block, thread}
		int          _seed;        //!< @brief XORWOW PRNG seed
		int          _offset;      //!< @brief Absolute offset of sequence
		curandState* _rand_state;  //!< @brief Device memory pointer of XORWOW PRNG state
		size_t       _memsize;     //!< @brief Total memory size (byte) of device side XORWOW PRNG state
		void _initialize(int block, int thread);

	public:
		

		/**
		* @brief Device XORWOW PRNG memory handler
		* @param dimension Dimension of device kernel {block, thread}
		* @param seed      XORWOW PRNG seed
		* @param offset    Absolute offset of sequence
		*/
		RandState(uint2 dimension, int seed, int offset);


		~RandState();


		/**
		* @brief Get the total memory size of device side XORWOW PRNG state
		* @return Total memory size (bytes)
		*/
		size_t memoryUsage();


		/**
		* @brief Device memory handle
		* @return CUdeviceptr handle
		*/
		CUdeviceptr deviceHandle();
	};

}