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
 * @brief   Detect GPU architecture
 * @author  CM Lee
 * @date    05/23/2023
 */

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

#include "parser/input.hpp"
#include "mclog/logger.hpp"

#include "tuning.cuh"


namespace mcutil {


	constexpr int BLOCK_DIM_QMD    = 128;
	constexpr int BLOCK_PER_SM_QMD = 64;


	/**
	* @brief Detect hardware architecture and calculate the number of 
	*        shading unit (CUDA core) of selected card
	* @param dev_prop Device property of selected card
	* @return Number of CUDA cores
	*/
	size_t getShadingUnitsNumber(const cudaDeviceProp& dev_prop);


	class DeviceController {
	private:
		size_t _gpu_id;
		size_t _number_of_shader;
		size_t _global_memory_cap;
		size_t _shared_memory_cap;
		// common
		size_t _thread;
		size_t _block;
		// QMD
		size_t _thread_qmd;
		size_t _block_qmd;
		size_t _seed;
		double _buffer_ratio;
		size_t _block_decay_limit;
		double _block_decay_rate;
	public:


		DeviceController();


		DeviceController(ArgInput& args);


		size_t freeMemory() const;


		size_t sharedMemory() const { return this->_shared_memory_cap; }
		size_t gpuID() const { return this->_gpu_id; }

		size_t thread() const { return this->_thread; }
		size_t block()  const { return this->_block; }

		size_t threadQMD() const { return this->_thread_qmd; }
		size_t blockQMD()  const { return this->_block_qmd; }

		size_t seed() const { return this->_seed; }
		double bufferRatio()     const { return this->_buffer_ratio; }
		double blockDecayRate()  const { return this->_block_decay_rate; }
		size_t blockDecayLimit() const { return this->_block_decay_limit; }


	};


}