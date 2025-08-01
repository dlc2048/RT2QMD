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
 * @file    mcutil/device/assert.cuh
 * @brief   NaN detect
 * @author  CM Lee
 * @date    08/22/2024
 */


#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <assert.h>
#include <stdio.h>


#ifdef CUDA_THROW_NAN

#define assertNAN(value) { if (isnan(value)) {  \
    printf("%s %d [%s is NaN]\n", __FILE__, __LINE__, #value); \
    assert(false); \
}}

#else

#define assertNAN(value) ((void)0)

#endif