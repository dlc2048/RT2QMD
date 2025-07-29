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
 * @file    module/hadron/kinematics.cuh
 * @brief   NN collision kinematics
 * @author  CM Lee
 * @date    07/03/2024
 */


#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "physics/constants.cuh"


namespace Hadron {


    __inline__ __device__ float3 makeBoostVector(float3 p1, float3 p2, bool first_is_proton, bool second_is_proton);


    __inline__ __device__ float momentumInLab(float e2cm, float m1, float m2);


#ifdef __CUDACC__


    __device__ float3 makeBoostVector(float3 p1, float3 p2, bool first_is_proton, bool second_is_proton) {
        float e1 = first_is_proton  ? constants::MASS_PROTON_GEV : constants::MASS_NEUTRON_GEV;
        float e2 = second_is_proton ? constants::MASS_PROTON_GEV : constants::MASS_NEUTRON_GEV;
        e1 = norm4df(p1.x, p1.y, p1.z, e1);
        e2 = norm4df(p2.x, p2.y, p2.z, e2);
        float eti = 1.f / (e1 + e2);
        float3 boost;
        boost.x = (p1.x + p2.x) * eti;
        boost.y = (p1.y + p2.y) * eti;
        boost.z = (p1.z + p2.z) * eti;
        return boost;
    }


    __device__ float momentumInLab(float e2cm, float m1, float m2) {
        float m1sq = m1 * m1;
        float m2sq = m2 * m2;
        float plab2 = (e2cm * e2cm - 2.f * e2cm * (m1sq + m2sq) + (m1sq - m2sq) * (m1sq - m2sq)) / (4.f * m2sq);
        return sqrtf(fmaxf(0.f, plab2));
    }


#endif


}