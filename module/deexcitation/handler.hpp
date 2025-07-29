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
 * @file    module/deexcitation/handler.hpp
 * @brief   De-excitation global variable, data, memory and device address handler
 * @author  CM Lee
 * @date    07/12/2024
 */


#pragma once

#include "device/memory.hpp"

#include "particles/define.hpp"
#include "hadron/nucleus.hpp"

#include "auxiliary.hpp"
#include "auxiliary.cuh"
#include "channel_fission.cuh"


namespace deexcitation {


    /**
    * @brief De-excitation device memory manager
    */
    class DeviceMemoryHandler : public mcutil::DeviceMemoryHandlerInterface {
    private:
        float* _dev_m;     // emitted mass [MeV/c^2]
        float* _dev_m2;    // m^2 [MeV^2/c^4]
        float* _dev_crho;  // coulomb barrier rho [fm]
    public:


        DeviceMemoryHandler();


        ~DeviceMemoryHandler();


        void summary() const;


    };

}