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
 * @file    mcutil/device/memory_manager.cpp
 * @brief   Memory manager, for tracking memory usage in MCRT2
 * @author  CM Lee
 * @date    05/22/2024
 */

#include "memory_manager.hpp"


namespace mcutil {


    void DeviceMemoryHandlerInterface::_memoryUsageAppend(size_t size) {
        this->_dev_mem_size += size;
    }

    DeviceMemoryHandlerInterface::DeviceMemoryHandlerInterface() :
        _is_dummy    (false),
        _dev_mem_size(0x0u) {}


}