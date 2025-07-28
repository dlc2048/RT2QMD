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
 * @file    mcutil/device/memory.cpp
 * @brief   Vector memory managing
 * @author  CM Lee
 * @date    05/23/2023
 */

#ifdef RT2QMD_STANDALONE
#include "exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "memory.hpp"


namespace mcutil {


    std::vector<float> cvtVectorDoubleToFloat(const std::vector<double>& arr) {
        return std::vector<float>(arr.begin(), arr.end());
    }


    std::vector<double> cvtVectorFloatToDouble(const std::vector<float>& arr) {
        return std::vector<double>(arr.begin(), arr.end());
    }


    std::vector<float2> cvtVectorDoubleToFloat(const std::vector<double2>& arr) {
        std::vector<float2> arr_out;
        arr_out.resize(arr.size());
        for (size_t i = 0; i < arr.size(); ++i) {
            arr_out[i].x = (float)arr[i].x;
            arr_out[i].y = (float)arr[i].y;
        }
        return arr_out;
    }


    std::vector<double2> cvtVectorDoubleToDouble2(const std::vector<double>& arr) {
        if (arr.size() % 2)
            throw std::length_error("Length of array must be even");

        std::vector<double2> arr_out;
        for (size_t i = 0; i < arr.size(); i += 2) {
            double2 xy;
            xy.x = arr[i];
            xy.y = arr[i + 1];
            arr_out.push_back(xy);
        }
        return arr_out;
    }

}