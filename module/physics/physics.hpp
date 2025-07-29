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
 * @file    module/physics/physics.hpp
 * @brief   Methods related to physics
 * @author  CM Lee
 * @date    04/01/2024
 */


#pragma once

#include <cmath>

#include <cuda_runtime.h>

#include "constants.hpp"


namespace physics {


    double coulombCorrection(int z);  // PEGS4 FCOULCP
    double xsifp(int z);              // PEGS4 XSIF


    inline int getZnumberFromZA(int za) {
        return za / 1000;
    }


    inline int getAnumberFromZA(int za) {
        return za % 1000;
    }


    inline int2 splitZA(int za) {
        return { za / 1000, za % 1000 };
    }


    class DensityEffect {
    private:
        static double _GAS_DENSITY_THRES;
        double        _m;
        double        _c;
        double        _a;
        double2       _x;
    public:
        DensityEffect(
            double mean_ie,
            double density,
            double plasma_frequency
        );
        double get(double energy);
    };

}