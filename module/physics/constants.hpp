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
 * @file    module/physics/constants.hpp
 * @brief   Universal constants
 * @author  CM Lee
 * @date    04/01/2024
 */


#pragma once

#include <map>
#include "singleton/singleton.hpp"
#include "constants.cuh"


namespace constants {

    constexpr double AVOGADRO           = 0.602214076;     //! @brief Avogadro number [#/mol/barn]
    constexpr double GAS_DENSITY_THRES  = 0.1;             //! @brief Gas density threshold [g/cm^3]
    constexpr double XS_MB_CM           = 1660.5655;       //! @brief XS conversion from [mb/atom] to [cm^2/g]

    constexpr double DALTON_MEV         = 931.49410242;    //! @brief Dalton unit in MeV [MeV]
    constexpr double BOHR_RADIUS        
        = HBARC / FSC / (double)MASS_ELECTRON;             //! @brief Bohr radius [cm]

    // Tsai's radiation logarithm
    constexpr double RADIATION_LOGARITHM_PRIM[4] = { 6.144, 5.621, 5.805, 5.924 };
    constexpr double RADIATION_LOGARITHM[4]      = { 5.310, 4.790, 4.740, 4.710 };

    // Radiation length
    constexpr double FSC_RCL = FSC * ELECTRON_RADIUS * ELECTRON_RADIUS * BARN;


}