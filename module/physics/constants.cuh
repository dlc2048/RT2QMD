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
 * @file    module/physics/constants.cuh
 * @brief   Universal constants
 * @author  CM Lee
 * @date    04/01/2024
 */


#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include <cmath>


namespace constants {


    // Mathematic quantities

    constexpr float  FP32_PI          = (float)M_PI;
    constexpr float  FP32_TWO_PI      = (float)(2.0 * M_PI);
    constexpr float  FP32_FOUR_PI     = (float)(4.0 * M_PI);
    constexpr float  FP32_TEN_PI      = (float)(10.0 * M_PI);
    constexpr float  FP32_TWE_PI      = (float)(20.0 * M_PI);
    constexpr float  FP32_TEN_PI_I    = (float)(1.0 / M_PI / 10.0);
    constexpr float  FP32_PI_SQ       = (float)(M_PI * M_PI);
    constexpr float  FP32_PI_OVER_TWO = (float)(0.5 * M_PI);

    constexpr float  FP32_DEG_TO_RAD  = (float)(M_PI / 180.0);

    constexpr double FP64_PI       = M_PI;
    constexpr double FP64_TWO_PI   = 2.0 * FP64_PI;
    constexpr double FP64_TWE_PI   = 20.0 * FP64_PI;
    constexpr double FP64_PI_SQ    = M_PI * M_PI;

    constexpr float  ONE_OVER_THREE  = (float)(1.0 / 3.0);
    constexpr float  ONE_OVER_SIXTH  = (float)(1.0 / 6.0);
    constexpr float  ONE_OVER_12     = (float)(1.0 / 12.0);
    constexpr float  TWO_OVER_THREE  = (float)(2.0 / 3.0);
    constexpr float  THREE_OVER_TWO  = (float)(3.0 / 2.0);
    constexpr float  FOUR_OVER_THREE = (float)(4.0 / 3.0);
    constexpr float  FIVE_OVER_SIXTH = (float)(5.0 / 6.0);
    constexpr float  FIVE_OVER_12    = (float)(5.0 / 12.0);

    constexpr float SQRT_FIVE_OVER_THREE   = 1.290994449f;         //! @brief sqrt(5/3)
    constexpr float SQRT_THREE_OVER_FIFTHS = 0.7745966692414834f;  //! @brief sqrt(3/5)


    // Physical quantities

    constexpr float  MASS_PROTON        = 938.272013f;          //! @brief Proton mass [MeV/c^2]
    constexpr float  MASS_PROTON_GEV    = MASS_PROTON * 1e-3f;  //! @brief Proton mass [GeV/c^2]
    constexpr float  MASS_PROTON_GEV_S  = MASS_PROTON_GEV * MASS_PROTON_GEV;  //! @brief Proton mass ^ 2 [GeV^2/c^4]
    //! @brief Proton mass x 2 [MeV/c^2]
    constexpr float  MASS_PROTON_D      = (float)((double)MASS_PROTON * 2.0);
    //! @brief 1 / proton mass [c^2/MeV]
    constexpr float  MASS_PROTON_I      = (float)(1.0 / (double)MASS_PROTON);  
    constexpr float  MASS_NEUTRON       = 939.56536f;            //! @brief Neutron mass [MeV/c^2]
    constexpr float  MASS_NEUTRON_GEV   = MASS_NEUTRON * 1e-3f;  //! @brief Neutron mass [GeV/c^2]
    constexpr float  MASS_NEUTRON_GEV_S = MASS_NEUTRON_GEV * MASS_NEUTRON_GEV;  //! @brief Neutron mass ^ 2 [GeV^2/c^4]
    constexpr float  MASS_ELECTRON      = 0.510998910f;          //! @brief Electron mass [MeV/c^2]
    //! @brief Electron mass x 2 [MeV/c^2]
    constexpr float  MASS_ELECTRON_D  = (float)((double)MASS_ELECTRON * 2.0);      
    //! @brief 1 / (Electron mass x 2) [c^2/MeV]
    constexpr float  MASS_ELECTRON_DI = (float)(1.0 / (2.0 * (double)MASS_ELECTRON)); 
    //! @brief Electron mass ^ 2 [MeV^2/c^4]
    constexpr float  MASS_ELECTRON_S  = (float)((double)MASS_ELECTRON * (double)MASS_ELECTRON); 
    //! @brief 1 / (Electron mass ^ 2) [c^4/MeV^2]
    constexpr float  MASS_ELECTRON_SI = (float)(1.0 / (double)MASS_ELECTRON / (double)MASS_ELECTRON);
    //! @brief 1 / Electron mass [c^2/MeV]
    constexpr float  MASS_ELECTRON_I  = (float)(1 / (double)MASS_ELECTRON);   

    constexpr double BARN             = 1e24;            //! @brief Barn unit [cm^2]
    constexpr double MILLIBARN        = BARN * 1e3;
    constexpr double ATOMIC_MASS_UNIT = 931.494028;      //! @brief Atomic mass unit, AMU [MeV/c^2]
    constexpr double FSC              = 7.2973525e-3;    //! @brief Fine structure constant (~1/137)
    constexpr double FSC2             = FSC * FSC;       //! @brief Square of fine structure constant
    constexpr double HBARC            = 197.32705e-13;   //! @brief Planck constant in unit of speed of light [MeV * cm]
    constexpr double ELECTRON_RADIUS  = 2.81794092e-13;  //! @brief Classical electron radius [cm]
    
    constexpr float  FP32_FSC           = (float)FSC;
    constexpr float  FP32_HBARC         = (float)(HBARC * 1e+10);        //! @brief Planck constant in unit of speed of light [GeV * fm]
    constexpr float  FP32_FSC_HBARC     = (float)(FSC * HBARC * 1e+10);  // [GeV * fm]
    constexpr float  FP32_FSC_HBARC_MEV = (float)(FSC * HBARC * 1e+13);  // [MeV * fm]
    constexpr float  FP32_PF            = 1.37f * FP32_HBARC;            //! @brief Fermi momentum [GeV/c]

    // Stopping power coefficient
    constexpr double TWOPI_MC2_RCL2
        = 2.0 * M_PI * ELECTRON_RADIUS * ELECTRON_RADIUS * BARN * (double)MASS_ELECTRON;

    constexpr float  FP32_TWOPI_MC2_RCL2 = (float)TWOPI_MC2_RCL2;

    constexpr float  LET_RANGE_UNDEFINED_I = 1e+3f;

    // constants
    constexpr float  NEUTRON_HP_CUTOFF = 19.9f;  //! @brief ENDF cutoff kinetic energy [MeV]

    // Fragmentation model energy boundary [MeV/u]  
    // BME (< 100 MeV/u) <-> QMD or Abrasion (> 70 MeV/u)
    constexpr float  FRAGMENTATION_MODEL_BOUNDARY     = 70.0f;
    constexpr float  FRAGMENTATION_MODEL_BOUNDARY_GEV = FRAGMENTATION_MODEL_BOUNDARY * 1e-3f;

    constexpr float  E_SQUARED = 1.439964f;  //! @brief Coulomb conversion factor (e^2/(4 pi epsilon0), [MeV*fm])

}
