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
 * @file    module/qmd/constants.hpp
 * @brief   QMD constants handler
 * @author  CM Lee
 * @date    02/14/2024
 */


#pragma once

#include "constants.cuh"


namespace RT2QMD {
    namespace constants {


        constexpr double RHO_SATURATION    = 0.168;          //! @brief saturation density                     << G4QMDParameters::rho0


        /**
        * @brief QMD parameter initializer for device memory
        */
        class ParameterInitializer {
        private:
            double _rmass;  //! @brief mass of nucleus [GeV/c^2]
            double _ebin;   //! @brief bounding energy [GeV]
            double _esymm;  //! @brief symmetric energy [GeV]
            double _rpot;
        public:
            ParameterInitializer();
        };


    }
}