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
 * @file    module/deexcitation/config.cpp
 * @brief   De-excitation config handler
 * @author  CM Lee
 * @date    08/07/2025
 */


#pragma once


#include "mclog/logger.hpp"
#include "parser/input.hpp"

#include "auxiliary.cuh"


namespace deexcitation {


    namespace Host {


        //! @brief De-excitation global configurations
        class Config : public Singleton<Config> {
            friend class Singleton<Config>;
        private:
            float    _coulomb_penetration_ratio;  // Coulomb barrier penetration ratio (corresponded to Geant4 evaporation OPTxs >= 1)
            XS_MODEL _xs_model;                   // Inverse-cross section model


            Config();


        public:


            Config(mcutil::ArgInput& args);


            float coulombPenetrationRatio() {
                return this->_coulomb_penetration_ratio;
            }


            XS_MODEL crossSectionModel() {
                return this->_xs_model;
            }


        };


    }


}