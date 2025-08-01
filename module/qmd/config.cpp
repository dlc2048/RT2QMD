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
 * @file    module/qmd/config.cpp
 * @brief   QMD config handler
 * @author  CM Lee
 * @date    02/15/2024
 */


#include "config.hpp"


namespace mcutil {


    template <>
    ArgumentCard InputCardFactory<RT2QMD::Host::Config>::_setCard() {
        ArgumentCard arg_card("QMD_SETTINGS");
        arg_card.insert<std::string>("nn_scattering", { "geant4" });
        arg_card.insert<bool>  ("measure_time" , std::vector<bool>{false});
        arg_card.insert<double>("timer_size"   , { 20 }, { 1 }, { 1000 });
        arg_card.insert<bool>  ("dump_action"  , std::vector<bool>{false});
        arg_card.insert<double>("dump_size"    , { 1000 }, { 1 }, { 1000000 });
        return arg_card;
    }


}


namespace RT2QMD {
    namespace Host {


        Config::Config() {
            this->_scattering_type = NN_SCATTERING_TYPE::NN_SCATTERING_GEANT4;
            this->_measure_time    = false;
            this->_timer_size      = 20;
            this->_do_dump_action  = false;
            this->_dump_size       = 1000;
        }


        Config::Config(mcutil::ArgInput& args) {
            this->_measure_time   = args["measure_time"].cast<bool>()[0];
            this->_timer_size     = (int)args["timer_size"].cast<double>()[0];
            this->_do_dump_action = args["dump_action"].cast<bool>()[0];
            this->_dump_size      = (int)args["dump_size"].cast<double>()[0];

            std::string scatt_type = args["nn_scattering"].cast<std::string>()[0];
            if (scatt_type == "geant4")
                this->_scattering_type = NN_SCATTERING_TYPE::NN_SCATTERING_GEANT4;
            else if (scatt_type == "incl") 
                this->_scattering_type = NN_SCATTERING_TYPE::NN_SCATTERING_INCL;
            else
                mclog::fatal("'nn_scatterin' must be 'geant4' or 'incl'");

            Config& def = Config::getInstance();
            def = *this;
        }


    }
}