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


#include "config.hpp"


namespace mcutil {


    template <>
    ArgumentCard InputCardFactory<deexcitation::Host::Config>::_setCard() {
        ArgumentCard arg_card("DEEX_SETTINGS");
        arg_card.insert<double>("coulomb_penetration_ratio", { 0.5 }, { 0.0 }, { 1.0 });
        arg_card.insert<std::string>("xs_model", { "dostrovsky" });
        return arg_card;
    }

}



namespace deexcitation {
    namespace Host {


        Config::Config() {
            this->_coulomb_penetration_ratio = 0.5f;
            this->_xs_model                  = XS_MODEL::XS_MODEL_DOSTROVSKY;
        }


        Config::Config(mcutil::ArgInput& args) : Config() {
            this->_coulomb_penetration_ratio = (float)args["coulomb_penetration_ratio"].cast<double>()[0];

            std::string xs_model = args["xs_model"].cast<std::string>()[0];

            if (xs_model == "dostrovsky")
                this->_xs_model = XS_MODEL::XS_MODEL_DOSTROVSKY;
            else if (xs_model == "chatterjee")
                this->_xs_model = XS_MODEL::XS_MODEL_CHATTERJEE;
            else if (xs_model == "kalbach")
                this->_xs_model = XS_MODEL::XS_MODEL_KALBACH;
            else 
                mclog::fatal("'xs_model' must be 'dostrovsky', 'chatterjee' or 'kalbach'");

            Config& def = Config::getInstance();
            def = *this;
        }


    }
}