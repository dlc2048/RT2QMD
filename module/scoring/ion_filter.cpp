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
 * @file    module/scoring/ion_filter.cpp
 * @brief   Generic ion ZA filter for scoring
 * @author  CM Lee
 * @date    02/18/2025
 */


#include "ion_filter.hpp"


namespace mcutil {


    template <>
    ArgumentCard InputCardFactory<tally::IonFilter>::_setCard() {
        ArgumentCard arg_card("ION_FILTER");
        arg_card.insert<std::string>("tally", 1);
        arg_card.insert<int>("za");
        return arg_card;
    }


}


namespace tally {


    IonFilter::IonFilter(mcutil::ArgInput& args) {
        this->_where = args["tally"].cast<std::string>()[0];

        std::vector<int> za_list = args["za"].cast<int>();
        for (int za_this : za_list) {
            if (this->_za.find(za_this) != this->_za.end()) {
                std::stringstream ss;
                mclog::fatalListElementAlreadyExist(za_this);
            }
            this->_za.insert(za_this);
        }
    }


}