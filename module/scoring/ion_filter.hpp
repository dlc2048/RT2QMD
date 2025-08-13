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
 * @file    module/scoring/ion_filter.hpp
 * @brief   Generic ion ZA filter for scoring
 * @author  CM Lee
 * @date    02/18/2025
 */


#pragma once

#include "tally_interface.hpp"


namespace tally {


    class IonFilter {
    protected:
        std::string   _where;  // target tally
        std::set<int> _za;      // target genion ZA lists


    public:


        IonFilter(mcutil::ArgInput& args);


        const std::string& target() const { return this->_where; }


        const std::set<int>& za() const { return this->_za; }


    };


}