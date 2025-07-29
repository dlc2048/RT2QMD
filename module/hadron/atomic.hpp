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
 * @file    module/hadron/atomic.hpp
 * @brief   ENDF atomic mass table
 * @author  CM Lee
 * @date    01/31/2025
 */

#pragma once

#include <filesystem>
#include <map>

#include "singleton/singleton.hpp"
#include "fortran/fortran.hpp"
#include "prompt/env.hpp"

#include "particles/define.hpp"

#include "nucleus.hpp"


namespace Atomic {


    inline const std::filesystem::path HOME = std::filesystem::path("resource/neutron");  // Using ENDF mass table


    class MassTable : public Singleton<MassTable> {
        friend class Singleton<MassTable>;
    private:
        static const std::filesystem::path _mass_file;  //! @brief Filename of ENDF atomic mass table
        std::map<int, double> _table;  //! @brief Atomic mass table


        MassTable();


    public:


        double getMass(int za);


    };


}