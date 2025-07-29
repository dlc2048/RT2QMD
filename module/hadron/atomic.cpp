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
 * @file    module/hadron/atomic.cpp
 * @brief   ENDF atomic mass table
 * @author  CM Lee
 * @date    01/31/2025
 */


#include "atomic.hpp"


namespace Atomic {


    const std::filesystem::path MassTable::_mass_file = std::filesystem::path("mass_table.bin");


    MassTable::MassTable() {
        namespace fp = std::filesystem;
        std::string home = mcutil::getMCRT2HomePath();
        if (home.empty())
            mclog::fatal("Environment variable 'MCRT2_HOME' is missing");
        fp::path file_name(home);
        fp::path lib(Define::Neutron::getInstance().library());
        file_name = file_name / HOME / lib / this->_mass_file;

        mcutil::FortranIfstream data(file_name.string());

        std::vector<int>   za   = data.read<int>();
        std::vector<float> mass = data.read<float>();

        if (za.size() != mass.size()) {
            std::stringstream ss;
            ss << "Mass table '" << file_name.string() << "' is corrupted";
            mclog::fatal(ss);
        }

        for (size_t i = 0; i < za.size(); ++i)
            this->_table.insert({ za[i], (double)mass[i] });
    }


    double MassTable::getMass(int za) {
        auto table_iter = this->_table.find(za);
        if (table_iter == this->_table.end()) {
            std::stringstream ss;
            ss << "No such isotope '" << za << "' in library '" << Define::Neutron::getInstance().library() << "'";
            mclog::warning(ss);
            double mass = Nucleus::MassTableHandler::getInstance().getMass(za) / constants::ATOMIC_MASS_UNIT;
            this->_table.insert({ za, mass });
            return mass;
        }
        else
            return table_iter->second;
    }


}