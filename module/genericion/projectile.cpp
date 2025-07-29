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
 * @file    module/genericion/projectile.cpp
 * @brief   Generic ion projectile definitions
 * @author  CM Lee
 * @date    06/19/2024
 */


#include "projectile.hpp"


namespace genion {


    Projectile::Projectile(int za) {
        const Nucleus::ENSDFTable&  table  = Nucleus::ENSDFTable::getInstance();
        const Nucleus::ENSDFRecord& record = table.get(za);
        // get mass
        double mass   = Nucleus::MassTableHandler::getInstance().getMass(za);
        this->_charge = physics::getZnumberFromZA(za);
        this->_spin   = record.spin();
        this->_mass   = mass;
    }


}