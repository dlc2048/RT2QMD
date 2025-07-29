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
 * @file    module/particles/projectile_interface.hpp
 * @brief   Projectile interface for phyisical quantity definition
 * @author  CM Lee
 * @date    04/03/2024
 */

#pragma once


namespace Define {


    class ProjectileInterface {
    protected:
        double _mass;     //! @brief Mass of projectile [MeV/c^2]
        int    _spin;     //! @brief Spin of projectile, multiplied by 2
        int    _charge;   //! @brief Elementary charge of projectile
    public:
        double mass()   const { return this->_mass; }
        int    spin()   const { return this->_spin; }
        int    charge() const { return this->_charge; }
    };


}