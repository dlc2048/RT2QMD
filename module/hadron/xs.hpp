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
 * @file    module/hadron/xs.hpp
 * @brief   Nucleus-Nucleus cross section (G4ComponentGGNuclNuclXsc)
 * @author  CM Lee
 * @date    06/11/2024
 */


#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "physics/constants.hpp"

#include "nucleus.hpp"


namespace Hadron {


    namespace Host {


        /**
        * @brief Compute coulomb barrier ratio. Two particles cannot collide if ratio is 0
        * @param zp  Charge number of projectile
        * @param zt  Charge number of target
        * @param mp  Mass of projectile [GeV/c^2]
        * @param mt  Mass of target [GeV/c^2]
        * @param eke Kinetic energy of projectile in lab system [GeV]
        * @param md  Minimum distance for touching, radius of target + radius of projectile [fm]
        *
        * @return Coulomb barrier ratio
        */
        double coulombBarrier(int zp, int zt, double mp, double mt, double eke, double md);


        /**
        * @brief Calculate the scattering XS of NN interaction
        * @param tp  Type of projectile (proton if true)
        * @param tt  Type of target (proton if true)
        * @param eke Kinetic energy of projectile in lab system [GeV]
        *
        * @return Cross-section of NN interaction [millibarn]
        */
        double xsNucleonNucleon(bool tp, bool tt, double eke);


        /**
        * @brief Calculate the XS of nuclei-nuclei collision
        * @param zap ZA number of projectile
        * @param zat ZA number of target
        * @param eke Kinetic energy of projectile in lab system [GeV]
        * @param mp  Mass of projectile [GeV/c^2]
        * @param mt  Mass of target [GeV/c^2]
        *
        * @return Cross-section of nuclei-nuclei collision { inelasic, elastic } [millibarn]
        */
        double2 xsNucleiNuclei(int zap, int zat, double eke, double mp, double mt);


    }


}