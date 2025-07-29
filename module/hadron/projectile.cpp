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
 * @file    module/hadron/projectile.cpp
 * @brief   Aligned mass table for heavy ion projectile
 * @author  CM Lee
 * @date    04/02/2024
 */


#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "projectile.hpp"


namespace Hadron {
    namespace Projectile {


        MassRatioTable::MassRatioTable() {
            Nucleus::MassTableHandler& nuc_mass = Nucleus::MassTableHandler::getInstance();
            // host side
            this->_table_host = std::vector<double>(TABLE_MAX_Z * TABLE_MAX_A);
            for (int i = 0; i < TABLE_MAX_Z; ++i) {
                int    z        = i + 1;
                int    za_ref   = REFERENCE_ZA[i];
                double mass_ref = z == 1 ? (double)constants::MASS_PROTON : nuc_mass.getMass(za_ref);
                for (int j = 0; j < TABLE_MAX_A; ++j) {
                    int    a    = j + 1;
                    int    za   = 1000 * z + a;
                    double mass = z == 1 && a == 1 ? (double)constants::MASS_PROTON : nuc_mass.getMass(za);
                    this->_table_host[i * TABLE_MAX_A + j] = mass / mass_ref;
                }
            }
            // device side
            std::vector<float> ftable = mcutil::cvtVectorDoubleToFloat(this->_table_host);
            this->_memoryUsageAppend(mcutil::cudaMemcpyVectorToDevice(ftable, &this->_table_dev));
        }


        MassRatioTable::~MassRatioTable() {
            /*
            CUDA_CHECK(cudaFree(this->_table_dev));
            */
        }


        CUdeviceptr MassRatioTable::deviceptr() {
            return reinterpret_cast<CUdeviceptr>(this->_table_dev);
        }


        double MassRatioTable::getRatio(int za) const {
            int z = physics::getZnumberFromZA(za);
            int a = physics::getAnumberFromZA(za);
            assert(z <= TABLE_MAX_Z && z >= 1);
            assert(a <= TABLE_MAX_A && a >= 1);
            int idx = (z - 1) * TABLE_MAX_A + (a - 1);
            return this->_table_host[idx];
        }


        int MassRatioTable::referenceZA(int za) const {
            int z = physics::getZnumberFromZA(za);
            assert(z <= TABLE_MAX_Z && z >= 1);
            return REFERENCE_ZA[z - 1];
        }


        int MassRatioTable::referenceSpin(int za) const {
            int z = physics::getZnumberFromZA(za);
            assert(z <= TABLE_MAX_Z && z >= 1);
            return REFERENCE_SPIN[z - 1];
        }


        ProjectileRef::ProjectileRef(int za) {
            int z = physics::getZnumberFromZA(za);
            assert(z <= TABLE_MAX_Z && z >= 1);
            this->_charge = z;
            this->_spin   = MassRatioTable::getInstance().referenceSpin(za);
            int za_ref    = MassRatioTable::getInstance().referenceZA(za);
            int a         = physics::getAnumberFromZA(za_ref);
            this->_mass   = z == 1 && a == 1 
                ? (double)constants::MASS_PROTON 
                : Nucleus::MassTableHandler::getInstance().getMass(za_ref);
        }


    }
}