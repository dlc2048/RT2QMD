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
 * @file    module/genericion/auxiliary.cpp
 * @brief   Auxiliary tables for generic ion
 * @author  CM Lee
 * @date    06/21/2024
 */


#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "auxiliary.hpp"


namespace genion {


    IsoProjectileTable::IsoProjectileTable() : 
        _table_size    (0x0),
        _iso_offset_dev(nullptr),
        _list_za_dev   (nullptr), 
        _list_m_dev    (nullptr),
        _list_mr_dev   (nullptr) {

        Nucleus::ENSDFTable&       ensdf_table = Nucleus::ENSDFTable::getInstance();
        Nucleus::MassTableHandler& mass_table  = Nucleus::MassTableHandler::getInstance();


        deexcitation::UnstableBreakUp& breakup_table
            = deexcitation::UnstableBreakUp::getInstance();

        double cutoff = Define::IonInelastic::getInstance().evaporationCutoff();

        // determine table dimension
        for (int i = 0; i < GENION_MAX_Z; ++i) {  // Z=1-8
            int z = i + 1;
            // get isotops list
            int    za_ref = Hadron::Projectile::REFERENCE_ZA[i];
            int    a_ref  = physics::getAnumberFromZA(za_ref);
            double mu_ref = mass_table.getMass(za_ref) / (double)physics::getAnumberFromZA(za_ref);
            std::vector<IsoProjectile> proj_list;
            for (int za : ensdf_table.isotopes(z)) {
                int    z                  = physics::getZnumberFromZA(za);
                int    a                  = physics::getAnumberFromZA(za);
                double life_time          = ensdf_table.get(za).lifeTime();
                bool   breakup_applicable = breakup_table.isApplicable(z, a);

                if (a > GENION_MAX_A)
                    continue;

                if ((life_time >= cutoff || life_time < 0.0) || 
                    (life_time  < cutoff && !breakup_applicable)) {
                    int z = physics::getZnumberFromZA(za);
                    int a = physics::getAnumberFromZA(za);
                    double mu_this = mass_table.getMass(za) / (double)a;
                    IsoProjectile proj;
                    proj.z    = z;
                    proj.a    = a;
                    proj.mu   = mu_this;
                    proj.mr   = mu_this / mu_ref * (double)a / (double)a_ref;
                    proj.spin = ensdf_table.get(za).spin();
                    proj_list.push_back(proj);
                }
            }
            this->_iso_proj[i] = proj_list;
        }
        this->_initDeviceMemory();
    }


    void IsoProjectileTable::_initDeviceMemory() {
        std::vector<int>    offset;
        std::vector<uchar2> za;
        std::vector<float>  m;
        std::vector<float>  mu;
        std::vector<float>  mr;
        std::vector<int>    spin;
        for (int i = 0; i < GENION_MAX_Z; ++i) {
            this->_offset_host[i] = (int)za.size();
            offset.push_back((int)za.size());
            for (const IsoProjectile& proj : this->_iso_proj[i]) {
                uchar2 za_this;
                za_this.x = proj.z;
                za_this.y = proj.a;
                za.push_back(za_this);
                m.push_back((float)(proj.mu * (double)proj.a));
                mu.push_back((float)proj.mu);
                mr.push_back((float)proj.mr);
                spin.push_back(proj.spin);
            }
        }
        this->_table_size = (int)za.size();
        mcutil::DeviceVectorHelper offset_vec(offset);
        mcutil::DeviceVectorHelper za_vec(za);
        mcutil::DeviceVectorHelper m_vec(m);
        mcutil::DeviceVectorHelper mu_vec(mu);
        mcutil::DeviceVectorHelper mr_vec(mr);
        mcutil::DeviceVectorHelper spin_vec(spin);

        this->_memoryUsageAppend(offset_vec.memoryUsage());
        this->_memoryUsageAppend(za_vec.memoryUsage());
        this->_memoryUsageAppend(m_vec.memoryUsage());
        this->_memoryUsageAppend(mu_vec.memoryUsage());
        this->_memoryUsageAppend(mr_vec.memoryUsage());
        this->_memoryUsageAppend(spin_vec.memoryUsage());

        this->_iso_offset_dev = offset_vec.address();
        this->_list_za_dev    = za_vec.address();
        this->_list_m_dev     = m_vec.address();
        this->_list_mu_dev    = mu_vec.address();
        this->_list_mr_dev    = mr_vec.address();
        this->_list_spin_dev  = spin_vec.address();

        // generic ion
#ifndef RT2QMD_STANDALONE
        CUDA_CHECK(PROJSOA1D::setOffset(this->_iso_offset_dev));
        CUDA_CHECK(PROJSOA1D::setTable(
            this->_list_za_dev,
            this->_list_m_dev,
            this->_list_mu_dev,
            this->_list_mr_dev,
            this->_list_spin_dev
        ));
#endif
        // hadron
        CUDA_CHECK(Hadron::PROJSOA1D::setTableSize(this->_table_size));
        CUDA_CHECK(Hadron::PROJSOA1D::setOffset(this->_iso_offset_dev));
        CUDA_CHECK(Hadron::PROJSOA1D::setTable(
            this->_list_za_dev,
            this->_list_m_dev,
            this->_list_mu_dev,
            this->_list_mr_dev,
            this->_list_spin_dev
        ));
    }


    IsoProjectileTable::~IsoProjectileTable() {
        /*
        mcutil::DeviceVectorHelper(this->_iso_offset_dev).free();
        mcutil::DeviceVectorHelper(this->_list_za_dev).free();
        mcutil::DeviceVectorHelper(this->_list_m_dev).free();
        mcutil::DeviceVectorHelper(this->_list_mr_dev).free();
        mcutil::DeviceVectorHelper(this->_list_spin_dev).free();
        */
    }


    bool IsoProjectileTable::exist(int za) const {
        int z = physics::getZnumberFromZA(za);
        int a = physics::getAnumberFromZA(za);
        if (z <= 0 || z > GENION_MAX_Z)
            return false;
        for (const IsoProjectile& proj : this->_iso_proj[z - 1]) {
            if (a == proj.a)
                return true;
        }
        return false;
    }


    std::vector<int> IsoProjectileTable::listProjectileZA() const {
        std::vector<int> za_list;
        for (int z = 1; z <= Hadron::Projectile::TABLE_MAX_Z; ++z)
            for (const IsoProjectile& proj : this->_iso_proj[z - 1])
                za_list.push_back((int)proj.z * 1000 + (int)proj.a);
        return za_list;
    }


    std::vector<double> IsoProjectileTable::listProjectileMass() const {
        std::vector<double> mass_list;
        for (int z = 1; z <= Hadron::Projectile::TABLE_MAX_Z; ++z)
            for (const IsoProjectile& proj : this->_iso_proj[z - 1])
                mass_list.push_back(proj.mass());
        return mass_list;
    }


    const IsoProjectile& IsoProjectileTable::refProjectile(int z) const {
        int za_ref = Hadron::Projectile::REFERENCE_ZA[z - 1];
        int a_ref  = physics::getAnumberFromZA(za_ref);
        for (const IsoProjectile& proj : this->_iso_proj[z - 1]) {
            if (a_ref == proj.a)
                return proj;
        }
        assert(false);
        return this->_iso_proj[0][0];
    }


    const IsoProjectile& IsoProjectileTable::projectile(int z, int a) const {
        for (const IsoProjectile& proj : this->_iso_proj[z - 1]) {
            if (a == proj.a)
                return proj;
        }
        assert(false);
        return this->_iso_proj[0][0];
    }



    int IsoProjectileTable::index(int za) const {
        if (!this->exist(za))
            return -1;
        int z   = physics::getZnumberFromZA(za);
        int a   = physics::getAnumberFromZA(za);
        int idx = this->offset(z);

        for (int i = 0; i < this->_iso_proj[z - 1].size(); ++i) {
            const IsoProjectile& proj = this->_iso_proj[z - 1][i];
            if (a == proj.a) {
                idx += i;
                break;
            }
        }
        return idx;
    }


}