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
 * @file    module/deexcitation/handler.cpp
 * @brief   De-excitation global variable, data, memory and device address handler
 * @author  CM Lee
 * @date    07/12/2024
 */


#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "handler.hpp"


namespace deexcitation {


    void DeviceMemoryHandler::_setFissionData() {
        // Cameron corrections
        this->_cameron_correction = std::make_unique<fission::CameronCorrection>();

        CUDA_CHECK(fission::setCameronSpinPairingCorrections(
            this->_cameron_correction->ptrSpinPairingProton(),
            this->_cameron_correction->ptrSpinPairingNeutron()
        ));
        CUDA_CHECK(fission::setCameronPairingCorrections(
            this->_cameron_correction->ptrPairingProton(),
            this->_cameron_correction->ptrPairingNeutron()
        ));
        CUDA_CHECK(fission::setCameronSpinCorrections(
            this->_cameron_correction->ptrSpinProton(),
            this->_cameron_correction->ptrSpinNeutron()
        ));
    }


    void DeviceMemoryHandler::_setMassData() {
        // Emitted particle info
        // mass 
        Nucleus::MassTableHandler& host_mass_table 
            = Nucleus::MassTableHandler::getInstance();

        std::vector<double> m;
        std::vector<double> m2;
        m.push_back(0.0);
        m.push_back(0.0);
        m.push_back(constants::MASS_NEUTRON);
        m.push_back(constants::MASS_PROTON);
        m.push_back(host_mass_table.getMass(1002));  // deuteron
        m.push_back(host_mass_table.getMass(1003));  // triton
        m.push_back(host_mass_table.getMass(2003));  // He3
        m.push_back(host_mass_table.getMass(2004));  // He4
        m.push_back(constants::MASS_NEUTRON * 2.0);  // 2n
        m.push_back(constants::MASS_PROTON  * 2.0);  // 2p

        for (double mass : m)
            m2.push_back(mass * mass);

        std::vector<float> m_float  = mcutil::cvtVectorDoubleToFloat(m);
        std::vector<float> m2_float = mcutil::cvtVectorDoubleToFloat(m2);

        mcutil::DeviceVectorHelper m_vec(m_float);
        mcutil::DeviceVectorHelper m2_vec(m2_float);

        this->_memoryUsageAppend(m_vec.memoryUsage());
        this->_memoryUsageAppend(m2_vec.memoryUsage());

        this->_dev_m  = m_vec.address();
        this->_dev_m2 = m2_vec.address();

        CUDA_CHECK(setEmittedParticleMass(this->_dev_m, this->_dev_m2));
    }


    void DeviceMemoryHandler::_setCoulombBarrierData() {
        // Coulomb radius
        this->_coulomb_barrier = std::make_unique<CoulombBarrier>();

        CUDA_CHECK(setCoulombBarrierRadius(this->_coulomb_barrier->ptrCoulombRadius()));

        // Coulomb ratio
        Host::Config& config = Host::Config::getInstance();

        CUDA_CHECK(setCoulombRatio(config.coulombPenetrationRatio()));

        // coulomb rho
        std::vector<double> crho;

        crho.push_back(0.0);
        crho.push_back(0.0);
        crho.push_back(this->_coulomb_barrier->coulombBarrierRadius(0, 1));  // neutron
        crho.push_back(this->_coulomb_barrier->coulombBarrierRadius(1, 1));  // proton
        crho.push_back(this->_coulomb_barrier->coulombBarrierRadius(1, 2));  // deuteron
        crho.push_back(this->_coulomb_barrier->coulombBarrierRadius(1, 3));  // triton
        crho.push_back(this->_coulomb_barrier->coulombBarrierRadius(2, 3));  // He3
        crho.push_back(this->_coulomb_barrier->coulombBarrierRadius(2, 4));  // He4
        crho.push_back(0.0);
        crho.push_back(0.0);

        for (double& cr : crho)
            cr *= 0.6;

        // memory
        std::vector<float> crho_flaot = mcutil::cvtVectorDoubleToFloat(crho);

        mcutil::DeviceVectorHelper crho_vec(crho_flaot);

        this->_memoryUsageAppend(crho_vec.memoryUsage());

        this->_dev_crho = crho_vec.address();

        CUDA_CHECK(setEmittedParticleCBRho(this->_dev_crho));
    }


    void DeviceMemoryHandler::_setChatterjeeXSData() {
        this->_chatterjee_xs = std::make_unique<ChatterjeeCrossSection>();
    }


    void DeviceMemoryHandler::_setKalbachXSData() {
        this->_kalbach_xs = std::make_unique<KalbachCrossSection>();
    }


    DeviceMemoryHandler::DeviceMemoryHandler() {
        mclog::debug("Initialize nuclear de-excitation data ...");

        this->_setFissionData();

        CUDA_CHECK(setFissionFlag(Define::IonInelastic::getInstance().activateFission()));

        this->_setMassData();
        this->_setCoulombBarrierData();

        if (Host::Config::getInstance().crossSectionModel() == XS_MODEL::XS_MODEL_CHATTERJEE) {
            this->_setChatterjeeXSData();
            this->_setKalbachXSData();
        }
        else if (Host::Config::getInstance().crossSectionModel() == XS_MODEL::XS_MODEL_KALBACH) {
            this->_setKalbachXSData();
        }

        // nucleus symbol
        mclog::debug("Link de-excitation device symbol ...");
        CUDA_CHECK(setStableTable(Nucleus::ENSDFTable::getInstance().ptrLongLivedNucleiTable()));
        CUDA_CHECK(setMassTableHandle(Nucleus::MassTableHandler::getInstance().deviceptr()));
    }


    DeviceMemoryHandler::~DeviceMemoryHandler() {
        mclog::debug("Destroy device memory of nuclear de-excitation data ...");
        mcutil::DeviceVectorHelper(this->_dev_m).free();
        mcutil::DeviceVectorHelper(this->_dev_m2).free();
        mcutil::DeviceVectorHelper(this->_dev_crho).free();
    }


    void DeviceMemoryHandler::summary() const {

        Define::IonInelastic& ie_def = Define::IonInelastic::getInstance();
        Host::Config&         config = Host::Config::getInstance();

        double bytes_to_mib = 1.0 / (double)mcutil::MEMSIZE_MIB;

        double   cp_ratio = config.coulombPenetrationRatio();
        XS_MODEL model    = config.crossSectionModel();

        std::string model_str;
        if (model == XS_MODEL::XS_MODEL_DOSTROVSKY) {
            model_str = "Dostrovsky";
            cp_ratio  = 0.0;
        }
        else if (model == XS_MODEL::XS_MODEL_CHATTERJEE)
            model_str = "Chatterjee";
        else if (model == XS_MODEL::XS_MODEL_KALBACH)
            model_str = "Kalbach";
        else
            assert(false);

        mclog::info("*** Nuclear De-Excitation Summaries ***");
        mclog::printVar("Evaporation cutoff    ", ie_def.evaporationCutoff(), "s");
        mclog::printVar("Fission branch        ", ie_def.activateFission() ? "On" : "Off");
        mclog::printVar("Inverse XS            ", model_str);
        mclog::printVar("CB penetration ratio  ", cp_ratio);
        mclog::printVar("Memory usage          ", this->memoryUsage() * bytes_to_mib, "MiB");
        mclog::print("");
    }


}