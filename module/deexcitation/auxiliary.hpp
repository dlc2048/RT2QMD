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
 * @file    module/deexcitation/auxiliary.hpp
 * @brief   Auxiliary tables for deexcitation
 * @author  CM Lee
 * @date    07/12/2024
 */


#pragma once

#include <vector>
#include <regex>

#include "fortran/fortran.hpp"
#include "singleton/singleton.hpp"

#include "hadron/nucleus.hpp"
#include "particles/define.hpp"

#include "auxiliary.cuh"


namespace deexcitation {


    inline const std::filesystem::path HOME 
        = std::filesystem::path("resource") / std::filesystem::path("deexcitation");


    class CoulombBarrier : public mcutil::DeviceMemoryHandlerInterface {
    private:
        static const std::filesystem::path _cr_file;

        // value (host)
        std::vector<float> _coulomb_radius;

        // value (dev)
        float* _coulomb_radius_dev;


    public:


        CoulombBarrier();


        ~CoulombBarrier();


        double coulombBarrierRadius(int z, int a) const;


        float* ptrCoulombRadius() const { return this->_coulomb_radius_dev; }


    };


    class UnstableBreakUp : public Singleton<UnstableBreakUp> {
        friend class Singleton<UnstableBreakUp>;
    private:
        std::vector<double> _m;
        std::vector<double> _m2;
        std::vector<int>    _z;
        std::vector<int>    _a;


        UnstableBreakUp();


    public:


        bool isApplicable(int z, int a);


    };


    class ChatterjeeCrossSection {
    private:
        static const std::filesystem::path _cj_file;
    public:


        ChatterjeeCrossSection();


    };


    class KalbachCrossSection {
    private:
        static const std::filesystem::path _kb_file;
    public:


        KalbachCrossSection();


    };


    namespace fission {


        class CameronCorrection : public mcutil::DeviceMemoryHandlerInterface {
        private:
            static const std::filesystem::path _sp_file;
            static const std::filesystem::path _spin_file;
            static const std::filesystem::path _pair_file;

            // value (host)
            std::vector<float> _sp_correction_p;
            std::vector<float> _sp_correction_n;
            std::vector<float> _spin_correction_p;
            std::vector<float> _spin_correction_n;
            std::vector<float> _pair_correction_p;
            std::vector<float> _pair_correction_n;

            // device
            float* _sp_correction_p_dev;
            float* _sp_correction_n_dev;
            float* _spin_correction_p_dev;
            float* _spin_correction_n_dev;
            float* _pair_correction_p_dev;
            float* _pair_correction_n_dev;


        public:


            CameronCorrection();


            ~CameronCorrection();


            float* ptrSpinPairingProton()  const { return this->_sp_correction_p_dev; }
            float* ptrSpinPairingNeutron() const { return this->_sp_correction_n_dev; }
            float* ptrSpinProton()         const { return this->_spin_correction_p_dev; }
            float* ptrSpinNeutron()        const { return this->_spin_correction_n_dev; }
            float* ptrPairingProton()      const { return this->_pair_correction_p_dev; }
            float* ptrPairingNeutron()     const { return this->_pair_correction_n_dev; }


        };


    }


    namespace photon {


        struct Transition {
            int    daughter_level;
            double transition_energy;
            double relative_intensity;
            int    multipolarity_id;
            double multipolarity_ratio;
            double ic_alpha;
            double ic_ratio[10];  // Internal conversion K, L1, L2, L3, M1, M2, M3, M4, M5, free


            Transition();


        };


        class NuclearLevel {
        private:
            double _level_energy;
            double _level_half_life;
            int    _spin;
            std::vector<Transition> _transition;
        public:


            NuclearLevel();


            NuclearLevel(double exc_energy, double half_life, int spin);


            void appendTransitionMode(const Transition& mode);


            bool excitationEnergy() const { return this->_level_energy; }
            bool stable()           const { return this->_level_half_life < 0.0; }
            bool halfLife()         const { return this->_level_half_life; }
            


        };


        class PhotonEvaporationTable : public Singleton<PhotonEvaporationTable> {
            friend class Singleton<PhotonEvaporationTable>;
        private:
            static const std::filesystem::path _library;
            // table, host
            std::map<int, std::vector<NuclearLevel>> _table;


            PhotonEvaporationTable() {}


            const std::vector<NuclearLevel>& _loadLevelData(int za);


        public:


            const NuclearLevel& get(int za, int level);


            void readAll();


            void clear();


        };


        std::stringstream& operator>>(std::stringstream& sstream, Transition& q);


        std::stringstream& operator>>(std::stringstream& sstream, NuclearLevel& q);


    }


}
