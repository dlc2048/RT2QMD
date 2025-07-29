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
 * @file    module/particles/define.hpp
 * @brief   RT2 particle definitions
 * @author  CM Lee
 * @date    04/08/2024
 */


#pragma once

#include <vector>
#include <set>
#include <filesystem>
#include <stdexcept>

#include "device/algorithm.hpp"
#include "mclog/logger.hpp"
#include "parser/input.hpp"
#include "parser/parser.hpp"
#include "prompt/env.hpp"
#include "physics/constants.hpp"

#include "projectile_interface.hpp"
#include "define_struct.hpp"


namespace Define {


    constexpr double CUTOFF_BASE_ELECTRON       = 1e-3;    // 1 keV
    constexpr double TRANS_CEIL_ELECTRON        = 1e+3;    // 1 GeV

    constexpr double CUTOFF_BASE_PHOTON         = 1e-3;    // 1 keV
    constexpr double TRANS_CEIL_PHOTON          = 1e+3;    // 1 GeV

    constexpr double CUTOFF_BASE_VACANCY        = 1e-3;    // 1 keV
    constexpr double TRANS_CEIL_VACANCY         = 1e+3;    // 1 GeV

    constexpr double CUTOFF_BASE_DELTA          = 1e-2;    // 10 keV
    constexpr double TRANS_CEIL_DELTA           = 1e+10;   // infinite

    constexpr double CUTOFF_BASE_NEUTRON_LOW    = 1e-11;   // 1e-5 eV

    constexpr double CUTOFF_BASE_NEUTRON_HIGH   = 1e-3;    // thermal
    constexpr double TRANS_CEIL_NEUTRON_HIGH    = 1e+3;    // 1 GeV
    constexpr double TRANS_CEIL_XS_NEUTRON_HIGH = 2e+3;    // 2 GeV (with safety margin)

    constexpr double CUTOFF_BASE_PROTON         = 2.0;     // 2 MeV
    constexpr double TRANS_CEIL_PROTON          = 1e+3;

    constexpr double CUTOFF_BASE_GION           = 2.0;     // 2 MeV/u
    constexpr double TRANS_CEIL_GION            = 1e+3;    // 1 GeV/u (input limit)
    constexpr double TRANS_CEIL_XS_GION         = 2e+3;    // 2 GeV/u (with safety margin)


    enum class EII_MODE {
        EII_OFF,
        EII_KAWRAKOW,
        EII_GRYZINSKI,
        EII_CASNATI,
        EII_KOLBENSTVEDT,
        EII_PENELOPE
    };


    template <typename T>
    class ParticleInterface : 
        public ProjectileInterface,
        public Singleton<T> {
        friend class Singleton<T>;
    protected:
        int          _pid       = INT32_MIN;  //! @brief particle id
        bool         _activated = false;      //! @brief activated?
        double       _t_cutoff  = 1e+20;      //! @brief transport cutoff kinetic energy [MeV]
        double       _p_cutoff  = 1e+20;      //! @brief production cutoff kinetic energy [MeV]
        double       _t_ceil    = 1e+20;      //! @brief transport ceil kinetic energy [MeV]
        size_t       _nbin      = 0x0u;       //! @brief Number of log-log interpolation bin
        std::string  _library   = "";         //! @brief library name


        ParticleInterface() {}


        void _readHeader(mcutil::ArgInput& args);


    public:
        int    pid();
        bool   activated();
        double transportCutoff();
        double productionCutoff();
        double transportCeil();
        void   setTransportCeil(double ceil);
        void   setTransportCutoff(double floor);
        void   setProductionCutoff(double floor);
        void   setActivation(bool option);
        size_t  nbin() { return this->_nbin; }
        double2 llx()  { return mcutil::LogLogCoeff(this->transportCutoff(), this->transportCeil(), this->nbin()).llx(); }
        std::string library();
        mcutil::ArgumentCard controlCard();
    };


    class Electron : public ParticleInterface<Electron> {
        friend class ParticleInterface<Electron>;
        friend class Singleton<Electron>;
    private:
        BREM_CORRECTION_METHOD _brem_corr;     // bremsstrahlung xs correction
        BREM_XS_METHOD         _brem_xs;       // bremsstrahlung xs generating method
        BREM_ANGLE_METHOD      _brem_angle;    // bremsstrahlung x-ray angle sampling method
        EII_MODE               _eii_mode;      // Electron Impact Ionization (EII) mode
        size_t _n_brem_split;  // bremsstrahlung x-ray splitting multiplier
        bool   _print_xs;      // print cross section & stopping power data
        bool   _do_spin;       // spin effect
        bool   _presta2;       // use PRESTA-II if true, PRESTA-I elsewhere
        bool   _ie_fudge;      // use SLAC-265 fudge for ionization energy
        double _fudgems;       // EGSnrc FUDGEMS
        double _smax;          // maximum step size of electron
        double _estepe;        // maximum fractional energy loss per step
        double _ximax;         // maximum 1st elastic scattering moment per step


        Electron();  // singleton

        
    public:


        Electron(mcutil::ArgInput& args);  // context


        BREM_CORRECTION_METHOD bremCorr()  { return this->_brem_corr; }
        BREM_XS_METHOD         bremXS()    { return this->_brem_xs; }
        BREM_ANGLE_METHOD      bremAngle() { return this->_brem_angle; }
        EII_MODE               eiiMode()   { return this->_eii_mode; }


        bool   doSpin()     { return this->_do_spin; }
        bool   usePresta2() { return this->_presta2; }
        bool   ieFudge()    { return this->_ie_fudge; }
        bool   printXS()    { return this->_print_xs; }
        bool   doEII()      { return this->_eii_mode != EII_MODE::EII_OFF; }
        size_t nbin()       { return this->_nbin;}
        double fudgems()    { return this->_fudgems; }
        double smax()       { return this->_smax; }
        double estepe()     { return this->_estepe; }
        double ximax()      { return this->_ximax; }
        size_t nBremSplit() { return this->_n_brem_split; }


        bool   hasElementDependency() { return this->doEII(); }
        void   turnOffBremSplit()     { this->_n_brem_split = 1; }


    };


    class Photon : public ParticleInterface<Photon> {
        friend class ParticleInterface<Photon>;
        friend class Singleton<Photon>;
    private:
        COMPTON_METHOD _compton_mode;  // Compton method
        bool _use_nrc_pair;  // Use NRC alias table for pair production if true
        bool _do_rayleigh;   // Rayleigh activation flag
        bool _simple_photo;  // Simplified photoelectric flag
        bool _use_sauter;    // Sauter distribution flag
        bool _print_xs;      // Print XS table

        Photon();


    public:


        Photon(mcutil::ArgInput& args);


        COMPTON_METHOD comptMode() { return this->_compton_mode; }


        bool nrcPair()     { return this->_use_nrc_pair; }
        bool doRayleigh()  { return this->_do_rayleigh; }
        bool simplePhoto() { return this->_simple_photo; }
        bool useSauter()   { return this->_use_sauter; }
        bool printXS()     { return this->_print_xs;}

        bool hasElementDependency() { return !(simplePhoto() && comptMode() == COMPTON_METHOD::COMPTON_SIMPLE); }

    };


    class Positron : public ParticleInterface<Positron> {
        friend class ParticleInterface<Positron>;
        friend class Singleton<Positron>;
    private:


        Positron();


    public:


        Positron(mcutil::ArgInput& args);


    };


    class Vacancy : public ParticleInterface<Vacancy> {
        friend class ParticleInterface<Vacancy>;
        friend class Singleton<Vacancy>;
    private:
        bool _local_deposit;     // Relaxation local deposit flag
        bool _print_detail;  // option for printing transition table in output file


        Vacancy();


    public:


        Vacancy(mcutil::ArgInput& args);


        bool localDeposit() { return this->_local_deposit; }
        bool printDetail()  { return this->_print_detail; }


    };


    class Neutron : public ParticleInterface<Neutron> {
        friend class ParticleInterface<Neutron>;
        friend class Singleton<Neutron>;
    private:
        std::vector<float> _ngroup;
        std::vector<float> _ggroup;

        bool _print_xs;      // Print XS table


        std::string _library_high;   // High energy neutron XS library


        Neutron();


    public:


        Neutron(mcutil::ArgInput& args);


        const std::vector<float>& ngroup() { return this->_ngroup; }
        const std::vector<float>& ggroup() { return this->_ggroup; }
        void setNeutronGroup(const std::vector<float>& ng) { this->_ngroup = ng; }
        void setGammaGroup(const std::vector<float>& gg)   { this->_ggroup = gg; }
        bool printXS() { return this->_print_xs; }


        const std::string& libraryHigh() { return this->_library_high; }


    };


    class DeltaRay : public ParticleInterface<DeltaRay> {
        friend class ParticleInterface<DeltaRay>;
        friend class Singleton<DeltaRay>;
    private:
        bool _free_scat;  // Free electron scattering flag;


        DeltaRay();


    public:


        DeltaRay(mcutil::ArgInput& args);


        bool freeScat() { return this->_free_scat; }


    };


    class GenericIon : public ParticleInterface<GenericIon> {
        friend class ParticleInterface<GenericIon>;
        friend class Singleton<GenericIon>;
    private:
        bool   _lat_disp;   // Urban lateral displacment flag
        bool   _lat_alg96;  // Urban lateral displacment, gaussian distribution flag
        bool   _loss_fluc;  // Energy loss straggling flag
        bool   _print_xs;   // Print XS table
        double _smax;       // Maximum step size [cm]
        double _estepe;     // Maximum energy loss per one step

        std::set<int> _xs_proj_list;


        GenericIon();


    public:


        GenericIon(mcutil::ArgInput& args);


        bool   latDisp()    { return this->_lat_disp; }
        bool   latAlg96()   { return this->_lat_alg96; }
        bool   straggling() { return this->_loss_fluc; }
        bool   printXS()    { return this->_print_xs; }
        double smax()       { return this->_smax; }
        double estepe()     { return this->_estepe; }

        const std::set<int>& xsProjectileList() { return this->_xs_proj_list; }


    };


    class IonInelastic :
        public Singleton<IonInelastic> {
        friend class Singleton<IonInelastic>;
    private:
        ION_INELASTIC_METHOD_HIGH _mode_high;
        ION_INELASTIC_METHOD_LOW  _mode_low;
        int    _pid;
        double _evap_cutoff;
        // evaporation
        bool   _activate_fission;
        bool   _use_exact_level;


        IonInelastic();


    public:


        IonInelastic(mcutil::ArgInput& args);


        ION_INELASTIC_METHOD_HIGH modeHigh()    { return this->_mode_high; }
        ION_INELASTIC_METHOD_LOW  modeLow()     { return this->_mode_low; }
        int     pid()                  { return this->_pid; }
        double  evaporationCutoff()    { return this->_evap_cutoff; }
        bool    activateFission()      { return this->_activate_fission; }
        bool    useExactLevel()        { return this->_use_exact_level; }


        void setModeHigh(ION_INELASTIC_METHOD_HIGH mode) { this->_mode_high = mode; }
        void setModeLow (ION_INELASTIC_METHOD_LOW  mode) { this->_mode_low  = mode; }


    };


    void initializeParticleStaticArgs(
        mcutil::ArgumentCard& card, 
        const std::string& library,
        double cutoff_default, 
        double cutoff_minimum,
        double cutoff_maximum
    );

}


#include "define.tpp"
