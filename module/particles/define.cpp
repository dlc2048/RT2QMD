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
 * @file    module/particles/define.cpp
 * @brief   RT2 particle definitions
 * @author  CM Lee
 * @date    04/08/2024
 */


#include "define.hpp"


namespace mcutil {


    template <>
    ArgumentCard InputCardFactory<Define::Electron>::_setCard() {
        ArgumentCard arg_card("ELECTRON");
        Define::initializeParticleStaticArgs(
            arg_card, "pegsless", 1e-1,
            Define::CUTOFF_BASE_ELECTRON,
            Define::TRANS_CEIL_ELECTRON
        );
        arg_card.insert<bool>("do_spin",     std::vector<bool>{ true });
        arg_card.insert<bool>("use_presta2", std::vector<bool>{ true });
        arg_card.insert<bool>("ie_fudge",    std::vector<bool>{ true });
        arg_card.insert<bool>("print_xs",    std::vector<bool>{ false });
        arg_card.insert<size_t>("n_brem_split", { 5 },     { 2 },    { 100 });
        arg_card.insert<size_t>("nbin",         { 1000 },  { 100 },  { 10000 });
        arg_card.insert<double>("fudgems",      { 1.e0 },  { 1e-5 }, { 1.e0 });
        arg_card.insert<double>("smax",         { 1e+10 }, { 1e-5 }, { 1e+10 });
        arg_card.insert<double>("estepe",       { 0.2 },   { 1e-5 }, { 0.25 });
        arg_card.insert<double>("ximax",        { 0.5 },   { 1e-5 }, { 1.0 });
        arg_card.insert<std::string>("brem_corr",  { "icru" });
        arg_card.insert<std::string>("brem_xs",    { "nrc" });
        arg_card.insert<std::string>("brem_angle", { "simple" });
        arg_card.insert<std::string>("eii_mode",   { "off" });
        return arg_card;
    }


    template <>
    ArgumentCard InputCardFactory<Define::Photon>::_setCard() {
        ArgumentCard arg_card("PHOTON");
        Define::initializeParticleStaticArgs(
            arg_card, "epdl97", 1e-1,
            Define::CUTOFF_BASE_PHOTON,
            Define::TRANS_CEIL_PHOTON
        );
        arg_card.insert<std::string>("compton_mode", { "simple" });
        arg_card.insert<bool>("use_nrc_pair", std::vector<bool>{ true  });
        arg_card.insert<bool>("do_rayleigh",  std::vector<bool>{ true  });
        arg_card.insert<bool>("simple_photo", std::vector<bool>{ false });
        arg_card.insert<bool>("sauter",       std::vector<bool>{ true  });
        arg_card.insert<bool>("print_xs",     std::vector<bool>{ false });
        arg_card.insert<size_t>("nbin", { 1000 }, { 100 }, { 10000 });
        return arg_card;
    }


    template <>
    ArgumentCard InputCardFactory<Define::Positron>::_setCard() {
        ArgumentCard arg_card("POSITRON");
        Define::initializeParticleStaticArgs(
            arg_card, "pegsless", 1e-1,
            Define::CUTOFF_BASE_ELECTRON,
            Define::TRANS_CEIL_ELECTRON
        );
        return arg_card;
    }


    template <>
    ArgumentCard InputCardFactory<Define::Vacancy>::_setCard() {
        ArgumentCard arg_card("VACANCY");
        Define::initializeParticleStaticArgs(
            arg_card, "eadl", 1e-2,
            Define::CUTOFF_BASE_VACANCY, 
            Define::TRANS_CEIL_VACANCY
        );
        arg_card.insert<bool>("local_deposit", std::vector<bool>{ false });
        arg_card.insert<bool>("print_detail",  std::vector<bool>{ false });
        return arg_card;
    }


    template <>
    ArgumentCard InputCardFactory<Define::Neutron>::_setCard() {
        ArgumentCard arg_card("NEUTRON");
        Define::initializeParticleStaticArgs(
            arg_card, "endf8R0_260", 1e-11,
            Define::CUTOFF_BASE_NEUTRON_LOW, 
            Define::TRANS_CEIL_NEUTRON_HIGH
        );
        arg_card.insert<bool>("print_xs", std::vector<bool>{ false });
        arg_card.insert<size_t>("nbin", { 1000 }, { 100 }, { 10000 });
        return arg_card;
    }


    template <>
    ArgumentCard InputCardFactory<Define::DeltaRay>::_setCard() {
        ArgumentCard arg_card("DELTA");
        Define::initializeParticleStaticArgs(
            arg_card, "", 0.3,
            Define::CUTOFF_BASE_DELTA,
            Define::TRANS_CEIL_DELTA
        );
        arg_card.insert<bool>("free_scat", std::vector<bool>{ false });
        return arg_card;
    }


    template <>
    ArgumentCard InputCardFactory<Define::GenericIon>::_setCard() {
        ArgumentCard arg_card("GENERIC_ION");
        Define::initializeParticleStaticArgs(
            arg_card, "bethe", 2.0,
            Define::CUTOFF_BASE_GION,  // genion -> MeV/u
            Define::TRANS_CEIL_GION
        );
        arg_card.insert<size_t>("nbin",      { 1000 }, { 100 }, { 10000 });
        arg_card.insert<bool>("lat_disp",   std::vector<bool>{ true });
        arg_card.insert<bool>("lat_alg96",  std::vector<bool>{ true });
        arg_card.insert<bool>("straggling", std::vector<bool>{ true });
        arg_card.insert<double>("smax",   { 1e+10 }, { 1e-5 }, { 1e+10 });
        arg_card.insert<double>("estepe", { 0.05  }, { 1e-5 }, { 0.25  });
        arg_card.insertUnlimitedFieldWithDefault<int>("print_xs", std::vector<int>{0});
        return arg_card;
    }


    template <>
    ArgumentCard InputCardFactory<Define::IonInelastic>::_setCard() {
        ArgumentCard arg_card("ION_INELASTIC");
        arg_card.insert<std::string>("mode_high", { "off"  });
        arg_card.insert<std::string>("mode_low",  { "incl" });
        arg_card.insert<double>("evap_cutoff",      { 1e-8 }, { 1e-30 }, { 0.1 });
        arg_card.insert<bool>("activate_fission", std::vector<bool>{ true  });
        arg_card.insert<bool>("use_exact_level",  std::vector<bool>{ false });
        return arg_card;
    }


}


namespace Define {


    Electron::Electron() {
        // Projectile definition
        this->_mass   = (double)constants::MASS_ELECTRON;
        this->_spin   = 1;
        this->_charge = -1;
        // Particle definition
        this->_pid       = PID::PID_ELECTRON;
        this->_activated = true;
        this->_t_cutoff  = 1e-1;  // Default 100 keV
        this->_p_cutoff  = 1e-1;  // Default 100 keV
        this->_t_ceil    = 1e+3;  // Maximum 1 GeV
        this->_library   = "pegsless";
        // Electron definition
        this->_do_spin      = true;
        this->_presta2      = true;
        this->_ie_fudge     = true;
        this->_print_xs     = false;
        this->_n_brem_split = 5;
        this->_nbin         = 1000;
        this->_fudgems      = 1.e0;
        this->_smax         = 5.e0;
        this->_estepe       = 0.2;
        this->_ximax        = 0.5;
        this->_brem_corr    = BREM_CORRECTION_METHOD::BREM_CORR_ICRU;
        this->_brem_xs      = BREM_XS_METHOD::BREM_XS_NRC;
        this->_brem_angle   = BREM_ANGLE_METHOD::BAM_SIMPLE;
        this->_eii_mode     = EII_MODE::EII_OFF;
    }


    Electron::Electron(mcutil::ArgInput& args) 
        : Electron::Electron() {
        this->_readHeader(args);
        this->_do_spin      = args["do_spin"].cast<bool>()[0];
        this->_presta2      = args["use_presta2"].cast<bool>()[0];
        this->_ie_fudge     = args["ie_fudge"].cast<bool>()[0];
        this->_print_xs     = args["print_xs"].cast<bool>()[0];
        this->_n_brem_split = args["n_brem_split"].cast<size_t>()[0];
        this->_nbin         = args["nbin"].cast<size_t>()[0];
        this->_fudgems      = args["fudgems"].cast<double>()[0];
        this->_smax         = args["smax"].cast<double>()[0];
        this->_estepe       = args["estepe"].cast<double>()[0];
        this->_ximax        = args["ximax"].cast<double>()[0];

        std::string brem_corr = args["brem_corr"].cast<std::string>()[0];
        if (brem_corr == "km")
            this->_brem_corr = BREM_CORRECTION_METHOD::BREM_CORR_KM;
        else if (brem_corr == "icru")
            this->_brem_corr = BREM_CORRECTION_METHOD::BREM_CORR_ICRU;
        else if (brem_corr == "off")
            this->_brem_corr = BREM_CORRECTION_METHOD::BREM_CORR_OFF;
        else
            mclog::fatal("'brem_corr' must be 'km', 'icru' or 'off'");

        std::string brem_xs = args["brem_xs"].cast<std::string>()[0];
        if (brem_xs == "bh")
            this->_brem_xs = BREM_XS_METHOD::BREM_XS_BH;
        else if (brem_xs == "nist")
            this->_brem_xs = BREM_XS_METHOD::BREM_XS_NIST;
        else if (brem_xs == "nrc")
            this->_brem_xs = BREM_XS_METHOD::BREM_XS_NRC;
        else
            mclog::fatal("'brem_xs' must be 'bh', 'nist' or 'nrc'");

        std::string brem_angle = args["brem_angle"].cast<std::string>()[0];
        if (brem_angle == "simple")
            this->_brem_angle = BREM_ANGLE_METHOD::BAM_SIMPLE;
        else if (brem_angle == "km")
            this->_brem_angle = BREM_ANGLE_METHOD::BAM_KM;
        else if (brem_angle == "off")
            this->_brem_angle = BREM_ANGLE_METHOD::BAM_INHERIT;
        else
            mclog::fatal("'brem_angle' must be 'simple', 'km' or 'off'");

        std::string eii_mode = args["eii_mode"].cast<std::string>()[0];
        if (eii_mode == "off")
            this->_eii_mode = EII_MODE::EII_OFF;
        else if (eii_mode == "kawrakow")
            this->_eii_mode = EII_MODE::EII_KAWRAKOW;
        else if (eii_mode == "gryzinski")
            this->_eii_mode = EII_MODE::EII_GRYZINSKI;
        else if (eii_mode == "casnati")
            this->_eii_mode = EII_MODE::EII_CASNATI;
        else if (eii_mode == "kolbenstvedt")
            this->_eii_mode = EII_MODE::EII_KOLBENSTVEDT;
        else if (eii_mode == "penelope")
            this->_eii_mode = EII_MODE::EII_PENELOPE;
        else
            mclog::fatal("'eii_mode' must be 'kawrakow', 'gryzinski', 'casnati', 'kolbenstvedt', or 'penelope'");

        Electron& def = Electron::getInstance();
        this->_t_ceil = def.transportCeil();
        def = *this;
    }


    Photon::Photon() {
        // Projectile definition
        this->_mass   = 0.0;
        this->_spin   = 0;
        this->_charge = 0;
        // Particle definition
        this->_pid       = PID::PID_PHOTON;
        this->_activated = true;
        this->_t_cutoff  = 1e-2;
        this->_p_cutoff  = 1e-2;
        this->_t_ceil    = 1e+3;
        this->_library   = "epdl97";
        // Photon definition
        this->_compton_mode = COMPTON_METHOD::COMPTON_SIMPLE;
        this->_use_nrc_pair = true;
        this->_do_rayleigh  = true;
        this->_simple_photo = false;
        this->_use_sauter   = true;
        this->_print_xs     = false;
        this->_nbin         = 1000;
    }


    Photon::Photon(mcutil::ArgInput& args) 
        : Photon() {
        this->_readHeader(args);
        this->_use_nrc_pair = args["use_nrc_pair"].cast<bool>()[0];
        this->_do_rayleigh  = args["do_rayleigh"].cast<bool>()[0];
        this->_simple_photo = args["simple_photo"].cast<bool>()[0];
        this->_use_sauter   = args["sauter"].cast<bool>()[0];
        this->_print_xs     = args["print_xs"].cast<bool>()[0];
        this->_nbin         = args["nbin"].cast<size_t>()[0];

        std::string compt_mode = args["compton_mode"].cast<std::string>()[0];
        if (compt_mode == "simple")
            this->_compton_mode = COMPTON_METHOD::COMPTON_SIMPLE;
        else if (compt_mode == "egsnrc")
            this->_compton_mode = COMPTON_METHOD::COMPTON_EGSNRC;
        else if (compt_mode == "livermore")
            this->_compton_mode = COMPTON_METHOD::COMPTON_GEANT4;
        else
            mclog::fatal("'compton_mode' must be 'simple', 'egsnrc' or 'livermore'");

        Photon& def   = Photon::getInstance();
        this->_t_ceil = def.transportCeil();
        def = *this;
    }


    Positron::Positron() {
        // Projectile definition
        this->_mass   = (double)constants::MASS_ELECTRON;
        this->_spin   = 1;
        this->_charge = 1;
        // Particle definition
        this->_pid       = PID::PID_POSITRON;
        this->_activated = true;
        this->_t_cutoff  = 1e-1;
        this->_p_cutoff  = 1e-1;
        this->_t_ceil    = 1e+3;
        this->_library   = "pegsless";
        // Positron definition
        // Share electron definition
    }


    Positron::Positron(mcutil::ArgInput& args) 
        : Positron() {
        this->_readHeader(args);
        Positron& def = Positron::getInstance();
        this->_t_ceil = def.transportCeil();
        def = *this;
    }


    Vacancy::Vacancy() {
        // Projectile definition
        this->_mass   = 0;
        this->_spin   = 0;
        this->_charge = 0;
        // Particle definition
        this->_pid       = PID::PID_VACANCY;
        this->_activated = true;
        this->_t_cutoff  = 1e-2;
        this->_p_cutoff  = 1e-2;
        this->_t_ceil    = 1e+3;
        this->_library   = "eadl";
        // Vacancy definition
        this->_local_deposit = false;
        this->_print_detail  = false;
    }


    Vacancy::Vacancy(mcutil::ArgInput& args)
        : Vacancy() {
        this->_readHeader(args);
        Vacancy& def = Vacancy::getInstance();
        this->_t_ceil        = def.transportCeil();
        this->_local_deposit = args["local_deposit"].cast<bool>()[0];
        this->_print_detail  = args["print_detail"].cast<bool>()[0];
        def = *this;
    }


    Neutron::Neutron() {
        // Projectile definition
        this->_mass   = (double)constants::MASS_NEUTRON;
        this->_spin   = 1;
        this->_charge = 0;
        // Particle definition
        this->_pid       = PID::PID_NEUTRON;
        this->_activated = true;
        // Low energy definition
        this->_t_cutoff  = 1e-11;
        this->_p_cutoff  = 1e-11;
        this->_library   = "endf8R0_260";
        // High energy definition
        this->_nbin   = 1000;
        this->_t_ceil = TRANS_CEIL_XS_NEUTRON_HIGH;
        this->_library_high = "Glauber-Gribov";
    }


    Neutron::Neutron(mcutil::ArgInput& args) 
        : Neutron() {
        this->_readHeader(args);
        Neutron& def    = Neutron::getInstance();
        this->_t_ceil   = def.transportCeil();
        this->_print_xs = args["print_xs"].cast<bool>()[0];
        this->_nbin     = args["nbin"].cast<size_t>()[0];
        def = *this;
    }


    DeltaRay::DeltaRay() {
        // Projectile definition
        this->_mass   = (double)constants::MASS_ELECTRON;
        this->_spin   = 1;
        this->_charge = -1;
        // Particle definition
        this->_pid       = PID::PID_DELTARAY;
        this->_activated = true;
        this->_t_cutoff  = 0.3;
        this->_p_cutoff  = 0.3;
        this->_t_ceil    = 1e+3;
        this->_library   = "";
        //DeltaRay definition
        this->_free_scat = false;
    }


    DeltaRay::DeltaRay(mcutil::ArgInput& args)
        : DeltaRay() {
        this->_readHeader(args);
        DeltaRay& def    = DeltaRay::getInstance();
        this->_t_ceil    = def.transportCeil();
        this->_free_scat = args["free_scat"].cast<bool>()[0];
        def = *this;
    }


    GenericIon::GenericIon() {
        // Projectile definition
        this->_mass   = 0.0;
        this->_spin   = 0;
        this->_charge = 0;
        // Particle definition
        this->_pid       = PID::PID_GENION;
        this->_activated = true;
        this->_t_cutoff  = 2.0;
        this->_p_cutoff  = 2.0;
        this->_t_ceil    = TRANS_CEIL_XS_GION;
        this->_library   = "";
        // Genion definition
        this->_nbin      = 1000;
        this->_lat_disp  = true;
        this->_lat_alg96 = true;
        this->_loss_fluc = true;
        this->_print_xs  = false;
        this->_smax      = 1e+10;
        this->_estepe    = 0.05;
    }


    GenericIon::GenericIon(mcutil::ArgInput& args)
        : GenericIon() {
        this->_readHeader(args);
        GenericIon& def  = GenericIon::getInstance();
        this->_t_ceil    = def.transportCeil();
        this->_nbin      = args["nbin"].cast<size_t>()[0];
        this->_lat_disp  = args["lat_disp"].cast<bool>()[0];
        this->_lat_alg96 = args["lat_alg96"].cast<bool>()[0];
        this->_loss_fluc = args["straggling"].cast<bool>()[0];
        this->_smax      = args["smax"].cast<double>()[0];
        this->_estepe    = args["estepe"].cast<double>()[0];

        // za list for print
        std::vector<int> za_list = args["print_xs"].cast<int>();
        if (za_list.size() == 1 && !za_list.front())
            this->_print_xs = false;
        else {
            this->_print_xs     = true;
            for (auto za : za_list) {
                if (this->_xs_proj_list.find(za) != this->_xs_proj_list.end()) {
                    std::stringstream ss;
                    ss << "Duplicated ZA element '" << za << "' in 'print_xs' argument";
                    mclog::fatal(ss);
                }
                this->_xs_proj_list.insert(za);
            }
        }

        def = *this;
    }


    IonInelastic::IonInelastic() {
        this->_mode_high   = ION_INELASTIC_METHOD_HIGH::ION_INELASTIC_HIGH_OFF;
        this->_mode_low    = ION_INELASTIC_METHOD_LOW::ION_INELASTIC_LOW_INCL;
        this->_evap_cutoff = 1e-8;
        this->_pid         = PID::PID_IONNUC;
        this->_activate_fission = true;
        this->_use_exact_level  = false;
    }


    IonInelastic::IonInelastic(mcutil::ArgInput& args) :
        IonInelastic() {
        IonInelastic& def = IonInelastic::getInstance();

        std::string mode_high = args["mode_high"].cast<std::string>()[0];
        if (mode_high == "off")
            this->_mode_high = ION_INELASTIC_METHOD_HIGH::ION_INELASTIC_HIGH_OFF;
        else if (mode_high == "qmd")
            this->_mode_high = ION_INELASTIC_METHOD_HIGH::ION_INELASTIC_HIGH_QMD;
        else if (mode_high == "abrasion")
            this->_mode_high = ION_INELASTIC_METHOD_HIGH::ION_INELASTIC_HIGH_ABRASION;
        else
            mclog::fatal("'mode_high' must be 'off', 'qmd', or 'abrasion'");

        std::string mode_low = args["mode_low"].cast<std::string>()[0];
        if (mode_low == "bme")
            this->_mode_low = ION_INELASTIC_METHOD_LOW::ION_INELASTIC_LOW_BME;
        else if (mode_low == "incl")
            this->_mode_low = ION_INELASTIC_METHOD_LOW::ION_INELASTIC_LOW_INCL;
        else
            mclog::fatal("'mode_low' must be 'bme', or 'incl'");

        this->_evap_cutoff = args["evap_cutoff"].cast<double>()[0];
        this->_activate_fission = args["activate_fission"].cast<bool>()[0];
        this->_use_exact_level  = args["use_exact_level"].cast<bool>()[0];
        if (this->_use_exact_level)
            mclog::fatal("'use_excat_level' option is not supported yet");

        def = *this;
    }


    void initializeParticleStaticArgs(
        mcutil::ArgumentCard& card,
        const std::string& library,
        double cutoff_default,
        double cutoff_minimum,
        double cutoff_maximum
    ) {
        card.insert<std::string>("library", { library });
        card.insert<bool>("activate", std::vector<bool>{ true });
        card.insert<double>(
            "transport_cutoff", 
            { cutoff_default }, 
            { cutoff_minimum }, 
            { cutoff_maximum }
        );
        card.insert<double>(
            "production_cutoff",
            { cutoff_default },
            { cutoff_minimum },
            { cutoff_maximum }
        );
    }


}