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
 * @file    module/particles/define_struct.hpp
 * @brief   RT2 particle enums
 * @author  CM Lee
 * @date    07/11/2023
 */


#pragma once


namespace Define {

    typedef enum BREM_CORRECTION_METHOD {
        BREM_CORR_KM,      // Koch & Motz empirical correction
        BREM_CORR_ICRU,    // NIST/ICRU correction
        BREM_CORR_OFF      // turn off Bremsstrahlung correction
    } BREM_CORRECTION_METHOD;


    typedef enum BREM_XS_METHOD {
        BREM_XS_BH,      // Bethe & Heitler
        BREM_XS_NIST,    // NIST bremsstrahlung xs database
        BREM_XS_NRC      // NIST + electron-electron correction
    } BREM_XS_METHOD;


    typedef enum BREM_ANGLE_METHOD {
        BAM_KM,      // Koch & Motz 2BS
        BAM_SIMPLE,  // Use KM leading term
        BAM_INHERIT  // photon 'inherits' electron direction
    } BREM_ANGLE_METHOD;


    typedef enum COMPTON_METHOD {
        COMPTON_SIMPLE,  // simple compton scattering, using plain KN equation
        COMPTON_EGSNRC,  // EGSnrc bounded compton & doppler broadening method
        COMPTON_GEANT4   // Geant4 penelope bounded compton & doppler broadening method
    } COMPTON_METHOD;


    typedef enum ION_INELASTIC_METHOD_HIGH {
        ION_INELASTIC_HIGH_OFF,      // Inactivate ion inelastic reaction in high energy
        ION_INELASTIC_HIGH_QMD,      // Quantum Molecular Dynamics
        ION_INELASTIC_HIGH_ABRASION  // Wilson abrasion
    } ION_INELASTIC_METHOD_HIGH;


    typedef enum ION_INELASTIC_METHOD_LOW {
        ION_INELASTIC_LOW_BME,       // Boltzmann Master Equation
        ION_INELASTIC_LOW_INCL       // INCL complete fusion
    } ION_INELASTIC_METHOD_LOW;


    typedef enum PID {
        PID_VACANCY      = -3,
        PID_DELTARAY     = -2,
        PID_ELECTRON     = -1,
        PID_PHOTON       = +0,
        PID_POSITRON     = +1,
        PID_NEUTRON      = +6,
        PID_PROTON       = +21,
        PID_GENION       = +26,
        PID_IONNUC       = +27,
        PID_ALL          = +99,  // for scoring
        PID_UNKNOWN      = -99
    } PID;

}