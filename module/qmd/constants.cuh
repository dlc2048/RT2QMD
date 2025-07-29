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
 * @file    module/qmd/constants.cuh
 * @brief   QMD constants
 * @author  CM Lee
 * @date    02/14/2024
 */


#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#define _USE_MATH_DEFINES
#include <cmath>

#include "physics/constants.cuh"


namespace RT2QMD {


    // Flags


    enum PARTICIPANT_FLAGS {
        PARTICIPANT_IS_PROTON     = (1 << 0),
        PARTICIPANT_IS_TARGET     = (1 << 1),
        PARTICIPANT_IS_PROJECTILE = (1 << 2),
        PARTICIPANT_IS_IN_CLUSTER = (1 << 3)
    };

    constexpr int CLUSTER_IDX_SHIFT      = 4;
    constexpr int PARTICIPANT_FLAGS_MASK = (1 << CLUSTER_IDX_SHIFT) - 1;


    constexpr int INITIAL_FLAG_PROTON  = (0 |  PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON);
    constexpr int INITIAL_FLAG_NEUTRON = (0 & ~PARTICIPANT_FLAGS::PARTICIPANT_IS_PROTON);


    typedef enum MODEL_FLAGS {
        MODEL_FAIL_PROPAGATION = (1 << 0)
    } MODEL_FLAGS;


    enum MODEL_STAGE {
        MODEL_IDLE,
        MODEL_PREPARE_PROJECTILE,
        MODEL_SAMPLE_PROJECTILE,
        MODEL_PREPARE_TARGET,
        MODEL_SAMPLE_TARGET,
        MODEL_PROPAGATE
    };


    namespace constants {

        constexpr float  MASS_DIF_2        = 
            ::constants::MASS_NEUTRON_GEV * ::constants::MASS_NEUTRON_GEV - 
            ::constants::MASS_PROTON_GEV  * ::constants::MASS_PROTON_GEV;
        constexpr float  CCOUL             = 1.439767e-3f;             //! @brief Coulomb force coefficient, e^2/(4 pi epsilon0), [GeV*fm]

        // G4QMDReaction

        constexpr float  ENVELOP_F         = 1.05f;          //! @brief impact parameter margin, 5%

        // G4QMDParameters

        constexpr float  WAVE_PACKET_WIDTH = 2.f;            //! @brief width of gaussian wave packet [fm]     << G4QMDParameters::wl
        constexpr float  PAULI_CPW         = 0.5f / WAVE_PACKET_WIDTH;                                       // << G4QMDParameters::cpw
        constexpr float  PAULI_CPH         = 2.f  * WAVE_PACKET_WIDTH 
            / (::constants::FP32_HBARC * ::constants::FP32_HBARC);                                          // << G4QMDParameters::cph
        constexpr float  PAULI_CPC         = 4.0f;                                                          // << G4QMDParameters::cpc
        constexpr float  PAULI_EPSX        = -20.0f;                                                        // << G4QMDParameters::epsx

        // QMD Hamiltonian constants

        constexpr float H_SKYRME_GAMM = 4.f / 3.f;  //! @brief Skyrme gamma                                    << G4QMDParameters::gamm
        extern __constant__ float H_SKYRME_C0;      //! @brief Skyrem first term                               << G4QMDParameters::c0
        extern __constant__ float H_SKYRME_C3;      //! @brief Skyrem second term                              << G4QMDParameters::c3
        extern __constant__ float H_SYMMETRY;       //! @brief Symmetry term                                   << G4QMDParameters::cs
        extern __constant__ float H_COULOMB;        //! @brief Coulomb repulsion term                          << G4QMDParameters::cl

        // QMD Hamiltonian distance parameter

        extern __constant__ float D_SKYRME_C0;                                                              // << G4QMDMeanField::c0w
        extern __constant__ float D_SKYRME_C0_S;                                                            // << G4QMDMeanField::c0sw
        extern __constant__ float D_COULOMB;                                                                // << G4QMDMeanField::clw

        constexpr float CLUSTER_R       = 4.0f;     //! @brief distance for cluster judgement  [fm]            << G4QMDMeanField::rclds
        constexpr float CLUSTER_R2      = CLUSTER_R * CLUSTER_R;                                            // << G4QMDMeanField::DoClusterJudgement::rcc2
        constexpr float COULOMB_EPSILON = 1e-4f;                                                            // << G4QMDMeanField::epscl
        extern __constant__ float CLUSTER_CPF2;                                                             // << G4QMDMeanField::DoClusterJudgement::cpf2


        // QMD Hamiltonian gradient parameter
        constexpr float G_SKYRME_GAMM = 1.f / 3.f;                                                          // << G4QMDMeanField::pag
        extern __constant__ float G_SKYRME_C0;                                                              // << G4QMDMeanField::c0g
        extern __constant__ float G_SKYRME_C3;                                                              // << G4QMDMeanField::c3g
        extern __constant__ float G_SYMMETRY;                                                               // << G4QMDMeanField::csg
        
        // QMD ground state nucleus

        extern __constant__ float GS_CD;                                                                    // << G4QMDParameters::cdp
        extern __constant__ float GS_C0;                                                                    // << G4QMDParameters::c0p
        extern __constant__ float GS_C3;                                                                    // << G4QMDParameters::c3p
        extern __constant__ float GS_CS;                                                                    // << G4QMDParameters::csp
        extern __constant__ float GS_CL;                                                                    // << G4QMDParameters::clp

        // QMD nucleus initialization

        constexpr float WOOD_SAXON_RADIUS_00 = 1.124f;     //! @brief radius parameter for Wood-Saxon [fm]     << G4QMDGroundStateNucleus::r00
        constexpr float WOOD_SAXON_RADIUS_01 = 0.5f;       //! @brief radius parameter for Wood-Saxon [fm]     << G4QMDGroundStateNucleus::r01
        constexpr float WOOD_SAXON_DIFFUSE   = 0.2f;       //! @brief diffuse parameter for Wood-Saxon         << G4QMDGroundStateNucleus::saa
        constexpr float QMD_NUCLEUS_CUTOFF_A = 0.9f;       //! @brief cutoff paramter                          << G4QMDGroundStateNucleus::rada
        constexpr float QMD_NUCLEUS_CUTOFF_B = 0.3f;       //! @brief cutoff paramter                          << G4QMDGroundStateNucleus::radb
        constexpr float NUCLEUS_MD2_ISO_2    = 1.5 * 1.5;  //! @brief minimum distance for nn, pp [fm^2]       << G4QMDGroundStateNucleus::dsam2
        constexpr float NUCLEUS_MD2_ISO_0    = 1.0 * 1.0;  //! @brief minimum distance for np [fm^2]           << G4QMDGroundStateNucleus::ddif2
        constexpr float EDIF_TOLERANCE       = 1e-4f;      //! @brief tolerance for energy adjust [GeV]        << G4QMDGroundStateNucleus::epse

        // QMD propagation

        constexpr float PROPAGATE_DT  = 1.f;
        constexpr float PROPAGATE_DT1 = -PROPAGATE_DT * 0.5f;
        constexpr float PROPAGATE_DT2 = +PROPAGATE_DT;
        constexpr float PROPAGATE_DT3 = +PROPAGATE_DT * 0.5f;

        constexpr int   PROPAGATE_MAX_TIME = 100;

        // QMD binary collision

        constexpr float FDELTAR2             = 4.f * 4.f;  //! @brief minimum distance of approach for BC      << G4QMDCollision::fdeltar ^ 2
        constexpr float FBCMAX0              = 1.323142f;  //! @brief NN maximum impact parameter              << G4QMDCollision::fbcmax0


        cudaError_t setSymbolHCoeffs(float h_skyrme_c0, float h_skyrme_c3, float h_symmetry, float h_coulomb);


        cudaError_t setSymbolHDistance(float d_skyrme_c0, float d_skyrme_c0_s, float d_coulomb);


        cudaError_t setSymbolClusterCoeffs(float cpf2);


        cudaError_t setSymbolHGradient(float g_skyrme_c0, float g_skyrme_c3, float g_symmetry);


        cudaError_t setSymbolGroundNucleusCoeffs(float gs_cd, float gs_c0, float gs_c3, float gs_cs, float gs_cl);

    }
}