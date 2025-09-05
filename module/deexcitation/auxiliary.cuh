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
 * @file    module/deexcitation/auxiliary.cuh
 * @brief   Data structure for de-excitation channel 
 * @author  CM Lee
 * @date    07/08/2024
 */


#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "device/shuffle.cuh"

#include "device/memory.cuh"
#include "hadron/nucleus.cuh"
#include "transport/buffer.cuh"
#include "scoring/tally.cuh"

#include "nuclear_level.cuh"


namespace deexcitation {


    typedef enum XS_MODEL {
        XS_MODEL_DOSTROVSKY,
        XS_MODEL_CHATTERJEE,
        XS_MODEL_KALBACH
    } XS_MODEL;


    typedef enum CHANNEL {
        CHANNEL_PHOTON,
        CHANNEL_FISSION,
        CHANNEL_NEUTRON,
        CHANNEL_PROTON,
        CHANNEL_DEUTERON,
        CHANNEL_TRITON,
        CHANNEL_HELIUM3,
        CHANNEL_ALPHA,
        CHANNEL_2N,       // for unstable breakup
        CHANNEL_2P,
        CHANNEL_UNKNWON   // for EOF
    } CHANNEL;


    typedef enum FLAGS {
        FLAG_IS_STABLE        = (1 << 0),
        FLAG_CHANNEL_FOUND    = (1 << 1),
        FLAG_CHANNEL_PHOTON   = (1 << 2),
        FLAG_CHANNEL_FISSION  = (1 << 3),
        FLAG_CHANNEL_UBREAKUP = (1 << 4),
        FLAG_CHANNEL_DOUBLE   = (1 << 5)   // unstable breakup 2p & 2n channel
    } FLAGS;


    constexpr float MIN_EXC_ENERGY  = (float)2.e-3;  // photon evaporation, min excitation energy [MeV]
    constexpr int   MAX_A_BREAKUP   = 30;
    constexpr float MAX_ENERGY_CJXS = 50.f;          // maximum excitation energy in Chatterjee XS
    constexpr int   DIM_XS          = 11;

    constexpr float EDELTA_NEUTRON  = 0.15f;
    constexpr float EDELTA_CHARGED  = 0.25f;

    // emitted particle info

    __device__ constexpr int PROJ_A[CHANNEL::CHANNEL_UNKNWON] = { 0, 0, 1, 1, 2, 3, 3, 4, 2, 2 };  // Mass number
    __device__ constexpr int PROJ_Z[CHANNEL::CHANNEL_UNKNWON] = { 0, 0, 0, 1, 1, 1, 2, 2, 0, 2 };  // Atomic number
    __device__ constexpr int PROJ_S[CHANNEL::CHANNEL_UNKNWON] = { 0, 0, 2, 2, 3, 2, 2, 1, 4, 4 };  // 2 x spin + 1

    extern __device__ float PROJ_M[CHANNEL::CHANNEL_UNKNWON];       // Emitted particle mass [MeV/c^2]
    extern __device__ float PROJ_M2[CHANNEL::CHANNEL_UNKNWON];      // Square of mass of emitted particle [MeV^2/c^4]
    extern __device__ float PROJ_CB_RHO[CHANNEL::CHANNEL_UNKNWON];  // Coulomb barrier rho [fm]

    // external

    extern __constant__ bool BUFFER_HAS_HID;
    extern __device__ mcutil::RingBuffer* buffer_catalog;

    extern __device__ curandState* rand_state;
    extern __device__ Nucleus::MassTable* mass_table;  // mass table
    extern __device__ Nucleus::LongLivedNucleiTable long_lived_table;  // stable nuclei list

    // coulomb
    extern __constant__ float COULOMB_RATIO;
    extern __device__ float* coulomb_r0;

    // inverse XS
    extern __device__ float CJXS_PARAM[DIM_XS][CHANNEL::CHANNEL_UNKNWON];  // Chatterjee
    extern __device__ float KMXS_PARAM[DIM_XS][CHANNEL::CHANNEL_UNKNWON];  // Kalbach-Mann

    extern __constant__ XS_MODEL XS_TYPE;             // inverse XS
    extern __constant__ bool     DO_FISSION;          // fission flag
    extern __constant__ bool     USE_DISCRETE_LEVEL;  // photon flag


    // Chatterjee
    namespace Chatterjee {


        typedef struct IntegrateSharedMem {
            int     __pad[2];                              //! @brief memory pad for queue push/pull actions & reduction
            int     condition_broadcast;                   //! @brief condition broadcast for consistent loop break
            // Reduction
            float   redux_r1[CUDA_WARP_SIZE];
            float   redux_r2[CUDA_WARP_SIZE];
            // Channel info
            int     channel;                               //! @brief current evaporation channel
            float   channel_prob[CHANNEL::CHANNEL_2N];     //! @brief channel selection probability
            float   channel_prob_max[CHANNEL::CHANNEL_2N]; //! @brief maximum probability of channel
            int     is_allowed;                            //! @brief true if allowd channel, false elsewhere
            float   emin;                                  //! @brief minimum energy of Weisskopf integrate [MeV] 
            float   emax;                                  //! @brief maximum energy of Weisskopf integrate [MeV] 
            float   cb;                                    //! @brief coulomb barrier [MeV]
            float   mass;                                  //! @brief mass of the parent nucleus [MeV/c^2]
            float   exc;                                   //! @brief excitation energy of the parent nucleus [MeV]
            float   a0;                                    //! @brief level density of the parent nucleus
            float   a1;                                    //! @brief level density of the residual nucleus
            int     res_a;                                 //! @brief mass number of the residual nucleus
            float   res_a13;                               //! @brief res_a^(1/3)
            float   res_mass;                              //! @brief mass of the residual nucleus [MeV/c^2]
            float   delta0;                                //! @brief pairing correction of the fragment
            float   delta1;                                //! @brief pairing correction of the residual nucleus
            // Numerical
            float   edelta;                                //! @brief numerical delta
            int     int_iter;                              //! @brief blockwised integral loop iter
            int     int_iter_2nd;                          //! @brief 2nd iterator stride
            // Chatterjee parameters
            float   p;
            float   landa;
            float   mu;
            float   nu;
            float   q;
            float   r;
            // for reduction
            float   prob;
            float   prob_max;
        } IntegrateSharedMem;


        constexpr int INTEGRATE_SHARED_MEM_SIZE   = sizeof(IntegrateSharedMem) / 4;
        constexpr int INTEGRATE_SHARED_MEM_OFFSET = ((INTEGRATE_SHARED_MEM_SIZE - 1) / 32 + 1) * 32;


    }


    // Kalbach
    namespace Kalbach {


        typedef struct IntegrateSharedMem {
            int     __pad[2];                              //! @brief memory pad for queue push/pull actions & reduction
            int     condition_broadcast;                   //! @brief condition broadcast for consistent loop break
            // Reduction
            float   redux_r1[CUDA_WARP_SIZE];
            float   redux_r2[CUDA_WARP_SIZE];
            // Channel info
            int     channel;                               //! @brief current evaporation channel
            float   channel_prob[CHANNEL::CHANNEL_2N];     //! @brief channel selection probability
            float   channel_prob_max[CHANNEL::CHANNEL_2N]; //! @brief maximum probability of channel
            int     is_allowed;                            //! @brief true if allowd channel, false elsewhere
            float   emin;                                  //! @brief minimum energy of Weisskopf integrate [MeV] 
            float   emax;                                  //! @brief maximum energy of Weisskopf integrate [MeV] 
            float   cb;                                    //! @brief coulomb barrier [MeV]
            float   mass;                                  //! @brief mass of the parent nucleus [MeV/c^2]
            float   exc;                                   //! @brief excitation energy of the parent nucleus [MeV]
            float   a0;                                    //! @brief level density of the parent nucleus
            float   a1;                                    //! @brief level density of the residual nucleus
            int     res_a;                                 //! @brief mass number of the residual nucleus
            float   res_a13;                               //! @brief res_a^(1/3)
            float   res_mass;                              //! @brief mass of the residual nucleus [MeV/c^2]
            float   delta0;                                //! @brief pairing correction of the fragment
            float   delta1;                                //! @brief pairing correction of the residual nucleus
            // Numerical
            float   edelta;                                //! @brief numerical delta
            int     int_iter;                              //! @brief blockwised integral loop iter
            int     int_iter_2nd;                          //! @brief 2nd iterator stride
            // Kalbach parameters
            float   signor;
            float   a;
            float   b;
            float   ecut;
            float   p;
            float   lambda;
            float   mu;
            float   nu;
            float   geom;
            // for reduction
            float   prob;
            float   prob_max;
        };


        constexpr int INTEGRATE_SHARED_MEM_SIZE   = sizeof(IntegrateSharedMem) / 4;
        constexpr int INTEGRATE_SHARED_MEM_OFFSET = ((INTEGRATE_SHARED_MEM_SIZE - 1) / 32 + 1) * 32;


    }


    __device__ float coulombBarrierRadius(int z, int a);


    __device__ float coulombBarrier(CHANNEL channel, int rz, int ra, float exc_energy);


    __host__ cudaError_t setBufferHandle(CUdeviceptr handle, bool has_hid);


    __host__ cudaError_t setPrngHandle(CUdeviceptr handle);


    __host__ cudaError_t setMassTableHandle(CUdeviceptr handle);


    __host__ cudaError_t setStableTable(const Nucleus::LongLivedNucleiTable& table_host);


    __host__ cudaError_t setCoulombBarrierRadius(float* cr_arr);


    __host__ cudaError_t setCoulombRatio(float coulomb_penetration_ratio);


    __host__ cudaError_t setChatterjeeXS(float xs_host[][(int)CHANNEL::CHANNEL_UNKNWON]);


    __host__ cudaError_t setKalbackXS(float xs_host[][(int)CHANNEL::CHANNEL_UNKNWON]);


    __host__ cudaError_t setEmittedParticleMass(float* mass_arr, float* mass2_arr);


    __host__ cudaError_t setEmittedParticleCBRho(float* rho_arr);


    __host__ cudaError_t setFissionFlag(bool flag);


    namespace photon {


        typedef enum INTERNAL_CONVERSION_MODE {
            GAMMA                 ,
            INTERNAL_CONVERSION_K , 
            INTERNAL_CONVERSION_L1,
            INTERNAL_CONVERSION_L2,
            INTERNAL_CONVERSION_L3,
            INTERNAL_CONVERSION_M1,
            INTERNAL_CONVERSION_M2,
            INTERNAL_CONVERSION_M3,
            INTERNAL_CONVERSION_M4,
            INTERNAL_CONVERSION_M5,
            INTERNAL_CONVERSION_FREE
        } INTERNAL_CONVERSION_SHELL;


    }


}