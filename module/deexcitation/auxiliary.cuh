/**
 * @file    module/deexcitation/auxiliary.cuh
 * @brief   Data structure for de-excitation channel 
 * @author  CM Lee
 * @date    07/08/2024
 */

#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "device/memory.cuh"
#include "hadron/nucleus.cuh"
#include "transport/buffer.cuh"
#include "scoring/tally.cuh"

#include "nuclear_level.cuh"


namespace deexcitation {


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


    constexpr float MIN_EXC_ENERGY = (float)2.e-3;  // photon evaporation, min excitation energy [MeV]

    constexpr int   MAX_A_BREAKUP  = 30;


    // emitted particle info

    __device__ constexpr int PROJ_A[CHANNEL::CHANNEL_UNKNWON] = { 0, 0, 1, 1, 2, 3, 3, 4, 2, 2 };  // Mass number
    __device__ constexpr int PROJ_Z[CHANNEL::CHANNEL_UNKNWON] = { 0, 0, 0, 1, 1, 1, 2, 2, 0, 2 };  // Atomic number
    __device__ constexpr int PROJ_S[CHANNEL::CHANNEL_UNKNWON] = { 0, 0, 2, 2, 3, 2, 2, 1, 4, 4 };  // 2 x spin + 1

    extern __device__ float PROJ_M[CHANNEL::CHANNEL_UNKNWON];       // Emitted particle mass [MeV/c^2]
    extern __device__ float PROJ_M2[CHANNEL::CHANNEL_UNKNWON];      // Square of mass of emitted particle [MeV^2/c^4]
    extern __device__ float PROJ_CB_RHO[CHANNEL::CHANNEL_UNKNWON];  // Coulomb barrier rho [fm]

    extern __constant__ bool BUFFER_HAS_HID;
    extern __device__ mcutil::RingBuffer* buffer_catalog;

    extern __device__ curandState* rand_state;

    // mass table
    extern __device__ Nucleus::MassTable* mass_table;

    // stable nuclei list
    extern __device__ Nucleus::LongLivedNucleiTable long_lived_table;

    // coulomb
    extern __device__ float* coulomb_r0;

    // fission flag
    extern __constant__ bool DO_FISSION;

    // photon flag
    extern __constant__ bool USE_DISCRETE_LEVEL;


    __device__ float coulombBarrierRadius(int z, int a);


    __device__ float coulombBarrier(CHANNEL channel, int rz, int ra, float exc_energy);


    __host__ cudaError_t setBufferHandle(CUdeviceptr handle, bool has_hid);


    __host__ cudaError_t setPrngHandle(CUdeviceptr handle);


    __host__ cudaError_t setMassTableHandle(CUdeviceptr handle);


    __host__ cudaError_t setStableTable(const Nucleus::LongLivedNucleiTable& table_host);


    __host__ cudaError_t setCoulombBarrierRadius(float* cr_arr);


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