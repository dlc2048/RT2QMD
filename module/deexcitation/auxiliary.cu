
#include "auxiliary.cuh"


namespace deexcitation {


    __device__ float PROJ_M[CHANNEL::CHANNEL_UNKNWON];
    __device__ float PROJ_M2[CHANNEL::CHANNEL_UNKNWON];
    __device__ float PROJ_CB_RHO[CHANNEL::CHANNEL_UNKNWON];

    __constant__ bool BUFFER_HAS_HID;
    __device__ mcutil::RingBuffer* buffer_catalog;

    __device__ curandState* rand_state;

    __device__ Nucleus::MassTable* mass_table;

    __device__ Nucleus::LongLivedNucleiTable long_lived_table;

    __device__ float* coulomb_r0;

    __constant__ bool DO_FISSION;

    __constant__ bool USE_DISCRETE_LEVEL;


    __device__ float coulombBarrierRadius(int z, int a) {
        float r = Nucleus::explicitNuclearRadius({ (unsigned char)z, (unsigned char)a });
        if (r <= 0.f) {
            z = min(z, 92);
            r = coulomb_r0[z] * powf((float)a, constants::ONE_OVER_THREE);
        }
        return r;
    }


    __device__ float coulombBarrier(CHANNEL channel, int rz, int ra, float exc_energy) {
        float cb = constants::FP32_FSC_HBARC_MEV * (float)(PROJ_Z[channel] * rz)
            / (coulombBarrierRadius(rz, ra) + PROJ_CB_RHO[channel]);
        if (exc_energy > 0.f)
            cb /= 1.f + sqrtf(exc_energy / (float)ra * 0.5f);
        return cb;
    }


    __host__ cudaError_t setBufferHandle(CUdeviceptr handle, bool has_hid) {
        M_SOASymbolMapper(mcutil::RingBuffer*, handle, buffer_catalog);
        M_SOAPtrMapper(bool, has_hid, BUFFER_HAS_HID);
        return cudaSuccess;
    }


    __host__ cudaError_t setPrngHandle(CUdeviceptr handle) {
        M_SOASymbolMapper(curandState*, handle, rand_state);
        return cudaSuccess;
    }


    __host__ cudaError_t setMassTableHandle(CUdeviceptr handle) {
        M_SOASymbolMapper(Nucleus::MassTable*, handle, mass_table);
        return cudaSuccess;
    }


    __host__ cudaError_t setStableTable(const Nucleus::LongLivedNucleiTable& table_host) {
        return cudaMemcpyToSymbol(long_lived_table, &table_host, sizeof(Nucleus::LongLivedNucleiTable));
    }


    __host__ cudaError_t setCoulombBarrierRadius(float* cr_arr) {
        M_SOAPtrMapper(float*, cr_arr, coulomb_r0);
        return cudaSuccess;
    }


    __host__ cudaError_t setEmittedParticleMass(float* mass_arr, float* mass2_arr) {
        cudaError_t res;
        res = cudaMemcpyToSymbol(PROJ_M[0],  mass_arr,  
            sizeof(float) * CHANNEL::CHANNEL_UNKNWON);
        if (res != cudaSuccess) return res;
        res = cudaMemcpyToSymbol(PROJ_M2[0], mass2_arr, 
            sizeof(float) * CHANNEL::CHANNEL_UNKNWON);
        if (res != cudaSuccess) return res;
        return res;
    }


    __host__ cudaError_t setEmittedParticleCBRho(float* rho_arr) {
        return cudaMemcpyToSymbol(PROJ_CB_RHO[0], rho_arr,
            sizeof(float) * CHANNEL::CHANNEL_UNKNWON);
    }


    __host__ cudaError_t setFissionFlag(bool flag) {
        M_SOAPtrMapper(bool, flag, DO_FISSION);
        return cudaSuccess;
    }


}