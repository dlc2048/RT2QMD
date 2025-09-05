#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "device/memory.cuh"
#include "transport/buffer.cuh"
#include "hadron/projectile.cuh"


namespace tally {


    typedef struct DeviceSecYield {
        // filter
        uint32_t za_mask[Hadron::Projectile::ZA_SCORING_MASK_SIZE];  // ZA mask
        double2* data;
        // energy structure (used if pointwise)
        int      type;     // false,0=linear, true,1=log
        float2   eih;      // energy indexing helper
        int      nebin;    // total number of energy bin
        float2   murange;  // solid angle range [min,max]


        __inline__ __device__ bool isIonTarget(uint32_t gion_id = Hadron::Projectile::ZA_SCORING_MASK_HION);


        __inline__ __device__ int getBinIndex(float energy);


        __inline__ __device__ void append(float mu, float weight, int ebin);


        __host__ void free();


    } DeviceSecYield;


#ifdef __CUDACC__


    __inline__ __device__ bool DeviceSecYield::isIonTarget(uint32_t gion_id) {
        uint32_t index  = gion_id / Hadron::Projectile::ZA_SCORING_MASK_STRIDE;
        uint32_t offset = gion_id % Hadron::Projectile::ZA_SCORING_MASK_STRIDE;
        return this->za_mask[index] & (0x1u << offset);
    }


    __inline__ __device__ int DeviceSecYield::getBinIndex(float energy) {
        if (this->type) {
            energy = logf(energy);
        }
        return floorf(eih.x + eih.y * energy);
    }


    __device__ void DeviceSecYield::append(float mu, float weight, int ebin) {

        if (ebin >= nebin || ebin < 0) return;  // energy out of range

        if (mu > this->murange.y || mu < this->murange.x) return;  // direction cosine out of range

        atomicAdd(&data[ebin].x, (double)weight);
        atomicAdd(&data[ebin].y, (double)weight * weight);
    }


#endif


    typedef struct DeviceYieldHandle {
        DeviceSecYield** yield;
        int              n_yield;


        __host__ static void Deleter(DeviceYieldHandle* ptr);


    } DeviceYieldHandle;


    // kernel lists


    __global__ void __kernel__appendYieldFromBuffer(
        mcutil::RingBuffer*    buffer,
        mcutil::BUFFER_TYPE    btype,
        DeviceSecYield**       yield_list,
        int                    n_yield
    );


    __host__ void appendYieldFromBuffer(int block, int thread, mcutil::RingBuffer* buffer, mcutil::BUFFER_TYPE btype, DeviceYieldHandle* handle_ptr);


}