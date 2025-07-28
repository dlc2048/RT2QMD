# pragma once

#include <curand_kernel.h>
#include <curand.h>

#include "device/memory.cuh"
#include "device/algorithm.cuh"
#include "transport/transport.cuh"
#include "transport/buffer.cuh"


namespace auxiliary {


    typedef struct DeviceEvent {
        float3 pos;
        float3 dir;
        float  energy;
        int    aux1;
        int    aux2;
    } DeviceEvent;


    extern __constant__ bool BUFFER_HAS_HID;

    extern __device__ mcutil::RingBuffer* buffer_catalog;
    extern __device__ geo::DeviceTracerHandle* tracer_handle;
    
    extern __device__ mcutil::RingBuffer* qmd_problems;

    // neutron
    extern __device__   float* NGROUP;     //!< Neutron group structure (MeV), not used in RT2QMD

    extern __device__ DeviceEvent* event_list;
    extern __device__ mcutil::DeviceAliasData* event_ptable;

    extern __device__ curandState* rand_state;


    __host__ void initPhaseSpace(int block, int thread, mcutil::BUFFER_TYPE bid, int hid);
    __global__ void __kernel__initPhaseSpace(mcutil::BUFFER_TYPE bid, int hid);


    __host__ void dummyOptixLaunch(int block, int thread, int region);
    __global__ void __kernel__dummyOptixLaunch(int region);


    __host__ cudaError_t setBufferHandle(CUdeviceptr handle, bool has_hid);
    __host__ cudaError_t setTracerHandle(CUdeviceptr handle);
    __host__ cudaError_t setQMDBuffer(CUdeviceptr ptr_qmd_buffer);
    __host__ cudaError_t setPrngHandle(CUdeviceptr handle);

    __host__ cudaError_t setNeutronGroupHandle(CUdeviceptr handle);

    __host__ cudaError_t setEventListPtr(DeviceEvent* ptr_event_list, mcutil::DeviceAliasData* ptr_event_ptable);


    __inline__ __device__ int getGroupFromEnergy(float energy);


#ifdef __CUDACC__


    __device__ int getGroupFromEnergy(float energy) {
        for (int i = 0;; ++i) {
            if (energy < NGROUP[i + 1])
                return i;
        }
        return -1;
    }


#endif


}