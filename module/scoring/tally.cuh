#pragma once

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <assert.h>

#ifndef RT2QMD_STANDALONE
#include "mesh_track.cuh"
#include "mesh_density.cuh"
#include "cross.cuh"
#include "track.cuh"
#include "density.cuh"
#include "detector.cuh"
#include "phase_space.cuh"
#include "activation.cuh"
#include "letd.cuh"
#include "mesh_letd.cuh"
#endif


namespace tally {


#ifdef RT2QMD_STANDALONE
    // dummy struct
    typedef struct DeviceMeshTrack   {};
    typedef struct DeviceMeshDensity {
        __inline__ __device__ bool isIonTarget() { return false; }
        __inline__ __device__ void append(float3 pos, float energy, float rbe, float weight, float density) {}
    };
    typedef struct DeviceCross       {};
    typedef struct DeviceTrack       {};
    typedef struct DeviceDensity     {
        __inline__ __device__ bool isIonTarget() { return false; }
        __inline__ __device__ void append(int rold, float energy, float rbe, float weight, float density) {}
    };
    typedef struct DeviceDetector    {
        __inline__ __device__ bool isIonTarget() { return false; }
        __inline__ __device__ void append(int rold, float weight, float depo, int hid) {}
    };
    typedef struct DevicePhaseSpace  {};
    typedef struct DeviceActivation  {};
    typedef struct DeviceDensityLETD {
        __inline__ __device__ bool isIonTarget() { return false; }
        __inline__ __device__ void append(int rold, float energy, float let, float weight) {}
    };
    typedef struct DeviceMeshLETD    {
        __inline__ __device__ bool isIonTarget() { return false; }
        __inline__ __device__ void append(float3 pos, float energy, float let, float weight) {}
    };
#endif


    typedef struct DeviceHandle {
        DeviceMeshTrack**   mesh_track;
        DeviceMeshDensity** mesh_density;
        DeviceCross**       cross;
        DeviceTrack**       track;
        DeviceDensity**     density;
        DeviceDetector**    detector;
        DevicePhaseSpace**  phase_space;
        DeviceActivation**  activation;
        DeviceDensityLETD** letd;
        DeviceMeshLETD**    mesh_letd;
        int                 n_mesh_track;
        int                 n_mesh_density;
        int                 n_cross;
        int                 n_track;
        int                 n_density;
        int                 n_detector;
        int                 n_phase_space;
        int                 n_activation;
        int                 n_letd;
        int                 n_mesh_letd;


        __host__ static void Deleter(DeviceHandle* ptr);


    } DeviceHandle;


    // prng
    extern __device__ curandState* rand_state;


    __host__ void __host__deviceGetPhaseSpaceSaturation(int thread, DevicePhaseSpace** ps, float* saturation);
    __global__ void __device__deviceGetPhaseSpaceSaturation(DevicePhaseSpace** ps, float* saturation);


    __host__ void __host__deviceInitDetector(int block, int thread, DeviceDetector* det, int det_size);
    __global__ void __device__deviceInitDetector(DeviceDetector* det, int det_size);


    __host__ void __host__deviceProcessDetector(int block, int thread, DeviceDetector* det, int det_size);
    __global__ void __device__deviceProcessDetector(DeviceDetector* det, int det_size);


    __host__ cudaError_t setPrngHandle(CUdeviceptr handle);


}


