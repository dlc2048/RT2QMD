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
 * @file    module/scoring/tally.cuh
 * @brief   Device side tally pointer structure
 * @author  CM Lee
 * @date    05/23/2023
 */


#pragma once

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <assert.h>

#ifndef RT2QMD_STANDALONE
#include "mesh_track.cuh"
#include "mesh_density.cuh"
#include "mesh_activation.cuh"
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
    typedef struct DeviceMeshTrack      {} DeviceMeshTrack;
    typedef struct DeviceMeshDensity    {
        __inline__ __device__ bool isIonTarget() { return false; }
        __inline__ __device__ void append(float3 pos, float energy, float rbe, float weight, float density) {}
    } DeviceMeshDensity;
    typedef struct DeviceCross          {} DeviceCross;
    typedef struct DeviceTrack          {} DeviceTrack;
    typedef struct DeviceDensity        {
        __inline__ __device__ bool isIonTarget() { return false; }
        __inline__ __device__ void append(int rold, float energy, float rbe, float weight, float density) {}
    } DeviceDensity;
    typedef struct DeviceDetector       {
        __inline__ __device__ bool isIonTarget() { return false; }
        __inline__ __device__ void append(int rold, float weight, float depo, int hid) {}
    } DeviceDetector;
    typedef struct DevicePhaseSpace     {} DevicePhaseSpace;
    typedef struct DeviceActivation     {} DeviceActivation;
    typedef struct DeviceDensityLETD    {
        __inline__ __device__ bool isIonTarget() { return false; }
        __inline__ __device__ void append(int rold, float energy, float let, float weight) {}
    };
    typedef struct DeviceMeshLETD       {
        __inline__ __device__ bool isIonTarget() { return false; }
        __inline__ __device__ void append(float3 pos, float energy, float let, float weight) {}
    } DeviceMeshLETD;
    typedef struct DeviceMeshActivation {
        __inline__ __device__ bool isIonTarget() { return false; }
        __inline__ __device__ void append(float3 pos, float energy, float let, float weight) {}
    } DeviceMeshActivation;
#endif


    typedef struct DeviceHandle {
        DeviceMeshTrack**      mesh_track;
        DeviceMeshDensity**    mesh_density;
        DeviceMeshActivation** mesh_activation;
        DeviceCross**          cross;
        DeviceTrack**          track;
        DeviceDensity**        density;
        DeviceDetector**       detector;
        DevicePhaseSpace**     phase_space;
        DeviceActivation**     activation;
        DeviceDensityLETD**    letd;
        DeviceMeshLETD**       mesh_letd;
        int                    n_mesh_track;
        int                    n_mesh_density;
        int                    n_mesh_activation;
        int                    n_cross;
        int                    n_track;
        int                    n_density;
        int                    n_detector;
        int                    n_phase_space;
        int                    n_activation;
        int                    n_letd;
        int                    n_mesh_letd;


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


