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
 * @file    module/scoring/tally.cpp
 * @brief   Data transfer and management between host and device
 * @author  CM Lee
 * @date    05/23/2023
 */


#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "tally.hpp"


namespace tally {


    __host__ void DeviceHandle::Deleter(DeviceHandle* ptr) {
        DeviceHandle host;
        CUDA_CHECK(cudaMemcpy(&host, ptr, sizeof(DeviceHandle), cudaMemcpyDeviceToHost));
        if (host.n_mesh_track)      CUDA_CHECK(cudaFree(host.mesh_track));
        if (host.n_mesh_density)    CUDA_CHECK(cudaFree(host.mesh_density));
        if (host.n_mesh_activation) CUDA_CHECK(cudaFree(host.mesh_activation));
        if (host.n_cross)           CUDA_CHECK(cudaFree(host.cross));
        if (host.n_track)           CUDA_CHECK(cudaFree(host.track));
        if (host.n_density)         CUDA_CHECK(cudaFree(host.density));
        if (host.n_detector)        CUDA_CHECK(cudaFree(host.detector));
        if (host.n_phase_space)     CUDA_CHECK(cudaFree(host.phase_space));
        if (host.n_activation)      CUDA_CHECK(cudaFree(host.activation));
        if (host.n_letd)            CUDA_CHECK(cudaFree(host.letd));
        if (host.n_mesh_letd)       CUDA_CHECK(cudaFree(host.mesh_letd));
        CUDA_CHECK(cudaFree(ptr));
    }


#ifndef RT2QMD_STANDALONE


    template <>
    void TallyHostDevicePair<MeshDensity, DeviceMeshDensity>::pullFromDevice(double total_weight) {
        this->_pullFromDeviceMesh(total_weight);
    }


    template <>
    void TallyHostDevicePair<MeshTrack, DeviceMeshTrack>::pullFromDevice(double total_weight) {
        this->_pullFromDeviceMesh(total_weight);
    }


    template <>
    void TallyHostDevicePair<MeshActivation, DeviceMeshActivation>::pullFromDevice(double total_weight) {
        this->_pullFromDeviceMesh(total_weight);
    }


    template <>
    void TallyHostDevicePair<MeshLETD, DeviceMeshLETD>::pullFromDevice(double total_weight) {
        this->_pullFromDeviceMesh(total_weight);
    }


#endif


}