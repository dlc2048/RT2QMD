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
 * @file    module/qmd/reaction.hpp
 * @brief   QMD top level handler
 * @author  CM Lee
 * @date    02/15/2024
 */


#pragma once

#include "device/tuning.hpp"
#include "device/shuffle.cuh"
#include "fortran/fortran.hpp"

#include "transport/buffer.hpp"

#ifndef RT2QMD_STANDALONE
#include "material/material.hpp"
#endif

#include "buffer.hpp"
#include "reaction.cuh"


namespace RT2QMD {


    constexpr size_t STANDALONE_BUFFER_SIZE_MULTIPLIER = 100;
    constexpr size_t BUFFER_CAP_THRESHOLD_MULTIPLIER   = 10;
    constexpr size_t BUFFER_CAP_MINIMUM_MULTIPLIER     = 2;

    constexpr size_t PULL_FIELD_MAX_COUNT = 50000;


    struct FlushPhaseSpaceStruct {
        short2 za;
        float  eke;
        float  excitation;
        float3 direction;
    };


    class DeviceMemoryHandler : public mcutil::DeviceMemoryHandlerInterface {
    private:
        size_t _block;    // kernel block size
        size_t _thread;   // kernel thread size

        // execution time measurement
        std::unique_ptr<std::ofstream> _timer_out;
        int _timer_counter;
        long long int* _clock_dev;
        std::vector<long long int> _clock_host;

        bool _use_timer;
        int  _timer_size;

        // global problem
        size_t _max_field_dimension;  // maximum field dimension (nxn)
        size_t _n_qmd_prob_group;     // number of QMD problem group (dimension32, 64 ...)

        std::vector<mcutil::RingBuffer> _host_qmd_problem;
        mcutil::RingBuffer* _dev_qmd_problem;  // device QMD problem lists (dimension32, 64 ...)

        // metadata
        size_t _n_metadata;  // maximum number of the metadata
        ModelMetadataQueue* _dev_metadata_queue;  // idle index queue
        int* _dev_current_metadata;

        // dump related
        bool   _do_dump_action;  // do dump action
        size_t _n_dump;          // dump size
        bool   _buffer_has_hid;  // History id activation flag
        
        Buffer::ModelDumpBuffer* _dev_dump_ptr;  // device dump pointer

        // pool 
        QMDModel* _dev_qmd_model_ptr;   // device QMD model pointer

        void* _soa_list_p1[Buffer::Participant::SOA_MEMBERS];   // one-dimensional participant element, device pointer
        void* _soa_list_f2[Buffer::MeanField::SOA_MEMBERS_2D];  // two-dimensional field element, device pointer
        void* _soa_list_f1[Buffer::MeanField::SOA_MEMBERS_1D];  // one-dimensional field element, device pointer

        // buffer priority
        int* _priority_pinned_host;
        int* _priority_pinned_dev;
        
        // cluster test (participant)
        // from here ...

        
        std::unique_ptr<mcutil::FortranOfunformatted> _cluster_dump;
        size_t _pull_counter;


        void _pullEligibleFields();


        // ... to here


        void _initMetadataBufferSystem();


        void _initTimer();


        void _initProblemPoolBufferSystem(size_t n_problem_buffer);


        void _initDumpSystem();


        // std::unique_ptr<Buffer::DeviceMemoryHandler> _handle_buffer;


        void _measureTime(const std::string& method_name, size_t block, size_t thread);


    public:

#ifdef RT2QMD_STANDALONE


        DeviceMemoryHandler(
            bool buffer_has_hid,
            const mcutil::DeviceController& device_settings,
            size_t max_dimension
        );


#else


        DeviceMemoryHandler(
            bool buffer_has_hid,
            const mcutil::DeviceController& device_settings,
            mat::LogicalMaterialHandler&    mat_handle
        );


#endif


        ~DeviceMemoryHandler();


        void setIonBuffer(CUdeviceptr raw_ptr, size_t size);


        void setRandStateBuffer(CUdeviceptr raw_ptr);


        std::vector<Buffer::ModelDumpHost> getDumpData() const;


        void writeDumpData(const std::string& file_name) const;


        CUdeviceptr ptrProblemBuffer() const { return reinterpret_cast<CUdeviceptr>(this->_dev_qmd_problem); }


        int numberOfBuffer() const { return (int)this->_n_qmd_prob_group; }


        int getBufferPriority();


        static void free();


        void summary() const;


        void reset();


        /**
        * @brief Launch QMD chain
        * @return true if QMD chain is launched, false elsewhere
        */
        bool launch();


    };


}