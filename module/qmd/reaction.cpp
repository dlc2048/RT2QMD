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
 * @file    module/qmd/reaction.cpp
 * @brief   QMD top level handler
 * @author  CM Lee
 * @date    02/15/2024
 */


#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "hadron/auxiliary.hpp"
#include "hadron/nucleus.hpp"

#include "mclog/logger.hpp"

#include "reaction.hpp"
#include "constants.hpp"
#include "config.hpp"

#include "collision.cuh"


namespace RT2QMD {


    void DeviceMemoryHandler::_initMetadataBufferSystem() {
        mclog::debug("Initialize QMD internal metadata queue ...");

        this->_n_metadata = 2 * this->_block;  // double size

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_dev_metadata_queue), 
            2 * sizeof(ModelMetadataQueue)));
        this->_memoryUsageAppend(2 * sizeof(ModelMetadataQueue));

        for (size_t i = 0; i < 2; ++i) {
            ModelMetadataQueue queue_host;
            queue_host.size = (int)this->_n_metadata;
            queue_host.head = !i ? queue_host.size : 0u;
            queue_host.tail = 0u;
            std::vector<int> idx_host(this->_n_metadata, 0);

            if (!i) {
                for (int j = 0; j < this->_n_metadata; ++j)
                    idx_host[j] = j;
            }

            mcutil::DeviceVectorHelper idx_dev(idx_host);
            this->_memoryUsageAppend(idx_dev.memoryUsage());
            queue_host.idx = idx_dev.address();

            CUDA_CHECK(cudaMemcpy(&this->_dev_metadata_queue[i], &queue_host, 
                sizeof(ModelMetadataQueue), cudaMemcpyHostToDevice));
        }

        std::vector<int> carray(this->_block, -1);
        mcutil::DeviceVectorHelper carray_dev(carray);
        this->_memoryUsageAppend(carray_dev.memoryUsage());
        this->_dev_current_metadata = carray_dev.address();

        CUDA_CHECK(Buffer::Metadata::setSymbolMetadata(
            (int)this->_n_metadata,
            this->_dev_metadata_queue,
            this->_dev_current_metadata
        ));

        return;
    }


    void DeviceMemoryHandler::_initTimer() {
        bool use_timer = Host::Config::getInstance().measureTime();
        if (use_timer) {
            this->_timer_counter = 0;
            this->_timer_out     = std::make_unique<std::ofstream>("exe_time.txt");
            this->_clock_host    = std::vector<long long int>(this->_block);

            mcutil::DeviceVectorHelper dev_vector(this->_clock_host);
            this->_memoryUsageAppend(dev_vector.memoryUsage());
            this->_clock_dev  = dev_vector.address();
        }
        CUDA_CHECK(setPtrTimer(use_timer, this->_clock_dev));
    }


    void DeviceMemoryHandler::_initProblemPoolBufferSystem(size_t n_problem_buffer) {
        mclog::debug("Initialize QMD internal memory pool ...");
        
        int    soa_stride_1d = (int)(this->_max_field_dimension);
        int    soa_stride_2d = (int)(this->_max_field_dimension * this->_max_field_dimension);

        size_t soa_size_1d = this->_n_metadata * this->_max_field_dimension;
        size_t soa_size_2d = this->_n_metadata * this->_max_field_dimension * this->_max_field_dimension;

        // initialize buffer priority mirror (pinned)
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>
            (&this->_priority_pinned_host), sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>
            (&this->_priority_pinned_dev), sizeof(int)));

        // problem buffer size
        // Initialize problem group
        this->_host_qmd_problem.resize(CUDA_WARP_SIZE);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_dev_qmd_problem),
            sizeof(mcutil::RingBuffer) * CUDA_WARP_SIZE));
        this->_memoryUsageAppend(sizeof(mcutil::RingBuffer) * CUDA_WARP_SIZE);

        for (size_t idx = 0; idx < (size_t)CUDA_WARP_SIZE; ++idx) {
            mcutil::RingBuffer& buffer_host = this->_host_qmd_problem[idx];
            buffer_host.size  = (int)n_problem_buffer;
            buffer_host.sizef = (float)buffer_host.size;
            buffer_host.head  = idx ? 0 : (int)(this->_block * BUFFER_CAP_THRESHOLD_MULTIPLIER);
            buffer_host.tail  = 0;

            if (!idx || idx >= this->_n_qmd_prob_group) {
                CUDA_CHECK(cudaMemcpy(&this->_dev_qmd_problem[idx], &buffer_host,
                    sizeof(mcutil::RingBuffer), cudaMemcpyHostToDevice));
                continue;
            }
                
            float** dest[8] = {
                &buffer_host.x,   &buffer_host.y,   &buffer_host.z,
                &buffer_host.u,   &buffer_host.v,   &buffer_host.w,
                &buffer_host.e,   &buffer_host.wee
            };
            for (size_t di = 0; di < 8; ++di) {
                float* soa;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&soa),
                    sizeof(float) * n_problem_buffer));
                CUDA_CHECK(cudaMemcpy(dest[di], &soa,
                    sizeof(float*), cudaMemcpyHostToHost));
                this->_memoryUsageAppend(sizeof(float) * n_problem_buffer);
            }

            unsigned int* soai;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&soai),
                sizeof(unsigned int) * n_problem_buffer));
            CUDA_CHECK(cudaMemcpy(&buffer_host.flags, &soai,
                sizeof(unsigned int*), cudaMemcpyHostToHost));
            this->_memoryUsageAppend(sizeof(unsigned int) * n_problem_buffer);

            // HID
            if (this->_buffer_has_hid) {
                unsigned int* hidi;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hidi),
                    sizeof(unsigned int) * n_problem_buffer));
                CUDA_CHECK(cudaMemcpy(&buffer_host.hid, &hidi,
                    sizeof(unsigned int*), cudaMemcpyHostToHost));
                this->_memoryUsageAppend(sizeof(unsigned int) * n_problem_buffer);
            }
            
            // ZA
            uchar4* za;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&za),
                sizeof(uchar4) * n_problem_buffer));
            CUDA_CHECK(cudaMemcpy(&buffer_host.za, &za,
                sizeof(uchar4*), cudaMemcpyHostToHost));
            this->_memoryUsageAppend(sizeof(uchar4) * n_problem_buffer);

            CUDA_CHECK(cudaMemcpy(&this->_dev_qmd_problem[idx], &buffer_host,
                sizeof(mcutil::RingBuffer), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(Buffer::setSymbolProblems(this->_dev_qmd_problem, (int)this->_n_qmd_prob_group));

        // Model bulk memory for child participants
        for (int i = 0; i < Buffer::Participant::SOA_MEMBERS; ++i) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_soa_list_p1[i]), 
                soa_size_1d * sizeof(float)));
#ifdef CUDA_THROW_NAN
            std::vector<unsigned int> soa_nan_debug(soa_size_1d, CUDA_NUMERIC_NAN);
            CUDA_CHECK(cudaMemcpy(this->_soa_list_p1[i], &soa_nan_debug[0],
                soa_size_1d * sizeof(float), cudaMemcpyHostToDevice));
#endif
            this->_memoryUsageAppend(soa_size_1d * sizeof(float));
        }
        CUDA_CHECK(Buffer::Participant::setSymbolBuffer(this->_soa_list_p1));

        // Model bulk memory for child field 2d
        for (int i = 0; i < Buffer::MeanField::SOA_MEMBERS_2D; ++i) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_soa_list_f2[i]), 
                soa_size_2d * sizeof(float)));
#ifdef CUDA_THROW_NAN
            std::vector<unsigned int> soa_nan_debug(soa_size_2d, CUDA_NUMERIC_NAN);
            CUDA_CHECK(cudaMemcpy(this->_soa_list_f2[i], &soa_nan_debug[0],
                soa_size_2d * sizeof(float), cudaMemcpyHostToDevice));
#endif
            this->_memoryUsageAppend(soa_size_2d * sizeof(float));
        }
        CUDA_CHECK(Buffer::MeanField::setSymbolBuffer2D(this->_soa_list_f2));

        // Model bulk memory for chield field 1d
        for (int i = 0; i < Buffer::MeanField::SOA_MEMBERS_1D; ++i) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_soa_list_f1[i]), 
                soa_size_1d * sizeof(float)));
#ifdef CUDA_THROW_NAN
            std::vector<unsigned int> soa_nan_debug(soa_size_1d, CUDA_NUMERIC_NAN);
            CUDA_CHECK(cudaMemcpy(this->_soa_list_f1[i], &soa_nan_debug[0],
                soa_size_1d * sizeof(float), cudaMemcpyHostToDevice));
#endif
            this->_memoryUsageAppend(soa_size_1d * sizeof(float));
        }
        CUDA_CHECK(Buffer::MeanField::setSymbolBuffer1D(this->_soa_list_f1));

        // Model buffer
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_dev_qmd_model_ptr),
            sizeof(QMDModel) * this->_n_metadata));
        this->_memoryUsageAppend(sizeof(QMDModel) * this->_n_metadata);
        for (int i = 0; i < this->_n_metadata; ++i) {
            QMDModel model_host;
            model_host.meta_idx      = i;
            model_host.initial_flags = mcutil::UNION_FLAGS();
            model_host.offset_1d     = i * soa_stride_1d;
            model_host.offset_2d     = i * soa_stride_2d;
            CUDA_CHECK(cudaMemcpy(&this->_dev_qmd_model_ptr[i], &model_host,
                sizeof(QMDModel), cudaMemcpyHostToDevice));
        }
        int model_caching_iter =
            (int)std::ceil((double)sizeof(QMDModel) / (double)(this->_thread * sizeof(float)));
        int field_iter =
            (int)std::ceil((double)this->_max_field_dimension / (double)this->_thread);
        CUDA_CHECK(Buffer::setSymbolModel(this->_dev_qmd_model_ptr, model_caching_iter, (int)this->_max_field_dimension, field_iter));
    }


    void DeviceMemoryHandler::_initDumpSystem() {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_dev_dump_ptr),
            sizeof(Buffer::ModelDumpBuffer) * this->_n_dump));
        this->_memoryUsageAppend(sizeof(Buffer::ModelDumpBuffer) * this->_n_dump);

        //set child particle buffer
        for (size_t i = 0; i < this->_n_dump; ++i) {
            Buffer::ModelDumpBuffer host;
            this->_memoryUsageAppend(host.malloc(this->_max_field_dimension));
            //CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&host.particles), 
            //    sizeof(ParticleDumpBuffer) * this->_max_field_dimension));
            //this->_memory_share += sizeof(ParticleDumpBuffer) * this->_max_field_dimension;
            CUDA_CHECK(cudaMemcpy(&this->_dev_dump_ptr[i], &host,
                sizeof(Buffer::ModelDumpBuffer), cudaMemcpyHostToDevice));
        }

        // set device symbol
        CUDA_CHECK(setSymbolDumpBuffer(this->_dev_dump_ptr, (int)this->_n_dump));
    }


    void DeviceMemoryHandler::_measureTime(const std::string& method_name, size_t block, size_t thread) {
        if (Host::Config::getInstance().measureTime()) {
            if (this->_timer_counter < Host::Config::getInstance().timerSize()) {
                CUDA_CHECK(cudaMemcpy(&this->_clock_host[0], this->_clock_dev,
                    sizeof(long long int) * this->_clock_host.size(), cudaMemcpyDeviceToHost));
                *this->_timer_out << method_name << "<<<" << block << "," << thread << ">>>" << std::endl;

                // time
                mclog::FormattedTable fmt_table(std::vector<size_t>(8, 12));
                for (size_t i = 0; i < this->_clock_host.size(); i += 8) {
                    fmt_table.clear();
                    for (size_t j = i; j < std::min(this->_clock_host.size(), i + 8); j++)
                        fmt_table << this->_clock_host[j];
                    *this->_timer_out << fmt_table.str() << std::endl;
                }

                this->_timer_counter++;
            }
        }
        return;
    }

#ifdef RT2QMD_STANDALONE


    DeviceMemoryHandler::DeviceMemoryHandler(
        bool buffer_has_hid,
        const mcutil::DeviceController& device_settings,
        size_t max_dimension
    ) : _block(device_settings.blockQMD()),
        _thread(device_settings.threadQMD()),
        _n_dump(0),
        _buffer_has_hid(buffer_has_hid),
        _max_field_dimension(max_dimension),
        _clock_dev(nullptr) {

        constants::ParameterInitializer();

        mclog::debug("Initialize QMD model data ...");

        this->_n_qmd_prob_group = (size_t)std::ceil((double)this->_max_field_dimension / (double)CUDA_WARP_SIZE) + 1;

        size_t problem_buffer_dim = (size_t)(device_settings.block() * device_settings.thread() * mcutil::BUFFER_RELEASE_MARGIN);

        // initialize buffer
        this->_initMetadataBufferSystem();
        this->_initProblemPoolBufferSystem(problem_buffer_dim);

        // initialize timer
        this->_initTimer();

        // initialize dump
        if (Host::Config::getInstance().doDumpAction()) {
            this->_n_dump = (size_t)Host::Config::getInstance().dumpSize();
            this->_initDumpSystem();
        }

        CUDA_CHECK(RT2QMD::setMassTableHandle(Nucleus::MassTableHandler::getInstance().deviceptr()));

        // initialize NN collision table
        bool use_incl_model = Host::Config::getInstance().usingINCLNNScattering();
        CUdeviceptr nn_table = 0x0u;
        if (!use_incl_model)
            nn_table = Hadron::NNScatteringTableHandler::getInstance().deviceptr();
        CUDA_CHECK(Collision::setNNTable(nn_table, use_incl_model));
        
    }

#else


    DeviceMemoryHandler::DeviceMemoryHandler(
        bool buffer_has_hid,
        const mcutil::DeviceController& device_settings,
        mat::LogicalMaterialHandler&    mat_handle
    ) : _block(device_settings.blockQMD()), 
        _thread(device_settings.threadQMD()), 
        _n_dump(0), 
        _buffer_has_hid(buffer_has_hid) {

        constants::ParameterInitializer();

        mclog::debug("Initialize QMD model data ...");

        // calculate the maximum field size
        // target
        int max_a_target = 0;
        for (int za : mat_handle.targetIsotopes()) {
            int a = physics::getAnumberFromZA(za);
            max_a_target = std::max(a, max_a_target);
        }
        // projectile
        int max_a_proj = 0;
        genion::IsoProjectileTable& proj_table = genion::IsoProjectileTable::getInstance();
        for (int za : proj_table.listProjectileZA()) {
            int a = physics::getAnumberFromZA(za);
            max_a_proj = std::max(a, max_a_proj);
        }
        this->_max_field_dimension = (size_t)max_a_target + (size_t)max_a_proj;
        this->_n_qmd_prob_group    = (size_t)std::ceil((double)this->_max_field_dimension / (double)CUDA_WARP_SIZE) + 1;

        size_t problem_buffer_dim = (size_t)(device_settings.block() * device_settings.thread() * mcutil::BUFFER_RELEASE_MARGIN);
        
        // initialize buffer
        this->_initMetadataBufferSystem();
        this->_initProblemPoolBufferSystem(problem_buffer_dim);

        // initialize dump
        if (Host::Config::getInstance().doDumpAction()) {
            this->_n_dump = (size_t)Host::Config::getInstance().dumpSize();
            this->_initDumpSystem();
        }

        CUDA_CHECK(RT2QMD::setMassTableHandle(Nucleus::MassTableHandler::getInstance().deviceptr()));

        // initialize NN collision table
        bool use_incl_model = Host::Config::getInstance().usingINCLNNScattering();
        CUdeviceptr nn_table = 0x0u;
        if (!use_incl_model)
            nn_table = Hadron::NNScatteringTableHandler::getInstance().deviceptr();
        CUDA_CHECK(Collision::setNNTable(nn_table, use_incl_model));
    }


#endif


    DeviceMemoryHandler::~DeviceMemoryHandler() {
        if (Host::Config::getInstance().doDumpAction()) {
            this->writeDumpData("QMD_dump.bin");
        }
        // free dump
        if (Host::Config::getInstance().doDumpAction()) {
            for (size_t i = 0; i < this->_n_dump; ++i) {
                Buffer::ModelDumpBuffer host;
                CUDA_CHECK(cudaMemcpy(&host, &this->_dev_dump_ptr[i],
                    sizeof(Buffer::ModelDumpBuffer), cudaMemcpyDeviceToHost));
                host.free();
            }
            CUDA_CHECK(cudaFree(this->_dev_dump_ptr));
        }
    }


    std::vector<Buffer::ModelDumpHost> DeviceMemoryHandler::getDumpData() const {
        int n_dump;
        CUDA_CHECK(Buffer::getSymbolDumpBuffer(&n_dump));
        std::vector<Buffer::ModelDumpHost> arr_dump;
        for (int i = 0; i < n_dump; ++i)
            arr_dump.push_back(Buffer::ModelDumpHost(&this->_dev_dump_ptr[i]));
        return arr_dump;
    }


    void DeviceMemoryHandler::writeDumpData(const std::string& file_name) const {
        mcutil::FortranOfstream stream(file_name);
        std::vector<Buffer::ModelDumpHost> dump_list = this->getDumpData();
        int size = (int)dump_list.size();
        stream.write(reinterpret_cast<unsigned char*>(&size),
            sizeof(int));
        for (const Buffer::ModelDumpHost& dump : dump_list)
            dump.write(stream);
    }



    int DeviceMemoryHandler::getBufferPriority() {
        Buffer::__host__deviceGetQMDPriority(
            this->_n_qmd_prob_group,
            this->_dev_qmd_problem,
            this->_priority_pinned_dev
        );
        cudaMemcpy(this->_priority_pinned_host, this->_priority_pinned_dev,
            sizeof(int), cudaMemcpyDeviceToHost);
        return this->_priority_pinned_host[0];
    }


    void DeviceMemoryHandler::summary() const {

        mclog::info("*** QMD Model Summaries ***");
        
    }


    void DeviceMemoryHandler::reset() {
        __host__deviceResetModelBuffer(this->_block, this->_thread, false);
    }


    bool DeviceMemoryHandler::launch() {
        // check whether sufficient number of event is collected to buffer or not
        int buffer_idx = this->getBufferPriority();
        if (!buffer_idx)
            return false;

        // reset data (return data = true)
        __host__deviceResetModelBuffer(this->_block, this->_thread, true);

        while (true) {

#ifdef RT2QMD_PERSISTENT_DISPATCHER
            CUDA_CHECK(__host__fieldDispatcher(this->_block, this->_thread, buffer_idx));  // initialize QMD field
            this->_measureTime("__host__fieldDispatcher", this->_block, this->_thread);
            CUDA_CHECK(__host__pullEligibleField(this->_block, this->_thread));            // pull eligible field from idx queue
            this->_measureTime("__host__pullEligibleField", this->_block, this->_thread);
#else
            CUDA_CHECK(__host__prepareModel(this->_block, this->_thread, buffer_idx));
            this->_measureTime("__host__prepareModel", this->_block, this->_thread);
            CUDA_CHECK(__host__prepareProjectile(this->_block, this->_thread));
            this->_measureTime("__host__prepareProjectile", this->_block, this->_thread);
            CUDA_CHECK(__host__prepareTarget(this->_block, this->_thread));
            this->_measureTime("__host__prepareTarget", this->_block, this->_thread);
#endif
            CUDA_CHECK(__host__propagate(this->_block, this->_thread));                    // do propagate
            CUDA_CHECK(__host__finalize(this->_block, this->_thread, buffer_idx));         // finalize

            // __host__deviceResetModelBuffer(this->_block, this->_thread, false);

            buffer_idx = this->getBufferPriority();
            if (!buffer_idx)
                break;
        }

        return true;
    }


}