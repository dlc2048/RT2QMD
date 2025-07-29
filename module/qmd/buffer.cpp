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
 * @file    module/qmd/buffer.cpp
 * @brief   Host side buffer initialization for the QMD
 * @author  CM Lee
 * @date    02/14/2024
 */


#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include <math.h>

#include "buffer.hpp"
#include "config.hpp"


namespace RT2QMD {
    namespace Buffer {


        __host__ size_t ModelDumpBuffer::malloc(size_t size_particle) {
            size_t memshare = 0;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->particles),
                sizeof(ModelDumpBuffer) * size_particle));
            memshare += sizeof(ModelDumpBuffer) * size_particle;

            // mean field size
            size_t size_2d = sizeof(float) * size_particle * size_particle;
            size_t size_1d = sizeof(float) * size_particle;

            // 2d
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.rr2),  size_2d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.pp2),  size_2d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.rbij), size_2d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.rha),  size_2d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.rhe),  size_2d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.rhc),  size_2d));
            memshare += size_2d * 6;

            // 1d
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.ffrx), size_1d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.ffry), size_1d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.ffrz), size_1d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.ffpx), size_1d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.ffpy), size_1d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.ffpz), size_1d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.f0rx), size_1d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.f0ry), size_1d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.f0rz), size_1d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.f0px), size_1d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.f0py), size_1d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.f0pz), size_1d));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->mean_field.rh3d), size_1d));
            memshare += size_1d * 13;
            return memshare;
        }


        __host__ void ModelDumpBuffer::free() {
            CUDA_CHECK(cudaFree(this->particles));

            CUDA_CHECK(cudaFree(this->mean_field.rr2));
            CUDA_CHECK(cudaFree(this->mean_field.pp2));
            CUDA_CHECK(cudaFree(this->mean_field.rbij));
            CUDA_CHECK(cudaFree(this->mean_field.rha));
            CUDA_CHECK(cudaFree(this->mean_field.rhe));
            CUDA_CHECK(cudaFree(this->mean_field.rhc));

            CUDA_CHECK(cudaFree(this->mean_field.ffrx));
            CUDA_CHECK(cudaFree(this->mean_field.ffry));
            CUDA_CHECK(cudaFree(this->mean_field.ffrz));
            CUDA_CHECK(cudaFree(this->mean_field.ffpx));
            CUDA_CHECK(cudaFree(this->mean_field.ffpy));
            CUDA_CHECK(cudaFree(this->mean_field.ffpz));
            CUDA_CHECK(cudaFree(this->mean_field.f0rx));
            CUDA_CHECK(cudaFree(this->mean_field.f0ry));
            CUDA_CHECK(cudaFree(this->mean_field.f0rz));
            CUDA_CHECK(cudaFree(this->mean_field.f0px));
            CUDA_CHECK(cudaFree(this->mean_field.f0py));
            CUDA_CHECK(cudaFree(this->mean_field.f0pz));
            CUDA_CHECK(cudaFree(this->mean_field.rh3d));
        }


        ModelDumpHost::ModelDumpHost(ModelDumpBuffer* dev) {
            ModelDumpBuffer host;
            CUDA_CHECK(cudaMemcpy(&host, dev, 
                sizeof(ModelDumpBuffer), cudaMemcpyDeviceToHost));
            this->_file     = std::string(host.file);
            this->_function = std::string(host.function);
            this->_line     = host.line;
            this->_model    = host.model;
            // mean field
            size_t size_1d = host.model.current_field_size;
            size_t size_2d = host.model.current_field_size_2;

            // 2D
            this->_rr2.resize(size_2d);
            this->_pp2.resize(size_2d);
            this->_rbij.resize(size_2d);
            this->_rha.resize(size_2d);
            this->_rhe.resize(size_2d);
            this->_rhc.resize(size_2d);
            CUDA_CHECK(cudaMemcpy(&this->_rr2[0],  host.mean_field.rr2, 
                sizeof(float) * size_2d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_pp2[0],  host.mean_field.pp2,
                sizeof(float) * size_2d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_rbij[0], host.mean_field.rbij,
                sizeof(float) * size_2d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_rha[0],  host.mean_field.rha,
                sizeof(float) * size_2d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_rhe[0],  host.mean_field.rhe,
                sizeof(float) * size_2d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_rhc[0],  host.mean_field.rhc,
                sizeof(float) * size_2d, cudaMemcpyDeviceToHost));

            // 1D
            this->_ffrx.resize(size_1d);
            this->_ffry.resize(size_1d);
            this->_ffrz.resize(size_1d);
            this->_ffpx.resize(size_1d);
            this->_ffpy.resize(size_1d);
            this->_ffpz.resize(size_1d);
            this->_f0rx.resize(size_1d);
            this->_f0ry.resize(size_1d);
            this->_f0rz.resize(size_1d);
            this->_f0px.resize(size_1d);
            this->_f0py.resize(size_1d);
            this->_f0pz.resize(size_1d);
            this->_rh3d.resize(size_1d);
            CUDA_CHECK(cudaMemcpy(&this->_ffrx[0], host.mean_field.ffrx,
                sizeof(float) * size_1d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_ffry[0], host.mean_field.ffry,
                sizeof(float) * size_1d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_ffrz[0], host.mean_field.ffrz,
                sizeof(float) * size_1d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_ffpx[0], host.mean_field.ffpx,
                sizeof(float) * size_1d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_ffpy[0], host.mean_field.ffpy,
                sizeof(float) * size_1d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_ffpz[0], host.mean_field.ffpz,
                sizeof(float) * size_1d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_f0rx[0], host.mean_field.f0rx,
                sizeof(float) * size_1d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_f0ry[0], host.mean_field.f0ry,
                sizeof(float) * size_1d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_f0rz[0], host.mean_field.f0rz,
                sizeof(float) * size_1d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_f0px[0], host.mean_field.f0px,
                sizeof(float) * size_1d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_f0py[0], host.mean_field.f0py,
                sizeof(float) * size_1d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_f0pz[0], host.mean_field.f0pz,
                sizeof(float) * size_1d, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&this->_rh3d[0], host.mean_field.rh3d,
                sizeof(float) * size_1d, cudaMemcpyDeviceToHost));

            // particles
            for (int i = 0; i < host.particle_size; ++i) {
                ParticleDumpBuffer host_particle;
                CUDA_CHECK(cudaMemcpy(&host_particle, &host.particles[i],
                    sizeof(ParticleDumpBuffer), cudaMemcpyDeviceToHost));
                this->_particles.push_back(host_particle);
            }

            return;
        };


        void ModelDumpHost::write(mcutil::FortranOfstream& stream) const {
            std::vector<char> file(this->_file.begin(), this->_file.end());
            stream.write(file);
            std::vector<char> func(this->_function.begin(), this->_function.end());
            stream.write(func);
            stream.write(reinterpret_cast<const unsigned char*>(&this->_line),  sizeof(int));
            stream.write(reinterpret_cast<const unsigned char*>(&this->_model), sizeof(QMDModel));
            stream.write(this->_rr2);
            stream.write(this->_pp2);
            stream.write(this->_rbij);
            stream.write(this->_rha);
            stream.write(this->_rhe);
            stream.write(this->_rhc);
            stream.write(this->_ffrx);
            stream.write(this->_ffry);
            stream.write(this->_ffrz);
            stream.write(this->_ffpx);
            stream.write(this->_ffpy);
            stream.write(this->_ffpz);
            stream.write(this->_f0rx);
            stream.write(this->_f0ry);
            stream.write(this->_f0rz);
            stream.write(this->_f0px);
            stream.write(this->_f0py);
            stream.write(this->_f0pz);
            stream.write(this->_rh3d);
            stream.write(this->_particles);
        }


        void DeviceMemoryHandler::_initBufferSystem() {
            size_t soa_size_1d   = sizeof(float) * this->_block * this->_max_field_dimension;
            size_t soa_size_2d   = sizeof(float) * this->_block * this->_max_field_dimension * this->_max_field_dimension;
            int    soa_stride_1d = (int)(this->_max_field_dimension);
            int    soa_stride_2d = (int)(this->_max_field_dimension * this->_max_field_dimension);

            // Model bulk memory for child participants
            for (int i = 0; i < Participant::SOA_MEMBERS; ++i) {
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_soa_list_p1[i]), soa_size_1d));
#ifndef NDEBUG
                std::vector<unsigned int> soa_nan_debug(this->_block * this->_max_field_dimension, CUDA_NUMERIC_NAN);
                CUDA_CHECK(cudaMemcpy(this->_soa_list_p1[i], &soa_nan_debug[0], 
                    soa_size_1d, cudaMemcpyHostToDevice));
#endif
                this->_memory_share += soa_size_1d;
            }
            CUDA_CHECK(Participant::setSymbolBuffer(this->_soa_list_p1));

            // Model bulk memory for child field 2d
            for (int i = 0; i < MeanField::SOA_MEMBERS_2D; ++i) {
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_soa_list_f2[i]), soa_size_2d));
#ifndef NDEBUG
                std::vector<unsigned int> soa_nan_debug(this->_block * this->_max_field_dimension * this->_max_field_dimension, CUDA_NUMERIC_NAN);
                CUDA_CHECK(cudaMemcpy(this->_soa_list_f2[i], &soa_nan_debug[0],
                    soa_size_2d, cudaMemcpyHostToDevice));
#endif
                this->_memory_share += soa_size_2d;
            }
            CUDA_CHECK(MeanField::setSymbolBuffer2D(this->_soa_list_f2));

            // Model bulk memory for chield field 1d
            for (int i = 0; i < MeanField::SOA_MEMBERS_1D; ++i) {
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_soa_list_f1[i]), soa_size_1d));
#ifndef NDEBUG
                std::vector<unsigned int> soa_nan_debug(this->_block * this->_max_field_dimension, CUDA_NUMERIC_NAN);
                CUDA_CHECK(cudaMemcpy(this->_soa_list_f1[i], &soa_nan_debug[0],
                    soa_size_1d, cudaMemcpyHostToDevice));
#endif
                this->_memory_share += soa_size_1d;
            }
            CUDA_CHECK(MeanField::setSymbolBuffer1D(this->_soa_list_f1));

            // Model buffer
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_dev_qmd_model_ptr), 
                sizeof(QMDModel) * this->_block));
            this->_memory_share += sizeof(QMDModel) * this->_block;
            for (int i = 0; i < this->_block; ++i) {
                QMDModel model_host;
                model_host.offset_1d = i * soa_stride_1d;
                model_host.offset_2d = i * soa_stride_2d;
                CUDA_CHECK(cudaMemcpy(&this->_dev_qmd_model_ptr[i], &model_host,
                    sizeof(QMDModel), cudaMemcpyHostToDevice));
            }
            int model_caching_iter = 
                (int)std::ceil((double)sizeof(QMDModel) / (double)(this->_thread * sizeof(float)));
            int field_iter         = 
                (int)std::ceil((double)this->_max_field_dimension / (double)this->_thread);
            CUDA_CHECK(setSymbolModel(this->_dev_qmd_model_ptr, model_caching_iter, (int)this->_max_field_dimension, field_iter));
        }


        void DeviceMemoryHandler::_initDumpSystem() {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_dev_dump_ptr),
                sizeof(ModelDumpBuffer) * this->_n_dump));
            this->_memory_share += sizeof(ModelDumpBuffer) * this->_n_dump;

            //set child particle buffer
            for (size_t i = 0; i < this->_n_dump; ++i) {
                ModelDumpBuffer host;
                this->_memory_share += host.malloc(this->_max_field_dimension);
                //CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&host.particles), 
                //    sizeof(ParticleDumpBuffer) * this->_max_field_dimension));
                //this->_memory_share += sizeof(ParticleDumpBuffer) * this->_max_field_dimension;
                CUDA_CHECK(cudaMemcpy(&this->_dev_dump_ptr[i], &host,
                    sizeof(ModelDumpBuffer), cudaMemcpyHostToDevice));
            }

            // set device symbol
            CUDA_CHECK(setSymbolDumpBuffer(this->_dev_dump_ptr, (int)this->_n_dump));
        }


        DeviceMemoryHandler::DeviceMemoryHandler(size_t block, size_t thread, size_t max_field_dimension)
            : _block(block), _thread(thread), _max_field_dimension(max_field_dimension), _memory_share(0), _n_dump(0) {

            // initialize buffer
            this->_initBufferSystem();

            // initialize dump
            if (Host::Config::getInstance().doDumpAction()) {
                this->_n_dump = (size_t)Host::Config::getInstance().dumpSize();
                this->_initDumpSystem();
            }

        }


        DeviceMemoryHandler::~DeviceMemoryHandler() {

            // free dump
            if (Host::Config::getInstance().doDumpAction()) {
                for (size_t i = 0; i < this->_n_dump; ++i) {
                    ModelDumpBuffer host;
                    CUDA_CHECK(cudaMemcpy(&host, &this->_dev_dump_ptr[i],
                        sizeof(ModelDumpBuffer), cudaMemcpyDeviceToHost));
                    host.free();
                }   
                CUDA_CHECK(cudaFree(this->_dev_dump_ptr));
            }

            // free participant SOA
            for (int i = 0; i < Participant::SOA_MEMBERS; ++i)
                CUDA_CHECK(cudaFree(this->_soa_list_p1[i]));

            // free mean field SOA 2d
            for (int i = 0; i < MeanField::SOA_MEMBERS_2D; ++i)
                CUDA_CHECK(cudaFree(this->_soa_list_f2[i]));

            // free mean field SOA 1d
            for (int i = 0; i < MeanField::SOA_MEMBERS_1D; ++i)
                CUDA_CHECK(cudaFree(this->_soa_list_f1[i]));

            // free model
            CUDA_CHECK(cudaFree(this->_dev_qmd_model_ptr));
        }


        std::vector<ModelDumpHost> DeviceMemoryHandler::getDumpData() const {
            int n_dump;
            CUDA_CHECK(getSymbolDumpBuffer(&n_dump));
            std::vector<ModelDumpHost> arr_dump;
            for (int i = 0; i < n_dump; ++i)
                arr_dump.push_back(ModelDumpHost(&this->_dev_dump_ptr[i]));
            return arr_dump;
        }


        void DeviceMemoryHandler::writeDumpData(const std::string& file_name) const {
            mcutil::FortranOfstream stream(file_name);
            std::vector<ModelDumpHost> dump_list = this->getDumpData();
            int size = (int)dump_list.size();
            stream.write(reinterpret_cast<unsigned char*>(&size),
                sizeof(int));
            for (const ModelDumpHost& dump : dump_list)
                dump.write(stream);
        }


    }
}