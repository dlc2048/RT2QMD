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
 * @file    module/qmd/buffer.cu
 * @brief   Device side internal queue for QMD
 * @author  CM Lee
 * @date    02/14/2024
 */


#include "buffer.cuh"
#include "device/shuffle.cuh"

#include <stdio.h>


namespace RT2QMD {


    __constant__ bool BUFFER_HAS_HID;

    __device__ curandState* rand_state;
    __device__ mcutil::RingBuffer* buffer_catalog;

    __device__ Nucleus::MassTable* mass_table;

    __device__ float* barashenkov_corr[2];


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


    __host__ cudaError_t setBarashenkovPtr(float* corr_neutron, float* corr_proton) {
        float* corr[2] = { corr_neutron, corr_proton };
        return cudaMemcpyToSymbol(barashenkov_corr[0], corr, sizeof(float*) * 2);
    }


    namespace Buffer {


        __constant__ int MAX_DIMENSION_PARTICIPANT;
        __constant__ int PARTICIPANT_MAX_ITER;


        namespace Metadata {


            __constant__ int N_METADATA;
            __device__ ModelMetadataQueue* meta_queue;
            __device__ int* current_metadata_idx;


            __host__ cudaError_t setSymbolMetadata(int n_metadata, ModelMetadataQueue* ptr_queue, int* ptr_idx_array) {
                M_SOAPtrMapper(int, n_metadata, N_METADATA);
                M_SOAPtrMapper(ModelMetadataQueue*, ptr_queue, meta_queue);
                M_SOAPtrMapper(int*, ptr_idx_array, current_metadata_idx);
                return cudaSuccess;
            }


        }


        namespace Participant {


            __device__ int*   flags;
            __device__ float* mass;
            __device__ float* position_x;
            __device__ float* position_y;
            __device__ float* position_z;
            __device__ float* momentum_x;
            __device__ float* momentum_y;
            __device__ float* momentum_z;

            __device__ float* ps[SOA_MEMBERS];


            __host__ cudaError_t setSymbolBuffer(void* soa_list[]) {
                cudaError_t res;
                res = cudaMemcpyToSymbol(ps, soa_list, sizeof(float*) * SOA_MEMBERS);
                if (res != cudaSuccess) return res;
                res = cudaMemcpyToSymbol(flags, &soa_list[0], sizeof(int*));
                if (res != cudaSuccess) return res;
                res = cudaMemcpyToSymbol(mass, &soa_list[1], sizeof(float*));
                if (res != cudaSuccess) return res;
                res = cudaMemcpyToSymbol(position_x, &soa_list[2], sizeof(float*));
                if (res != cudaSuccess) return res;
                res = cudaMemcpyToSymbol(position_y, &soa_list[3], sizeof(float*));
                if (res != cudaSuccess) return res;
                res = cudaMemcpyToSymbol(position_z, &soa_list[4], sizeof(float*));
                if (res != cudaSuccess) return res;
                res = cudaMemcpyToSymbol(momentum_x, &soa_list[5], sizeof(float*));
                if (res != cudaSuccess) return res;
                res = cudaMemcpyToSymbol(momentum_y, &soa_list[6], sizeof(float*));
                if (res != cudaSuccess) return res;
                return cudaMemcpyToSymbol(momentum_z, &soa_list[7], sizeof(float*));
            }


        }


        namespace MeanField {


            __device__ float* rr2;
            __device__ float* pp2;
            __device__ float* rbij;
            __device__ float* rha;
            __device__ float* rhe;
            __device__ float* rhc;


            __device__ float* ffrx;
            __device__ float* ffry;
            __device__ float* ffrz;
            __device__ float* ffpx;
            __device__ float* ffpy;
            __device__ float* ffpz;
            __device__ float* f0rx;
            __device__ float* f0ry;
            __device__ float* f0rz;
            __device__ float* f0px;
            __device__ float* f0py;
            __device__ float* f0pz;
            __device__ float* rh3d;


            __host__ cudaError_t setSymbolBuffer2D(void* soa_list[]) {
                M_SOAPtrMapper(float*, soa_list[0], rr2);
                M_SOAPtrMapper(float*, soa_list[1], pp2);
                M_SOAPtrMapper(float*, soa_list[2], rbij);
                M_SOAPtrMapper(float*, soa_list[3], rha);
                M_SOAPtrMapper(float*, soa_list[4], rhe);
                M_SOAPtrMapper(float*, soa_list[5], rhc);
                return cudaSuccess;
            }


            __host__ cudaError_t setSymbolBuffer1D(void* soa_list[]) {
                M_SOAPtrMapper(float*, soa_list[0],  ffrx);
                M_SOAPtrMapper(float*, soa_list[1],  ffry);
                M_SOAPtrMapper(float*, soa_list[2],  ffrz);
                M_SOAPtrMapper(float*, soa_list[3],  ffpx);
                M_SOAPtrMapper(float*, soa_list[4],  ffpy);
                M_SOAPtrMapper(float*, soa_list[5],  ffpz);
                M_SOAPtrMapper(float*, soa_list[6],  f0rx);
                M_SOAPtrMapper(float*, soa_list[7],  f0ry);
                M_SOAPtrMapper(float*, soa_list[8],  f0rz);
                M_SOAPtrMapper(float*, soa_list[9],  f0px);
                M_SOAPtrMapper(float*, soa_list[10], f0py);
                M_SOAPtrMapper(float*, soa_list[11], f0pz);
                M_SOAPtrMapper(float*, soa_list[12], rh3d);
                return cudaSuccess;
            }


        }


        __device__ QMDModel*           models;
        __device__ mcutil::RingBuffer* qmd_problems;
        __device__ ModelDumpBuffer*    model_dump;

        __constant__ int N_PROBLEM_BUFFERS;

        __constant__ int GLOBAL_INC_BUFFER_SIZE;
        __constant__ int GLOBAL_DUMP_BUFFER_SIZE;
        __device__   int GLOBAL_MODEL_COUNTER;
        __device__   int CURRENT_DUMP_IDX;
        __constant__ int MODEL_CACHING_ITER;

        /*
        __device__ void readModelFromBuffer(int meta_idx) {
            assert(meta_idx >= 0 && meta_idx < Metadata::N_METADATA);
            float* cache  = reinterpret_cast<float*>(model_cached);
            float* buffer = reinterpret_cast<float*>(&models[meta_idx]);
            for (int i = 0; i < MODEL_CACHING_ITER; ++i) {
                int idx = i * blockDim.x + threadIdx.x;
                if (idx < MODEL_CACHING_SIZE)
                    cache[idx] = buffer[idx];
                __syncthreads();
            }
            assert(meta_idx == model_cached->meta_idx);
        }


        __device__ void writeModelToBuffer(int meta_idx) {
            assert(meta_idx >= 0 && meta_idx < Metadata::N_METADATA);
            float* cache  = reinterpret_cast<float*>(model_cached);
            float* buffer = reinterpret_cast<float*>(&models[meta_idx]);
            for (int i = 0; i < MODEL_CACHING_ITER; ++i) {
                int idx = i * blockDim.x + threadIdx.x;
                if (idx < MODEL_CACHING_SIZE)
                    buffer[idx] = cache[idx];
                __syncthreads();
            }
            assert(meta_idx == model_cached->meta_idx);
        }
        */


        __device__ void readModelFromBuffer(int meta_idx) {
            assert(meta_idx >= 0 && meta_idx < Metadata::N_METADATA);
            float* cache  = reinterpret_cast<float*>(model_cached);
            float* buffer = reinterpret_cast<float*>(&models[meta_idx]);
            for (int i = 0; i <= MODEL_CACHING_ITER; ++i) {
                int idx = i * blockDim.x + threadIdx.x;
                if (idx < MODEL_CACHING_SIZE)
                    cache[idx] = buffer[idx];
                __syncthreads();
            }
            assert(meta_idx == model_cached->meta_idx);
        }


        __device__ void writeModelToBuffer(int meta_idx) {
            assert(meta_idx >= 0 && meta_idx < Metadata::N_METADATA);
            float* cache  = reinterpret_cast<float*>(model_cached);
            float* buffer = reinterpret_cast<float*>(&models[meta_idx]);
            for (int i = 0; i <= MODEL_CACHING_ITER; ++i) {
                int idx = i * blockDim.x + threadIdx.x;
                if (idx < MODEL_CACHING_SIZE)
                    buffer[idx] = cache[idx];
                __syncthreads();
            }
            assert(meta_idx == model_cached->meta_idx);
        }


        __device__ void tryDumpAction(int line, const char* func, const char* file, int func_len, int file_len) {
            if (CURRENT_DUMP_IDX >= GLOBAL_DUMP_BUFFER_SIZE) return;
            if (blockIdx.x) return;  // write only leading model

            // write dump data of leading model
            ModelDumpBuffer& dump = model_dump[CURRENT_DUMP_IDX];
            // macros
            int idx;
            for (int i = 0; i <= blockDim.x / sizeof(float); ++i) {
                idx = i * blockDim.x + threadIdx.x;
                if (idx < file_len) {
                    dump.file[idx]     = file[idx];
                }
                if (idx < func_len) {
                    dump.function[idx] = func[idx];
                }
                __syncthreads();
            }
            // universal properties
            float* cache = reinterpret_cast<float*>(model_cached);
            float* buffer = reinterpret_cast<float*>(&dump.model);
            for (int i = 0; i < MODEL_CACHING_ITER; ++i) {
                idx = i * blockDim.x + threadIdx.x;
                if (idx < MODEL_CACHING_SIZE)
                    buffer[idx] = cache[idx];
                __syncthreads();
            }
            dump.line = line;
            dump.particle_size = Buffer::MAX_DIMENSION_PARTICIPANT;

            // write mean field 2D
            for (int it = 0; it <= model_cached->iter_2body; ++it) {
                // indexing matrix
                int mi = it * blockDim.x + threadIdx.x;
                int ti;
                if (mi < Buffer::model_cached->current_field_size_2) {
                    ti = mi + Buffer::model_cached->offset_2d;
                    dump.mean_field.rr2[mi]  = Buffer::MeanField::rr2[ti];
                    dump.mean_field.pp2[mi]  = Buffer::MeanField::pp2[ti];
                    dump.mean_field.rbij[mi] = Buffer::MeanField::rbij[ti];
                    dump.mean_field.rha[mi]  = Buffer::MeanField::rha[ti];
                    dump.mean_field.rhe[mi]  = Buffer::MeanField::rhe[ti];
                    dump.mean_field.rhc[mi]  = Buffer::MeanField::rhc[ti];
                }
                __syncthreads();
            }

            // write mean field 1D
            for (int it = 0; it <= Buffer::model_cached->current_field_size / blockDim.x; ++it) {
                // indexing matrix
                int mi = it * blockDim.x + threadIdx.x;
                int ti;
                if (mi < Buffer::model_cached->current_field_size) {
                    ti = mi + Buffer::model_cached->offset_1d;
                    dump.mean_field.ffrx[mi] = Buffer::MeanField::ffrx[ti];
                    dump.mean_field.ffry[mi] = Buffer::MeanField::ffry[ti];
                    dump.mean_field.ffrz[mi] = Buffer::MeanField::ffrz[ti];
                    dump.mean_field.ffpx[mi] = Buffer::MeanField::ffpx[ti];
                    dump.mean_field.ffpy[mi] = Buffer::MeanField::ffpy[ti];
                    dump.mean_field.ffpz[mi] = Buffer::MeanField::ffpz[ti];
                    dump.mean_field.f0rx[mi] = Buffer::MeanField::f0rx[ti];
                    dump.mean_field.f0ry[mi] = Buffer::MeanField::f0ry[ti];
                    dump.mean_field.f0rz[mi] = Buffer::MeanField::f0rz[ti];
                    dump.mean_field.f0px[mi] = Buffer::MeanField::f0px[ti];
                    dump.mean_field.f0py[mi] = Buffer::MeanField::f0py[ti];
                    dump.mean_field.f0pz[mi] = Buffer::MeanField::f0pz[ti];
                    dump.mean_field.rh3d[mi] = Buffer::MeanField::rh3d[ti];
                }
                __syncthreads();
            }

            // write participants
            for (int i = 0; i <= Buffer::MAX_DIMENSION_PARTICIPANT / blockDim.x; ++i) {
                idx = i * blockDim.x + threadIdx.x;
                if (idx < Buffer::MAX_DIMENSION_PARTICIPANT) {
                    int p = idx + model_cached->offset_1d;
                    dump.particles[idx].flags = Participant::flags[p];
                    dump.particles[idx].m     = Participant::mass[p];
                    dump.particles[idx].x     = Participant::position_x[p];
                    dump.particles[idx].y     = Participant::position_y[p];
                    dump.particles[idx].z     = Participant::position_z[p];
                    dump.particles[idx].px    = Participant::momentum_x[p];
                    dump.particles[idx].py    = Participant::momentum_y[p];
                    dump.particles[idx].pz    = Participant::momentum_z[p];
                }
                __syncthreads();
            }
            if (!threadIdx.x)
                CURRENT_DUMP_IDX++;
            __syncthreads();
        }


        __host__ void __host__deviceGetQMDPriority(size_t thread, mcutil::RingBuffer* buffer, int* target) {
            __device__deviceGetQMDPriority <<< 1, CUDA_WARP_SIZE, mcutil::MAX_DIMENSION_THREAD * 3 * sizeof(int) >>> (buffer, target);
        }


        __global__ void __device__deviceGetQMDPriority(mcutil::RingBuffer* buffer, int* target) {
            // queue occupation list
            float* q_occupy = reinterpret_cast<float*>(&mcutil::cache_univ[mcutil::SHARED_OFFSET_BUFFER_REDUX]);
            // id of most crowded queue
            int* q_target = reinterpret_cast<int*>(&mcutil::cache_univ[mcutil::SHARED_OFFSET_BUFFER_REDUX + mcutil::MAX_DIMENSION_THREAD]);

            q_target[threadIdx.x] = threadIdx.x;
            q_occupy[threadIdx.x] = buffer[threadIdx.x].getBufferSaturation();

            for (int stride = 1; stride < blockDim.x; stride *= 2) {
                if (threadIdx.x % (2 * stride) == 0) {
                    if (q_occupy[threadIdx.x] < q_occupy[threadIdx.x + stride]) {
                        q_occupy[threadIdx.x] = q_occupy[threadIdx.x + stride];
                        q_target[threadIdx.x] = q_target[threadIdx.x + stride];
                    }
                }
                __syncthreads();
            }
            assert(q_occupy[0] < 0.95f);
            *target = q_target[0];
        }


        __host__ cudaError_t setSymbolModel(QMDModel* ptr_qmd_model, int model_caching_iter, int field_dimension, int field_iter) {
            M_SOAPtrMapper(QMDModel*, ptr_qmd_model, models);
            M_SOAPtrMapper(int, model_caching_iter, MODEL_CACHING_ITER);
            M_SOAPtrMapper(int, field_dimension,    MAX_DIMENSION_PARTICIPANT);
            M_SOAPtrMapper(int, field_iter,         PARTICIPANT_MAX_ITER);
            return cudaSuccess;
        }


        __host__ cudaError_t setSymbolProblems(mcutil::RingBuffer* ptr_qmd_problems, int buffer_size) {
            M_SOAPtrMapper(mcutil::RingBuffer*, ptr_qmd_problems, qmd_problems);
            M_SOAPtrMapper(int, buffer_size, N_PROBLEM_BUFFERS);
            return cudaSuccess;
        }


        __host__ cudaError_t setSymbolDumpBuffer(ModelDumpBuffer* ptr_dump_buffer, int max_size) {
            M_SOAPtrMapper(ModelDumpBuffer*, ptr_dump_buffer, model_dump);
            int current_dump_idx = 0;
            M_SOAPtrMapper(int, max_size,         GLOBAL_DUMP_BUFFER_SIZE);
            M_SOAPtrMapper(int, current_dump_idx, CURRENT_DUMP_IDX);
            return cudaSuccess;
        }


        __host__ cudaError_t setSymbolProblemBuffer(mcutil::RingBuffer* ptr_buffer) {
            M_SOAPtrMapper(mcutil::RingBuffer*, ptr_buffer, qmd_problems);
            return cudaSuccess;
        }


        __host__ cudaError_t getSymbolDumpBuffer(int* n_buffer) {
            return cudaMemcpyFromSymbol(n_buffer, CURRENT_DUMP_IDX,
                sizeof(int));
        }


    }


}