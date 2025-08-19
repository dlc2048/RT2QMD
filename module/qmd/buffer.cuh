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
 * @file    module/qmd/buffer.cuh
 * @brief   Device side internal queue for QMD
 * @author  CM Lee
 * @date    02/14/2024
 */


#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "device/memory.cuh"
#include "transport/buffer.cuh"
#include "hadron/nucleus.cuh"


#ifdef NDEBUG
    #define QMD_DUMP_ACTION__
#else
    #define QMD_DUMP_ACTION__(function, file, lfunction, lfile) RT2QMD::Buffer::tryDumpAction(__LINE__, function, file, lfunction, lfile)
#endif


namespace RT2QMD {


    constexpr unsigned int CUDA_NUMERIC_NAN = 0xFFE00000;


    constexpr int MAX_DIMENSION_CLUSTER   = 256;  //! @brief maximum number of participants
    constexpr int MAX_DIMENSION_CLUSTER_B 
        = MAX_DIMENSION_CLUSTER / 32;             //! @brief for boolean initialization


    typedef struct ModelMetadataQueue {
        unsigned long long int head;
        unsigned long long int tail;
        int  size;
        int* idx;


        __inline__ __device__ int pull();


        __inline__ __device__ void push(int idx);


        __inline__ __device__ int nElements();


    } ModelMetadataQueue;


    typedef struct QMDModel { 
        int    __pad[2];                  //! @brief memory pad for queue push/pull actions & reduction
        int    condition_broadcast;       //! @brief condition broadcast for consistent loop break
        int    meta_idx;                  //! @brief metadata idx (used only in shared memory)
        unsigned char participant_idx[MAX_DIMENSION_CLUSTER];
        mcutil::UNION_FLAGS initial_flags;   //! @brief initial flags
        int    offset_1d;                 //! @brief n-vector type memory & participant offset
        int    offset_2d;                 //! @brief n by n matrix type offset
        int    initial_hid;               //! @brief initial history id
        float  initial_eke;               //! @brief initial kinetic energy of projectile [GeV]
        float  initial_momentum;          //! @brief initial momentum of projectile [GeV/c]
        float  initial_weight;            //! @brief initial weight of projectile
        float  initial_position[3];       //! @brief initial position of projectile [cm]
        float  initial_polar[2];          //! @brief initial direction polar (sin, cos) of projectile
        float  initial_azim[2];           //! @brief initial direction azimuthal (sin, cos) of projectile
        float  maximum_impact_parameter;  //! @brief maximum impact parameter [fm]
        short2 za_nuc[2];                 //! @brief ZA number of projectile / target
        uchar4 nuc_counter;               //! @brief Nucleus initializer counter
        float  mass[2];                   //! @brief mass of projectile / target [GeV/c^2]
        float  beta_lab_nn;               //! @brief boost beta LAB to NN system
        float  beta_nn_cm;                //! @brief boost beta NN to CM system
        float  cc_gamma[2];               //! @brief gamma of projectile / target
        float  cc_rx[2];                  //! @brief x-axis position offset of projectile / target [fm]
        float  cc_rz[2];                  //! @brief z-axis position offset of projectile / target [fm]
        float  cc_px[2];                  //! @brief x-axis momentum offset of projectile / target [GeV/c]
        float  cc_pz[2];                  //! @brief z-axis momentum offset of projectile / target [GeV/c]
        float  cphi;                      //! @brief azimuthal cosine, uniform RN
        float  sphi;                      //! @brief azimuthal sine, uniform RN
        short2 za_system;                 //! @brief ZA number of current system
        float  momentum_system[3];        //! @brief net momentum of current system [GeV/c]
        float  mass_system;               //! @brief net mass of current system [GeV/c^2]
        float  ekinal;                    //! @brief total internal kinetic energy of current system [GeV]
        float  vtot;                      //! @brief total internal potential energy of current system [GeV]
        float  excitation;                //! @brief excitation energy of current system [GeV]
        // field information related to energy level
        float  ebinal;
        float  ebini;
        float  ebin0;
        float  ebin1;
        float  dtc;
        float  edif0;
        float  cfrc;
        float  edif;
        int    ifrc;
        int    jfrc;
        // field information related to collision
        int    n_collisions;              //! @brief number of collisions
        int    n_cluster;                 //! @brief number of remnant cluster
        // field information related to dimension
        int    offset;                    //! @brief participant address offset
        int    current_field_size;        //! @brief current field size
        int    current_field_size_2;      //! @brief current field size ^ 2
        int    iter_2body;                //! @brief number of two-body mean field iteration 
        int    iter_1body;                //! @brief number of one-body mean field iteration 
        int    iter_gphase;               //! @brief number of graduate phase-space loading iteration
        int    iter_grh3d;                //! @brief number of graduate rh3d calculation iteration
        int    iter_gwarp;                //! @brief number of graduate warp iteration
        int    iter_gblock;               //! @brief number of graduate block iteration
        int    iter_glane;                //! @brief number of graduate lane iteration
    } QMDModel;

    extern __constant__ bool BUFFER_HAS_HID;
    extern __device__ mcutil::RingBuffer* buffer_catalog;  //! @brief Global buffer

    extern __device__ curandState* rand_state;      //! @brief External rand state buffer

    // mass table
    extern __device__ Nucleus::MassTable* mass_table;

    // HN corrections
    extern __device__ float* barashenkov_corr[2];


    __host__ cudaError_t setBufferHandle(CUdeviceptr handle, bool has_hid);


    __host__ cudaError_t setPrngHandle(CUdeviceptr handle);


    __host__ cudaError_t setMassTableHandle(CUdeviceptr handle);


    __host__ cudaError_t setBarashenkovPtr(float* corr_neutron, float* corr_proton);


    namespace Buffer {


        extern __constant__ int MAX_DIMENSION_PARTICIPANT;
        extern __constant__ int PARTICIPANT_MAX_ITER;


        //! @brief Meta data
        namespace Metadata {

            extern __constant__ int N_METADATA;
            extern __device__ ModelMetadataQueue* meta_queue;   // {idle, launch}
            extern __device__ int* current_metadata_idx;   // current IDX per block


            __host__ cudaError_t setSymbolMetadata(int n_metadata, ModelMetadataQueue* ptr_queue, int* ptr_idx_array);


        }


        //! @brief Nucleon SOA
        namespace Participant{


            constexpr int SOA_MEMBERS = 8;


            extern __device__ int*   flags;
            extern __device__ float* mass;             //! @brief participant mass [GeV/c^2]  (maybe needed to handle Delta baryons)
            extern __device__ float* position_x;       //! @brief x-position of participant [fm]
            extern __device__ float* position_y;       //! @brief y-position of participant [fm]
            extern __device__ float* position_z;       //! @brief z-position of participant [fm]
            extern __device__ float* momentum_x;       //! @brief x-momentum of participant [GeV/c]
            extern __device__ float* momentum_y;       //! @brief y-momentum of participant [GeV/c]
            extern __device__ float* momentum_z;       //! @brief z-momentum of participant [GeV/c]

            extern __device__ float* ps[SOA_MEMBERS];  //! @brief vectorized phase-space


            __host__ cudaError_t setSymbolBuffer(void* soa_list[]);


        }


        //! @brief Mean-field SOA
        namespace MeanField {


            constexpr int SOA_MEMBERS_2D = 6;
            constexpr int SOA_MEMBERS_1D = 13;
            

            // 2D field

            extern __device__ float* rr2;   //! @brief G4QMDMeanField::rr2
            extern __device__ float* pp2;   //! @brief G4QMDMeanField::pp2
            extern __device__ float* rbij;  //! @brief G4QMDMeanField::rbij
            extern __device__ float* rha;   //! @brief G4QMDMeanField::rha
            extern __device__ float* rhe;   //! @brief G4QMDMeanField::rhe
            extern __device__ float* rhc;   //! @brief G4QMDMeanField::rhc

            // 1D field

            extern __device__ float* ffrx;  //! @brief G4QMDMeanField::ffr.x()
            extern __device__ float* ffry;  //! @brief G4QMDMeanField::ffr.y()
            extern __device__ float* ffrz;  //! @brief G4QMDMeanField::ffr.z()
            extern __device__ float* ffpx;  //! @brief G4QMDMeanField::ffp.x()
            extern __device__ float* ffpy;  //! @brief G4QMDMeanField::ffp.y()
            extern __device__ float* ffpz;  //! @brief G4QMDMeanField::ffp.z()
            extern __device__ float* f0rx;  //! @brief G4QMDMeanField::DoPropagation()::f0r.x()
            extern __device__ float* f0ry;  //! @brief G4QMDMeanField::DoPropagation()::f0r.y()
            extern __device__ float* f0rz;  //! @brief G4QMDMeanField::DoPropagation()::f0r.z()
            extern __device__ float* f0px;  //! @brief G4QMDMeanField::DoPropagation()::f0p.x()
            extern __device__ float* f0py;  //! @brief G4QMDMeanField::DoPropagation()::f0p.y()
            extern __device__ float* f0pz;  //! @brief G4QMDMeanField::DoPropagation()::f0p.z()
            extern __device__ float* rh3d;  //! @brief G4QMDMeanField::rh3d, G4QMDGroundStateNucleus::phase_g


            __host__ cudaError_t setSymbolBuffer2D(void* soa_list[]);


            __host__ cudaError_t setSymbolBuffer1D(void* soa_list[]);


        }

        /*
        //! @brief External ring buffer for ion-nuclear inelastic scattering
        typedef struct IonInteractionBuffer {
            int   size;   //!< @brief Buffer size (int), dummy in QMD
            float sizef;  //!< @brief Buffer size (float), dummy in QMD
            unsigned long long int head;
            unsigned long long int tail;

            // SOA data
            float* x;     //!< @brief Particle x position (cm)
            float* y;     //!< @brief Particle y position (cm)
            float* z;     //!< @brief Particle z position (cm)
            float* u;     //!< @brief Particle x direction
            float* v;     //!< @brief Particle y direction
            float* w;     //!< @brief Particle z direction
            float* e;     //!< @brief Particle kinetic energy (MeV)
            float* wee;   //!< @brief Particle weight
            int*   zapt;  //!< @brief uchar4 -> za projectile, za target


            __inline__ __device__ int getBufferOccupancy();
            __inline__ __device__ int pushBulk();
            __inline__ __device__ int pushAtomic();
            __inline__ __device__ int pullAtomic();


            __host__ void malloc(size_t capacity);


            __host__ void free();


        } IonInteractionBuffer;
        */

        /*
        //! @brief Ring buffer for excited cluster, QMD <-> GEM communication
        typedef struct EvaporationBuffer {
            int   size;   //!< @brief Buffer size (int)
            float sizef;  //!< @brief Buffer size (float)
            unsigned long long int head;
            unsigned long long int tail;

            // SOA data
            float* x;      //!< @brief Nucleus x position [cm]
            float* y;      //!< @brief Nucleus y position [cm]
            float* z;      //!< @brief Nucleus z position [cm]
            float* u;      //!< @brief Nucleus x direction
            float* v;      //!< @brief Nucleus y direction
            float* w;      //!< @brief Nucleus z direction
            float* e;      //!< @brief Nucleus kinetic energy [GeV]
            float* wee;    //!< @brief Particle weight
            float* excit;  //!< @brief Nucleus excitation energy [GeV]
            int*   za;     //!< @brief short2 -> za of nucleus


            __inline__ __device__ int getBufferOccupancy();
            __inline__ __device__ int pushAtomic();


            __host__ void malloc(size_t capacity);


            __host__ void free();


        } EvaporationBuffer;
        */
        

        //! @brief AOS particle dump buffer
        typedef struct ParticleDumpBuffer {
            int   flags;
            float m;
            float x;
            float y;
            float z;
            float px;
            float py;
            float pz;
        } ParticleDumpBuffer;


        typedef struct MeanFieldDumpBuffer {
            // 2D field
            float* rr2;
            float* pp2;
            float* rbij;
            float* rha;
            float* rhe;
            float* rhc;
            // 1D field
            float* ffrx;
            float* ffry;
            float* ffrz;
            float* ffpx;
            float* ffpy;
            float* ffpz;
            float* f0rx;
            float* f0ry;
            float* f0rz;
            float* f0px;
            float* f0py;
            float* f0pz;
            float* rh3d;
        } MeanFieldDumpBuffer;


        //! @brief AOS model dump buffer
        typedef struct ModelDumpBuffer {
            char                 file[256];
            char                 function[256];
            int                  line;
            int                  particle_size;
            QMDModel             model;
            MeanFieldDumpBuffer  mean_field;
            ParticleDumpBuffer*  particles;
            

            __host__ size_t malloc(size_t size_particle);


            __host__ void free();


        } ModelDumpBuffer;


        //! @brief SOA particle buffer for shared memory
        typedef struct ParticleSOA {
            int   flags;
            float x;
            float y;
            float z;
            float px;
            float py;
            float pz;
            float e;
        } ParticleSOA;

        constexpr int PARTICLE_CACHING_STRIDE = sizeof(ParticleSOA) / 4;

        
        extern __shared__ QMDModel            model_cached[];  //! @brief Model shared memory (offset = 0)
        extern __device__ QMDModel*           models;          //! @brief Global memory, model lists
        extern __device__ mcutil::RingBuffer* qmd_problems;    //! @brief QMD problem lists (dimension32, 64 ...)
        extern __device__ ModelDumpBuffer*    model_dump;      //! @brief QMD dump lists

        extern __constant__ int N_PROBLEM_BUFFERS;

        extern __constant__ int GLOBAL_INC_BUFFER_SIZE;
        extern __constant__ int GLOBAL_DUMP_BUFFER_SIZE;
        extern __device__   int GLOBAL_MODEL_COUNTER;
        extern __device__   int CURRENT_DUMP_IDX;
        extern __constant__ int MODEL_CACHING_ITER;

        // shared memories I/O
        constexpr int MODEL_CACHING_SIZE   = sizeof(QMDModel) / 4;
        constexpr int MODEL_CACHING_OFFSET = (MODEL_CACHING_SIZE / 2 + 1) * 2;


        __device__ void readModelFromBuffer(int meta_idx = 0);


        __device__ void writeModelToBuffer(int meta_idx = 0);


        __device__ void tryDumpAction(int line, const char* func, const char* file, int func_len, int file_len);


        __host__ void __host__deviceGetQMDPriority(size_t thread, mcutil::RingBuffer* buffer, int* target);
        __global__ void __device__deviceGetQMDPriority(mcutil::RingBuffer* buffer, int* target);


        __host__ cudaError_t setSymbolModel(QMDModel* ptr_qmd_model, int model_caching_iter, int field_dimension, int field_iter);


        __host__ cudaError_t setSymbolProblems(mcutil::RingBuffer* ptr_qmd_problems, int buffer_size);


        __host__ cudaError_t setSymbolDumpBuffer(ModelDumpBuffer* ptr_dump_buffer, int max_size);


        __host__ cudaError_t getSymbolDumpBuffer(int* n_buffer);


    }


#ifdef __CUDACC__


    __device__ int ModelMetadataQueue::nElements() {
        int* usage_univ = reinterpret_cast<int*>(mcutil::cache_univ);
        if (!threadIdx.x) {
            *usage_univ = this->head - this->tail;
        }
        __syncthreads();
        int usage_local = usage_univ[0];
        __syncthreads();
        return usage_local;
    }


    __device__ int ModelMetadataQueue::pull() {
        unsigned long long int* tail_univ
            = reinterpret_cast<unsigned long long int*>(mcutil::cache_univ);
        if (!threadIdx.x) {
            unsigned long long int old_val, assumed;
            do {
                old_val = this->tail;
                if (old_val >= this->head) {
                    old_val = UINT64_MAX;
                    break;
                }
                assumed = old_val;
                old_val = atomicCAS(&this->tail, assumed, assumed + 1);
            } while (old_val != assumed);
            tail_univ[0] = old_val;
        }
        __syncthreads();
        unsigned long long int tail_local = tail_univ[0];
        __syncthreads();
        if (tail_local == UINT64_MAX)
            return -1;
        else {
            assert(tail_local <= this->head);
            return this->idx[tail_local % this->size];
        }
    }


    __device__ void ModelMetadataQueue::push(int idx) {
        unsigned long long int* head_univ
            = reinterpret_cast<unsigned long long int*>(mcutil::cache_univ);
        if (!threadIdx.x) {
            head_univ[0] = atomicAdd(&this->head, 1);
            this->idx[head_univ[0] % this->size] = idx;
        }
        __syncthreads();
        
    }


#endif



}

