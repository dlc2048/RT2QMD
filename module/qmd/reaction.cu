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
 * @file    module/qmd/reaction.cu
 * @brief   QMD reaction
 * @author  CM Lee
 * @date    02/15/2024
 */


#include "reaction.cuh"
#include "buffer.cuh"
#include "mean_field.cuh"
#include "collision.cuh"

#include "device/assert.cuh"
#include "hadron/xs_dev.cuh"
#include "hadron/nucleus.cuh"

#include <stdio.h>


namespace RT2QMD {


    namespace REACTION {
        constexpr size_t const_strlen(const char* str) {
            size_t len = 0;
            while (str[len] != '\0') ++len;
            return len;
        }
        constexpr size_t FILE_LEN     = const_strlen(__FILE__);
    }


    // timer
    __constant__ bool           USE_CLOCK = false;
    __device__   long long int* timer;


    __global__ void __kernel__fieldDispatcher(int field_target) {
        int meta_idx;

        Buffer::model_cached->meta_idx = -1;  // initial state
#ifdef RT2QMD_ACTIVATE_TIMER
        if (USE_CLOCK) {
            if (!threadIdx.x)
                timer[blockIdx.x] = clock64();
        }
#endif
        __syncthreads();

        // persistent loop
        while (true) {

            __syncthreads();
            meta_idx = Buffer::model_cached->meta_idx;

            // exit condition
            if (Buffer::Metadata::meta_queue[1].nElements() >= gridDim.x) {
                if (meta_idx >= 0) {
                    Buffer::writeModelToBuffer(meta_idx);  // write current status to global buffer
                    __syncthreads();
                    Buffer::Metadata::meta_queue[0].push(meta_idx);
                }
                __syncthreads();
                break;
            }

            __syncthreads();
            if (meta_idx < 0) {  // new batch
                meta_idx = Buffer::Metadata::meta_queue[0].pull();
                __syncthreads();
                if (meta_idx >= 0)
                    Buffer::readModelFromBuffer(meta_idx);
                else
                    continue;
            }
            __syncthreads();

            unsigned short phase = Buffer::model_cached->initial_flags.qmd.phase;
            __syncthreads();
            switch (phase) {
            case MODEL_STAGE::MODEL_IDLE:
                prepareModel(field_target);  // pull from the problem buffer
                __syncthreads();
                prepareImpact();
                break;
            case MODEL_STAGE::MODEL_PREPARE_PROJECTILE:
                if (Buffer::model_cached->za_nuc[0].y != 1)
                    MeanField::prepareNuclei(false);
                Buffer::model_cached->initial_flags.qmd.phase
                    = MODEL_STAGE::MODEL_SAMPLE_PROJECTILE;
                break;
            case MODEL_STAGE::MODEL_SAMPLE_PROJECTILE:
                if (Buffer::model_cached->za_nuc[0].y != 1) {
                    if (MeanField::sampleNuclei(false))
                        Buffer::model_cached->initial_flags.qmd.phase
                        = MODEL_STAGE::MODEL_PREPARE_TARGET;
                    else
                        Buffer::model_cached->initial_flags.qmd.phase
                        = MODEL_STAGE::MODEL_PREPARE_PROJECTILE;
                }
                else {
                    MeanField::sampleNucleon(false);
                    Buffer::model_cached->initial_flags.qmd.phase
                        = MODEL_STAGE::MODEL_PREPARE_TARGET;
                }
                break;
            case MODEL_STAGE::MODEL_PREPARE_TARGET:
                if (Buffer::model_cached->za_nuc[1].y != 1)
                    MeanField::prepareNuclei(true);
                Buffer::model_cached->initial_flags.qmd.phase
                    = MODEL_STAGE::MODEL_SAMPLE_TARGET;
                break;
            case MODEL_STAGE::MODEL_SAMPLE_TARGET:
                if (Buffer::model_cached->za_nuc[1].y != 1) {
                    if (MeanField::sampleNuclei(true))
                        Buffer::model_cached->initial_flags.qmd.phase
                        = MODEL_STAGE::MODEL_PROPAGATE;
                    else
                        Buffer::model_cached->initial_flags.qmd.phase
                        = MODEL_STAGE::MODEL_PREPARE_TARGET;
                }
                else {
                    MeanField::sampleNucleon(true);
                    Buffer::model_cached->initial_flags.qmd.phase
                        = MODEL_STAGE::MODEL_PROPAGATE;
                }
                break;
            case MODEL_STAGE::MODEL_PROPAGATE:
                meta_idx = Buffer::model_cached->meta_idx;
                __syncthreads();
                Buffer::Metadata::meta_queue[1].push(meta_idx);   // push to metadata queue (ready)
                __syncthreads();
                Buffer::writeModelToBuffer(meta_idx);  // write to global memory buffer
                __syncthreads();
                Buffer::model_cached->meta_idx = -1;  // clear cache
                break;
            default:
                assert(false);
            }
            __syncthreads();
        }
#ifdef RT2QMD_ACTIVATE_TIMER
        if (USE_CLOCK) {
            if (!threadIdx.x)
                timer[blockIdx.x] = clock64() - timer[blockIdx.x];
        }
#endif
        __syncthreads();
        return;
    }


    __host__ cudaError_t __host__fieldDispatcher(int block, int thread, int field_target) {
        __kernel__fieldDispatcher <<< block, thread, mcutil::SIZE_SHARED_MEMORY_QMD >>> (field_target);
        return cudaDeviceSynchronize();
    }


    __global__ void __kernel__pullEligibleField() {
#ifdef RT2QMD_ACTIVATE_TIMER
        if (USE_CLOCK) {
            if (!threadIdx.x)
                timer[blockIdx.x] = clock64();
        }
        __syncthreads();
#endif
        Buffer::Metadata::current_metadata_idx[blockIdx.x] = Buffer::Metadata::meta_queue[1].pull();
#ifdef RT2QMD_ACTIVATE_TIMER
        if (USE_CLOCK) {
            if (!threadIdx.x)
                timer[blockIdx.x] = clock64() - timer[blockIdx.x];
        }
        __syncthreads();
#endif
    }


    __host__ cudaError_t __host__pullEligibleField(int block, int thread) {
        __kernel__pullEligibleField <<< block, thread, mcutil::SIZE_SHARED_MEMORY_QMD >>> ();
        return cudaDeviceSynchronize();
    }


    __device__ void prepareModel(int field_target) {
        int  idx_ion;
        int* idx_ion_leading = (int*)&mcutil::cache_univ[Buffer::MODEL_CACHING_OFFSET];
        __syncthreads();
        if (!threadIdx.x)
            *idx_ion_leading = Buffer::qmd_problems[field_target].pullAtomic();
        __syncthreads();
        idx_ion = *idx_ion_leading;

        // initial phase-space
        Buffer::model_cached->initial_weight      = Buffer::qmd_problems[field_target].wee[idx_ion];
        Buffer::model_cached->initial_position[0] = Buffer::qmd_problems[field_target].x[idx_ion];
        Buffer::model_cached->initial_position[1] = Buffer::qmd_problems[field_target].y[idx_ion];
        Buffer::model_cached->initial_position[2] = Buffer::qmd_problems[field_target].z[idx_ion];

        assertNAN(Buffer::model_cached->initial_position[0]);
        assertNAN(Buffer::model_cached->initial_position[1]);
        assertNAN(Buffer::model_cached->initial_position[2]);

        // calculate coord transform angles
        float3 direction;
        direction.x = Buffer::qmd_problems[field_target].u[idx_ion];
        direction.y = Buffer::qmd_problems[field_target].v[idx_ion];
        direction.z = Buffer::qmd_problems[field_target].w[idx_ion];

        assertNAN(direction.x);
        assertNAN(direction.y);
        assertNAN(direction.z);

        float cost, sint;
        float cosp, sinp;

        cost = direction.z;
        sint = sqrtf(fmaxf(0.f, 1.f - cost * cost));
        sint = direction.z < 0.f ? -sint : sint;
        if (fabsf(sint) > 1.e-10f) {
            cosp = direction.x / sint;
            sinp = direction.y / sint;
        }
        else {
            sinp = 0.f;
            cosp = 1.f;
        }

        Buffer::model_cached->initial_polar[0] = sint;
        Buffer::model_cached->initial_polar[1] = cost;
        Buffer::model_cached->initial_azim[0]  = sinp;
        Buffer::model_cached->initial_azim[1]  = cosp;

        assertNAN(sint);
        assertNAN(cost);
        assertNAN(sinp);
        assertNAN(cosp);
 
        // ZA
        uchar4 zapt = Buffer::qmd_problems[field_target].za[idx_ion];        
        Buffer::model_cached->za_nuc[0] = { zapt.x, zapt.y };
        Buffer::model_cached->za_nuc[1] = { zapt.z, zapt.w };

        assert(zapt.y > 0);
        assert(zapt.w > 0);

        // mass
        float m1 = mass_table[zapt.x].get(zapt.y) * 1e-3f;  // [GeV/c^2]
        float m2 = mass_table[zapt.z].get(zapt.w) * 1e-3f;  // [GeV/c^2]
        Buffer::model_cached->mass[0] = m1;  
        Buffer::model_cached->mass[1] = m2;  

        // flags (initialize except the region id)
        mcutil::UNION_FLAGS flags(Buffer::qmd_problems[field_target].flags[idx_ion]);
        flags.qmd.fmask  = 0u;
        flags.qmd.phase  = MODEL_STAGE::MODEL_PREPARE_PROJECTILE;   // next -> prepare projectile
        Buffer::model_cached->initial_flags = flags;

        // hid
        if (BUFFER_HAS_HID)
            Buffer::model_cached->initial_hid = Buffer::qmd_problems[field_target].hid[idx_ion];

        // kinematics

        // energy & momentum
        float eke = Buffer::qmd_problems[field_target].e[idx_ion] * (float)zapt.y;  // per nucleon -> total
        float p1  = sqrtf(eke * (eke + 2.f * m1));  // [GeV/c]

        Buffer::model_cached->initial_eke      = eke;
        Buffer::model_cached->initial_momentum = p1;
        p1 /= (float)zapt.y;  // per nucleon

        assertNAN(eke);
        assertNAN(p1);

        // coordinate system
        float e1, e2, ln, lc;

        // LAB to NN boost
        e1 = m1 / (float)zapt.y;
        e1 = sqrtf(p1 * p1 + e1 * e1);
        e2 = m2 / (float)zapt.w;
        ln = -p1 / (e1 + e2);

        __syncthreads();

        // NN to CM boost
        p1 = Buffer::model_cached->initial_momentum;
        e1 = sqrtf(p1 * p1 + m1 * m1);
        lc = -p1 / (e1 + m2);

        Buffer::model_cached->beta_lab_nn = ln;
        Buffer::model_cached->beta_nn_cm  = (fabsf(ln) - fabsf(lc)) / (1.f - fabsf(lc * ln));

        assertNAN(Buffer::model_cached->beta_lab_nn);
        assertNAN(Buffer::model_cached->beta_nn_cm);

        // impact parameter
        eke = Buffer::model_cached->initial_eke;
        float rp    = Nucleus::nuclearRadius({ zapt.x, zapt.y });
        float rt    = Nucleus::nuclearRadius({ zapt.z, zapt.w });
        float cb    = Hadron::coulombBarrier((int)zapt.x * (int)zapt.z, m1, m2, eke, rp + rt);

        eke /= (float)zapt.y;  // per nucleon
        float sigpp = Hadron::xsNucleonNucleon(true, true,  eke);
        float sigpn = Hadron::xsNucleonNucleon(true, false, eke);
        float ns    = ::constants::FP32_TWE_PI * (rp * rp + rt * rt);
        float xs0   = Hadron::xsNucleiNuclei({ zapt.x, zapt.y }, { zapt.z, zapt.w }, cb, sigpp, sigpn, ns);

        Buffer::model_cached->maximum_impact_parameter = sqrtf(xs0 * ::constants::FP32_TEN_PI_I) * constants::ENVELOP_F;
        assertNAN(Buffer::model_cached->maximum_impact_parameter);
    }


    __device__ void returnModel(int field_target) {
        int  idx_ion;
        int* idx_ion_leading = (int*)&mcutil::cache_univ[Buffer::MODEL_CACHING_OFFSET];
        __syncthreads();
        if (!threadIdx.x)
            *idx_ion_leading = Buffer::qmd_problems[field_target].pushAtomic();
        __syncthreads();
        idx_ion = *idx_ion_leading;

        Buffer::qmd_problems[field_target].wee[idx_ion] = Buffer::model_cached->initial_weight;
        Buffer::qmd_problems[field_target].x[idx_ion]   = Buffer::model_cached->initial_position[0];
        Buffer::qmd_problems[field_target].y[idx_ion]   = Buffer::model_cached->initial_position[1];
        Buffer::qmd_problems[field_target].z[idx_ion]   = Buffer::model_cached->initial_position[2];

        float cost, sint;
        float cosp, sinp;

        sint = Buffer::model_cached->initial_polar[0];
        cost = Buffer::model_cached->initial_polar[1];
        sinp = Buffer::model_cached->initial_azim[0];
        cosp = Buffer::model_cached->initial_azim[1];

        // calculate coord transform angles
        float3 direction;

        direction.z = cost;
        if (fabsf(sint) > 1e-10f) {
            direction.x = sint * sinp;
            direction.y = sint * cosp;
        }
        else {
            direction.x = 0.f;
            direction.y = 0.f;
        }
        Buffer::qmd_problems[field_target].u[idx_ion] = direction.x;
        Buffer::qmd_problems[field_target].v[idx_ion] = direction.y;
        Buffer::qmd_problems[field_target].w[idx_ion] = direction.z;

        // ZA
        short2 zap = Buffer::model_cached->za_nuc[0];
        short2 zat = Buffer::model_cached->za_nuc[1];

        assert(zap.y > 0);
        assert(zat.y > 0);

        Buffer::qmd_problems[field_target].za[idx_ion] = { 
            (unsigned char)zap.x, 
            (unsigned char)zap.y, 
            (unsigned char)zat.x, 
            (unsigned char)zat.y 
        };

        Buffer::model_cached->initial_flags.qmd.phase = 0u;
        Buffer::model_cached->initial_flags.qmd.fmask = 0u;
        __syncthreads();

        // flags (init)
        Buffer::qmd_problems[field_target].flags[idx_ion]
            = Buffer::model_cached->initial_flags.astype<unsigned int>();

        // hid
        if (BUFFER_HAS_HID)
            Buffer::qmd_problems[field_target].hid[idx_ion] = Buffer::model_cached->initial_hid;

        // energy
        float eke = Buffer::model_cached->initial_eke;
        Buffer::qmd_problems[field_target].e[idx_ion] = eke / (float)zap.y;  // total -> per nucleon
    }


    __device__ void prepareImpact() {

        int   idx  = blockIdx.x * blockDim.x + threadIdx.x;
        int   side = idx % 2;
        mcutil::cache_univ[Buffer::MODEL_CACHING_OFFSET] // syncronize random impact parameter
            = sqrtf(1.f - curand_uniform(&rand_state[idx])) * Buffer::model_cached->maximum_impact_parameter;
        __syncthreads();
        float b    = mcutil::cache_univ[Buffer::MODEL_CACHING_OFFSET];
        float mt   = Buffer::model_cached->mass[0] + Buffer::model_cached->mass[1];
        float etot = Buffer::model_cached->initial_eke + mt;
        float stot = sqrtf(etot * etot - Buffer::model_cached->initial_momentum * Buffer::model_cached->initial_momentum);
        float eccm = stot - mt;
        float rmax = Buffer::model_cached->maximum_impact_parameter + 4.f;
              rmax = sqrtf(rmax * rmax + b * b);
        float z2   = constants::CCOUL * (float)(Buffer::model_cached->za_nuc[0].x * (int)Buffer::model_cached->za_nuc[1].x);
        float pccf = b / rmax;
              pccf = 1.f - z2  / eccm / rmax - pccf * pccf;
              pccf = sqrtf(pccf);
        
        float aas = 0.f;
        float bbs = 0.f;
        if (Buffer::model_cached->za_nuc[0].x && Buffer::model_cached->za_nuc[1].x) {
            aas = 2.f * eccm * b / z2;
            bbs = 1.f / sqrtf(1.f + aas * aas);
            aas = (1.f + aas * b / rmax) * bbs;
        }

        float sint, cost;
        if (1.f <= aas || 1.f <= bbs) {
            cost = 1.f;
            sint = 0.f;
        }
        else {
            aas = atanf(aas / sqrtf(1.f - aas * aas));
            bbs = atanf(bbs / sqrtf(1.f - bbs * bbs));
            sincosf(aas - bbs, &sint, &cost);
        }

        // position
        float zf = rmax * cost / mt;
        Buffer::model_cached->cc_rx[side] = rmax * (1.f - (float)(2 * side)) / 2.f * sint;
        Buffer::model_cached->cc_rz[side] = ((float)(2 * side) - 1.f) * zf * Buffer::model_cached->mass[side];

        // momentum
        float md = Buffer::model_cached->mass[0] - Buffer::model_cached->mass[1];
        float px, pz, pzc, ec;

        // momentum projectile
        pzc = sqrtf((stot * stot - mt * mt) * (stot * stot - md * md)) / (2.f * stot);
        px  = pzc * (-sint * pccf + cost * b / rmax);
        pzc = pzc * (+cost * pccf + sint * b / rmax);

        if (side) {
            pzc = -pzc;
            px  = -px;
        }
        __syncthreads();

        // CM to NN
        float bcm = -Buffer::model_cached->beta_nn_cm;
        float gcm = 1.f / sqrtf(1.f - bcm * bcm);
            
        ec  = Buffer::model_cached->mass[side];
        ec  = sqrtf(pzc * pzc + px * px + ec * ec);
        pz  = pzc + bcm * gcm * (gcm / (1.f + gcm) * pzc * bcm + ec);
        ec  = gcm * (ec + bcm * pzc);
        Buffer::model_cached->cc_gamma[side] = ec / Buffer::model_cached->mass[side];
        Buffer::model_cached->cc_px[side]    = px / (float)Buffer::model_cached->za_nuc[side].y;
        Buffer::model_cached->cc_pz[side]    = pz / (float)Buffer::model_cached->za_nuc[side].y;

        // initialize participants
        for (int i = 0; i < Buffer::PARTICIPANT_MAX_ITER; ++i) {
            int ti = threadIdx.x + i * blockDim.x;
            if (ti < Buffer::MAX_DIMENSION_PARTICIPANT)
                Buffer::Participant::flags[Buffer::model_cached->offset_1d + ti] = 0;
            __syncthreads();
        }

        // phi
        float phi = (1.f - curand_uniform(&rand_state[idx])) * ::constants::FP32_TWO_PI;
        float sphi, cphi;
        sincosf(phi, &sphi, &cphi);
        if (!threadIdx.x) {
            Buffer::model_cached->sphi = sphi;
            Buffer::model_cached->cphi = cphi;
        }
        __syncthreads();

        // initialize model
        Buffer::model_cached->n_collisions = 0;
        Buffer::model_cached->n_cluster    = 0;
    }


    __global__ void __kernel__prepareModel(int field_target) {
        Buffer::model_cached->meta_idx = blockIdx.x;
#ifdef RT2QMD_ACTIVATE_TIMER
        if (USE_CLOCK) {
            if (!threadIdx.x)
                timer[blockIdx.x] = clock64();
        }
#endif
        __syncthreads();
        Buffer::readModelFromBuffer(blockIdx.x);
        __syncthreads();
        prepareModel(field_target);  // pull from the problem buffer
        __syncthreads();
        prepareImpact();
        Buffer::writeModelToBuffer(blockIdx.x);
#ifdef RT2QMD_ACTIVATE_TIMER
        if (USE_CLOCK) {
            if (!threadIdx.x)
                timer[blockIdx.x] = clock64() - timer[blockIdx.x];
        }
        __syncthreads();
#endif
        return;
    }
    

    __host__ cudaError_t __host__prepareModel(int block, int thread, int field_target) {
        __kernel__prepareModel <<< block, thread, mcutil::SIZE_SHARED_MEMORY_QMD >>> (field_target);
        return cudaDeviceSynchronize();
    }


    __global__ void __kernel__prepareProjectile() {
#ifdef RT2QMD_ACTIVATE_TIMER
        if (USE_CLOCK) {
            if (!threadIdx.x)
                timer[blockIdx.x] = clock64();
        }
#endif
        Buffer::readModelFromBuffer(blockIdx.x);
        __syncthreads();
        if (Buffer::model_cached->za_nuc[0].y != 1) {
            while (true) {
                MeanField::prepareNuclei(false);
                if (MeanField::sampleNuclei(false))
                    break;
            }
        }
        else {
            MeanField::sampleNucleon(false);
        }
        __syncthreads();
        Buffer::writeModelToBuffer(blockIdx.x);
#ifdef RT2QMD_ACTIVATE_TIMER
        if (USE_CLOCK) {
            if (!threadIdx.x)
                timer[blockIdx.x] = clock64() - timer[blockIdx.x];
        }
        __syncthreads();
#endif
        return;
    }


    __host__ cudaError_t __host__prepareProjectile(int block, int thread) {
        __kernel__prepareProjectile <<< block, thread, mcutil::SIZE_SHARED_MEMORY_QMD >>> ();
        return cudaDeviceSynchronize();
    }


    __global__ void __kernel__prepareTarget() {
#ifdef RT2QMD_ACTIVATE_TIMER
        if (USE_CLOCK) {
            if (!threadIdx.x)
                timer[blockIdx.x] = clock64();
        }
#endif
        Buffer::readModelFromBuffer(blockIdx.x);
        __syncthreads();
        if (Buffer::model_cached->za_nuc[1].y != 1) {
            while (true) {
                MeanField::prepareNuclei(true);
                if (MeanField::sampleNuclei(true))
                    break;
            }
        }
        else {
            MeanField::sampleNucleon(true);
        }
        __syncthreads();
        Buffer::writeModelToBuffer(blockIdx.x);
#ifdef RT2QMD_ACTIVATE_TIMER
        if (USE_CLOCK) {
            if (!threadIdx.x)
                timer[blockIdx.x] = clock64() - timer[blockIdx.x];
        }
        __syncthreads();
#endif
        return;
    }


    __host__ cudaError_t __host__prepareTarget(int block, int thread) {
        __kernel__prepareTarget <<< block, thread, mcutil::SIZE_SHARED_MEMORY_QMD >>> ();
        return cudaDeviceSynchronize();
    }


    __global__ void __kernel__propagate() {
#ifdef RT2QMD_PERSISTENT_DISPATCHER
        {
            __syncthreads();
            int meta_idx = Buffer::Metadata::current_metadata_idx[blockIdx.x];
            __syncthreads();
            Buffer::readModelFromBuffer(meta_idx);
        }
#else
        Buffer::readModelFromBuffer(blockIdx.x);
#endif
        __syncthreads();
        int n = (int)(
            Buffer::model_cached->za_nuc[0].y +
            Buffer::model_cached->za_nuc[1].y
        );
        MeanField::setDimension(n);
        __syncthreads();

        // mean field injection
        // MeanField::testG4QMDSampleSystem();

        // set participant target tape & type
        for (int i = 0; i <= n / blockDim.x; ++i) {
            int   pp = i * blockDim.x + threadIdx.x;
            if (pp < n)
                Buffer::model_cached->participant_idx[pp] = (unsigned char)(pp);
            __syncthreads();
        }
        MeanField::cal2BodyQuantities();

        // phase space injection
        /*
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            for (int i = Buffer::model_cached->offset_1d; i < n + Buffer::model_cached->offset_1d; ++i) {
                printf("%d %f %f %f %f %f %f \n",
                    Buffer::Participant::flags[i],
                    Buffer::Participant::position_x[i],
                    Buffer::Participant::position_y[i],
                    Buffer::Participant::position_z[i],
                    Buffer::Participant::momentum_x[i],
                    Buffer::Participant::momentum_y[i],
                    Buffer::Participant::momentum_z[i]
                );
            }
        }
        */

        for (int i = 0; i < constants::PROPAGATE_MAX_TIME; ++i) {
            //if (blockIdx.x == 0 && threadIdx.x == 0)
            //    printf("              iter %d \n", i);
            MeanField::doPropagate();
            Collision::calKinematicsOfBinaryCollisions();
#ifndef NDEBUG
            MeanField::calTotalKineticEnergy();
            MeanField::calTotalPotentialEnergy();
            MeanField::doClusterJudgement();
            QMD_DUMP_ACTION__(__FUNCTION__, __FILE__, 28, REACTION::FILE_LEN);
#endif
        }
        {
            __syncthreads();
            int meta_idx = Buffer::model_cached->meta_idx;
            __syncthreads();
            Buffer::writeModelToBuffer(meta_idx);
        }
    }


    __host__ cudaError_t __host__propagate(int block, int thread) {
        __kernel__propagate <<< block, thread, mcutil::SIZE_SHARED_MEMORY_QMD >>> ();
        return cudaDeviceSynchronize();
    }


    __global__ void __kernel__finalize(int field_target) {
#ifdef RT2QMD_PERSISTENT_DISPATCHER
        {
            __syncthreads();
            int meta_idx = Buffer::Metadata::current_metadata_idx[blockIdx.x];
            __syncthreads();
            Buffer::readModelFromBuffer(meta_idx);
        }
#else
        Buffer::readModelFromBuffer(blockIdx.x);
#endif
        __syncthreads();
        int n = (int)(
            Buffer::model_cached->za_nuc[0].y +
            Buffer::model_cached->za_nuc[1].y
            );
        MeanField::setDimension(n);
        __syncthreads();
        Buffer::model_cached->initial_flags.qmd.phase = MODEL_STAGE::MODEL_IDLE;
        MeanField::checkFieldIntegrity();
        __syncthreads();
        bool fail = Buffer::model_cached->initial_flags.qmd.fmask & MODEL_FLAGS::MODEL_FAIL_PROPAGATION;
        __syncthreads();
        if (fail)
            returnModel(field_target);
        else {
            MeanField::doClusterJudgement();
            if (MeanField::calculateClusterKinematics()) {
                MeanField::writeSingleNucleons();
            }
            else  // elastic, bring back problem to buffer
                returnModel(field_target);
        }
        {
            __syncthreads();
            int meta_idx = Buffer::model_cached->meta_idx;
            __syncthreads();
#ifdef RT2QMD_PERSISTENT_DISPATCHER
            Buffer::Metadata::meta_queue[0].push(meta_idx);
            __syncthreads();
#endif
            Buffer::writeModelToBuffer(meta_idx);
        }
    }


    __host__ cudaError_t __host__finalize(int block, int thread, int field_target) {
        __kernel__finalize <<< block, thread, mcutil::SIZE_SHARED_MEMORY_QMD >>> (field_target);
        return cudaDeviceSynchronize();
    }


    __host__ void __host__deviceResetModelBuffer(int block, int thread, bool return_data) {
        __device__deviceResetModelBuffer <<< block, thread >>> (return_data);
    }


    __global__ void __device__deviceResetModelBuffer(bool return_data) {
        // reset model buffer (blockwise)
        for (int i = 0; i <= Buffer::Metadata::N_METADATA / gridDim.x; ++i) {
            int idx = i * gridDim.x + blockIdx.x;
            if (idx < Buffer::Metadata::N_METADATA) {
                Buffer::readModelFromBuffer(idx);
                __syncthreads();
                unsigned short phase = Buffer::model_cached->initial_flags.qmd.phase;
                __syncthreads();
                if (phase > MODEL_STAGE::MODEL_IDLE) {
                    Buffer::model_cached->initial_flags.qmd.phase = MODEL_STAGE::MODEL_IDLE;
                    if (return_data) {
                        // dimension
                        int dim  = Buffer::model_cached->za_nuc[0].y + Buffer::model_cached->za_nuc[1].y;
                        assert(dim >= 2);

                        int sbid = dim / CUDA_WARP_SIZE;
                        if (dim % CUDA_WARP_SIZE)
                            sbid++;
                        __syncthreads();
                        returnModel(sbid);
                    }
                }
                __syncthreads();
                Buffer::writeModelToBuffer(idx);
            }
        }
        // reset metadata (threadwise)
        // head & tail
        Buffer::Metadata::meta_queue[0].head = Buffer::Metadata::N_METADATA;
        Buffer::Metadata::meta_queue[0].tail = 0u;
        Buffer::Metadata::meta_queue[1].head = 0u;
        Buffer::Metadata::meta_queue[1].tail = 0u;
        // queue data
        for (int i = 0; i <= Buffer::Metadata::N_METADATA / (gridDim.x * blockDim.x); ++i) {
            int idx = (i * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx < Buffer::Metadata::N_METADATA)
                Buffer::Metadata::meta_queue[0].idx[idx] = idx;
            __syncthreads();
        }
    }


    __host__ cudaError_t setPtrTimer(bool use_clock, long long int* clock_ptr) {
        M_SOAPtrMapper(bool, use_clock, USE_CLOCK);
        if (use_clock)
            M_SOASymbolMapper(long long int*, clock_ptr, timer);
        return cudaSuccess;
    }


#ifndef NDEBUG


    __global__ void __kernel__testG4QMDSampleSystem() {
        Buffer::readModelFromBuffer();
        MeanField::testG4QMDSampleSystem();
    }


    __host__ void testG4QMDSampleMeanFieldSystem(int block, int thread) {
        __kernel__testG4QMDSampleSystem <<< block, thread, mcutil::SIZE_SHARED_MEMORY_QMD >>> ();
    }


#endif


}