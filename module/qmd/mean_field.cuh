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
 * @file    module/qmd/mean_field.cuh
 * @brief   QMD mean field (G4QMDMeanField.hh)
 * @author  CM Lee
 * @date    02/16/2024
 */


#pragma once

#include <cuda_runtime.h>
#include <assert.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "constants.cuh"
#include "buffer.cuh"

#include "device/shuffle.cuh"
#include "device/assert.cuh"


namespace RT2QMD {
    namespace MeanField {


        /**
        * @brief set mean field dimension
        * @param n field dimension
        */
        __device__ void setDimension(int n);


        /**
        * @brief Calculate two-body quntities for hamiltonian coefficients
        */
        __device__ void cal2BodyQuantities(bool prepare_collision=false);


        /**
        * @brief Calculate gradient for hamiltonian
        */
        __device__ void calGraduate(bool post_step=false);


        /**
        * @brief Calculate potential of target participant
        * @param i  participant id
        * @param ci charge number of participant
        */
        __device__ float getPotential(int i, bool ci);


        /**
        * @brief Calculate total kinetic energy
        */
        __device__ void calTotalKineticEnergy();


        /**
        * @brief Calculate excitation energy and net momentum
        */
        __device__ void calExcitationEnergyAndMomentum();


        /**
        * @brief Calculate total potential energy
        */
        __device__ void calTotalPotentialEnergy();


        /**
        * @brief for caching memory in nuclei sampling phase
        */
        typedef struct NucleiSharedMem {
            float __offset_shared[Buffer::MODEL_CACHING_OFFSET];
            int   blocked;   //! @brief global participant blocking flag
            float radm;      //! @brief Wood-saxon radius parameter
            float rt00;      //! @brief Wood-saxon radius parameter 
            float rmax;      //! @brief Wood-saxon radius parameter
            float pfm;       //! @brief momentum sampling coefficient
            float dpot;      //! @brief momentum sampling coefficient
            float ps_cumul;  //! @brief phase-space distance for pauli principle
            int   ia;        //! @brief sampling loop index a
            int   ib;        //! @brief sampling loop index b
            float x[CUDA_WARP_SIZE];
            float y[CUDA_WARP_SIZE];
            float z[CUDA_WARP_SIZE];
            float ps[MAX_DIMENSION_CLUSTER];   //! @brief temporary  phase-space distance data
            float phg[MAX_DIMENSION_CLUSTER];  //! @brief cumulative phase-space distance data
        } NucleiSharedMem;


        constexpr int NUCLEI_SHARED_MEM_OFFSET 
            = (sizeof(NucleiSharedMem) / 4) - 3 * CUDA_WARP_SIZE - MAX_DIMENSION_CLUSTER;


        /**
        * @brief for caching memory in gradient phase
        */
        typedef struct GradientSharedMem {
            float __offset_shared[NUCLEI_SHARED_MEM_OFFSET];
            int   flag[CUDA_WARP_SIZE];
            float mass[CUDA_WARP_SIZE];
            float rx[CUDA_WARP_SIZE];
            float ry[CUDA_WARP_SIZE];
            float rz[CUDA_WARP_SIZE];
            float px[CUDA_WARP_SIZE];
            float py[CUDA_WARP_SIZE];
            float pz[CUDA_WARP_SIZE];
            float ei[CUDA_WARP_SIZE];
            float vi[CUDA_WARP_SIZE];
            float rh3d[CUDA_WARP_SIZE];
            float ffrx[CUDA_WARP_SIZE];
            float ffry[CUDA_WARP_SIZE];
            float ffrz[CUDA_WARP_SIZE];
            float ffpx[CUDA_WARP_SIZE];
            float ffpy[CUDA_WARP_SIZE];
            float ffpz[CUDA_WARP_SIZE];
        } GradientSharedMem;


        /**
        * @brief Prepare nuclei property before sampling stage
        * @param target True if sample target else projectile
        */
        __device__ void prepareNuclei(bool target);


        /**
        * @brief Sample nucleon phase-space for nuclei
        * @param target True if sample target else projectile
        * 
        * @return true if success, false elsewhere
        */
        __device__ bool sampleNuclei(bool target);


        /**
        * @brief Sample single nucleon phase-space
        * @param target True if sample target else  projectile
        */
        __device__ void sampleNucleon(bool target);


        /**
        * @brief Sample nucleon positions in nuclei
        * @param target True if sample target else projectile
        *
        * @return true if success, false elsewhere
        */
        __device__ bool sampleNucleonPosition(bool target);


        /**
        * @brief Sample nucleon momenta in nuclei
        * @param target True if sample target else projectile
        *
        * @return true if success, false elsewhere
        */
        __device__ bool sampleNucleonMomentum(bool target);


        /**
        * @brief Kill CM and angular motion, then check the energy diffferences
        * @param target True if sample target else projectile
        *
        * @return true if success, false elsewhere
        */
        __device__ bool finalizeNuclei(bool target);


        /**
        * @brief Adjust all nucleon to center of mass system
        */
        __device__ void killCMMotion(bool target, int offset);


        /**
        * @brief Kill angular momentum
        */
        __device__ void killAngularMomentum(bool target, int offset);


        /**
        * @brief Adjust nuclei to make excitation energy as zero
        */
        __device__ bool forcingConservationLaw(bool target);


        /**
        * @brief Adjust nucleon phase-space 
        */
        __device__ void adjustNucleonPhaseSpace(int offset);


        /**
        * @brief Adjust nucleon phase-space to sampled impact parameter
        */
        __device__ void setNucleiPhaseSpace(bool target, int offset);


        __device__ void doPropagate();


        /**
        * @brief for caching memory in clustering phase
        */
        typedef struct ClusterSharedMem {
            float __offset_shared[Buffer::MODEL_CACHING_OFFSET];
            int   n_cluster;
            int   neighbor_found;
            int   cluster_found;
            short cluster_idx[MAX_DIMENSION_CLUSTER];
            float rhoa[MAX_DIMENSION_CLUSTER];
            bool  neighbor[MAX_DIMENSION_CLUSTER];
        } ClusterSharedMem;


        /**
        * @brief Remnant clustering
        */
        __device__ void doClusterJudgement();


        __device__ void checkFieldIntegrity();


        /**
        * @brief for caching memory in finalize phase
        */
        typedef struct FinalizeSharedMem {
            float __offset_shared[Buffer::MODEL_CACHING_OFFSET];
            int   elastic_count[2];                // projectile like, target like
            int   elastic_like_energy;
            int   dim_original;
            int   iter_original;
            int   cluster_z[CUDA_WARP_SIZE];       // atomic number of cluster
            int   cluster_a[CUDA_WARP_SIZE];       // mass number of cluster
            float cluster_mass[CUDA_WARP_SIZE];    // mass of cluster [GeV/c^2]
            float cluster_excit[CUDA_WARP_SIZE];   // excitation energy of cluster [GeV]
            float cluster_u[CUDA_WARP_SIZE];       // direction vector x
            float cluster_v[CUDA_WARP_SIZE];       // direction vector y
            float cluster_w[CUDA_WARP_SIZE];       // direction vector z
        } FinalizeSharedMem;


        /**
        * @brief Calculate cluster kinematics
        * 
        * @return true if inelastic otherwise false
        */
        __device__ bool calculateClusterKinematics();


        /**
        * @brief Write nucleons
        * 
        */
        __device__ void writeSingleNucleons();


//#ifndef NDEBUG

        constexpr int    __SYSTEM_TEST_DIMENSION = 24;
        constexpr short2 __TEST_PROJECTILE_ZA    = { 6, 12 };
        constexpr short2 __TEST_TARGET_ZA        = { 6, 12 };


        /**
        * @brief Initialize system for the 2body - graduate validation 
        *        Use initial state of G4QMDReaction::DoPropagation 
        *        default phase-space C12-C12 3600 MeV reaction
        */
        __device__ void testG4QMDSampleSystem();


//#endif
    }
}