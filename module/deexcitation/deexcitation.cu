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
 * @file    module/deexcitation/deexcitation.cu
 * @brief   Deexcitation kernel
 * @author  CM Lee
 * @date    07/15/2024
 */


#include "deexcitation.cuh"
#include "device/shuffle.cuh"

#include <stdio.h>


namespace deexcitation {


    __device__ void boostAndWrite(mcutil::BUFFER_TYPE origin, mcutil::UNION_FLAGS flags, float m0, bool is_secondary) {
        int lane_idx = threadIdx.x % CUDA_WARP_SIZE;

        int  mask, size, offset;
        bool target_evap, target_nsec;
        bool long_lived;

        int* cache_univ_i = reinterpret_cast<int*>(mcutil::cache_univ);
        int  targetp      = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];

        // ZA of remnant
        uchar4* cache_zaev = reinterpret_cast<uchar4*>(mcutil::cache_univ + CUDA_WARP_SIZE);
        uchar4  zaev       = cache_zaev[threadIdx.x];

        // momentum of initial nucleus
        float3 axis, momentum;
        float  etot, m, exc;
        axis.x = -buffer_catalog[origin].u[targetp];  // [GeV/c]
        axis.y = -buffer_catalog[origin].v[targetp];
        axis.z = -buffer_catalog[origin].w[targetp];

        // DEBUG phasespace
        assert(!isnan(axis.x));
        assert(!isnan(axis.y));
        assert(!isnan(axis.z));

        float p    = norm3df(axis.x, axis.y, axis.z);  // momentum norm [GeV/c]

        // normalize
        if (p > 0.f) {
            axis.x /= p;
            axis.y /= p;
            axis.z /= p;
        }
        // float sint0 = axis.x * axis.x + axis.y * axis.y;

        // calculate the Lorentz boost vector
        float beta  = p / sqrtf(p * p + m0 * m0 * 1e-6f);  // m0 to GeV/c^2
        float gamma = 1.f / sqrtf(1.f - beta * beta);
        float alpha = (gamma * gamma) / (1.f + gamma);

        // primary remnant
        m          = mcutil::cache_univ[CUDA_WARP_SIZE + (1 + is_secondary) * blockDim.x + threadIdx.x] * 1e-3f;  // [GeV/c^2]
        exc        = mcutil::cache_univ[CUDA_WARP_SIZE + (3 + is_secondary) * blockDim.x + threadIdx.x];          // [MeV]
        momentum.x = mcutil::cache_univ[CUDA_WARP_SIZE + 5 * blockDim.x + threadIdx.x] * 1e-3f;  // [GeV/c]
        momentum.y = mcutil::cache_univ[CUDA_WARP_SIZE + 6 * blockDim.x + threadIdx.x] * 1e-3f;
        momentum.z = mcutil::cache_univ[CUDA_WARP_SIZE + 7 * blockDim.x + threadIdx.x] * 1e-3f;
        etot       = norm4df(momentum.x, momentum.y, momentum.z, m + exc * 1e-3f);

        if (is_secondary) {
            momentum.x = -momentum.x;
            momentum.y = -momentum.y;
            momentum.z = -momentum.z;
        }

        float bp = momentum.x * axis.x + momentum.y * axis.y * momentum.z * axis.z;
        bp *= beta;

        // boost
        momentum.x += beta * axis.x * (alpha * bp - gamma * etot);
        momentum.y += beta * axis.y * (alpha * bp - gamma * etot);
        momentum.z += beta * axis.z * (alpha * bp - gamma * etot);

        // boost to Z direction
        /*
        momentum.z -= beta * (gamma * (gamma / (1.f + gamma) * momentum.z * beta - etot));

        p = norm3df(momentum.x, momentum.y, momentum.z);

        // rotate
        float sint0i, sphi0, cphi0, u2p;
        if (sint0 > 1e-10f) {
            sint0 = sqrtf(sint0);
            sint0i = 1.f / sint0;
            cphi0 = sint0i * axis.x;
            sphi0 = sint0i * axis.y;

            u2p = axis.z * momentum.x + sint0 * momentum.z;
            momentum.z = axis.z * momentum.z - sint0 * momentum.x;
            momentum.x = u2p * cphi0 - momentum.y * sphi0;
            momentum.y = u2p * sphi0 + momentum.y * cphi0;
        }
        */

        // product is in evaporation chain
        // build vector 

        mcutil::UNION_FLAGS deex_flags(flags);

        deex_flags.deex.z = is_secondary ? zaev.z : zaev.x;
        deex_flags.deex.a = is_secondary ? zaev.w : zaev.y;

        long_lived     = long_lived_table.longLived(deex_flags.deex.z, deex_flags.deex.a);

        target_evap = flags.base.fmask & FLAGS::FLAG_CHANNEL_FOUND ? true : false;
        if (exc < MIN_EXC_ENERGY && (long_lived || deex_flags.deex.a >= MAX_A_BREAKUP))
            target_evap = false;

        if (deex_flags.deex.a <= 1)
            target_evap = false;

        if (flags.base.fmask & (FLAGS::FLAG_CHANNEL_PHOTON | FLAGS::FLAG_CHANNEL_FISSION))
            target_evap = false;

        mask = __ballot_sync(0xffffffff, target_evap);
        size = __popc(mask);
        if (size) {
            // index
            int target_from = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];
            if (!lane_idx)
                offset = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].pushAtomicWarp(size);
            offset  = __shfl_sync(0xffffffff, offset, 0);
            mask   &= ~(0xffffffff << lane_idx);
            mask    = __popc(mask);
            offset += mask;
            offset %= buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].size;
            if (target_evap) {

                assert(deex_flags.deex.a > 1);  // proton & neutron cannot be the target of deexcitation

                // position
                buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].x[offset]
                    = buffer_catalog[origin].x[target_from];
                buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].y[offset]
                    = buffer_catalog[origin].y[target_from];
                buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].z[offset]
                    = buffer_catalog[origin].z[target_from];

                // weight
                buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].wee[offset]
                    = buffer_catalog[origin].wee[target_from];

                // flag
                buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].flags[offset]
                    = deex_flags.astype<unsigned int>();

                // DEBUG phasespace
                assert(!isnan(momentum.x));
                assert(!isnan(momentum.y));
                assert(!isnan(momentum.z));

                assert(!isinf(momentum.x));
                assert(!isinf(momentum.y));
                assert(!isinf(momentum.z));

                // momentum
                buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].u[offset] = momentum.x;
                buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].v[offset] = momentum.y;
                buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].w[offset] = momentum.z;

                // energy
                buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].e[offset] = exc;

                if (BUFFER_HAS_HID) {
                    buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].hid[offset]
                        = buffer_catalog[origin].hid[target_from];
                }
            }
        }
        __syncthreads();

        // product is in result vector 

        target_nsec = !target_evap;

        if (flags.base.fmask & (FLAGS::FLAG_CHANNEL_PHOTON | FLAGS::FLAG_CHANNEL_FISSION))
            target_nsec = false;

        if (deex_flags.deex.a <= 0)
            target_nsec = false;

        mask = __ballot_sync(0xffffffff, target_nsec);
        size = __popc(mask);
        if (size) {
            // index
            int target_from = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];
            if (!lane_idx)
                offset = buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].pushAtomicWarp(size);
            offset  = __shfl_sync(0xffffffff, offset, 0);
            mask   &= ~(0xffffffff << lane_idx);
            mask    = __popc(mask);
            offset += mask;
            offset %= buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].size;

            if (target_nsec) {

                assert(deex_flags.deex.a > 0);

                // position
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].x[offset]
                    = buffer_catalog[origin].x[target_from];
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].y[offset]
                    = buffer_catalog[origin].y[target_from];
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].z[offset]
                    = buffer_catalog[origin].z[target_from];

                // weight
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].wee[offset]
                    = buffer_catalog[origin].wee[target_from];
                // flag
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].flags[offset]
                    = deex_flags.astype<unsigned int>();

                // DEBUG phasespace
                assert(!isnan(momentum.x));
                assert(!isnan(momentum.y));
                assert(!isnan(momentum.z));

                assert(!isinf(momentum.x));
                assert(!isinf(momentum.y));
                assert(!isinf(momentum.z));

                // momentum
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].u[offset] = momentum.x;
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].v[offset] = momentum.y;
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].w[offset] = momentum.z;

                if (BUFFER_HAS_HID) {
                    buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].hid[offset]
                        = buffer_catalog[origin].hid[target_from];
                }
            }
        }
        __syncthreads();
    }
    

    namespace Dostrovsky {

        
        __global__ void __kernel__deexcitationStep() {
            int idx      = threadIdx.x + blockDim.x * blockIdx.x;
            int lane_idx = threadIdx.x % CUDA_WARP_SIZE;

            // pull particle data from buffer
            buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].pullShared(blockDim.x);

            // position & direction are not required here
            // cache structure
            // cache_univ[0:2] -> buffer idx, parent
        
            // cache_univ[32:32 + 8 * blockDim.x] -> de-excitation emission probability (for each channels)
            // 

            int* cache_univ_i = reinterpret_cast<int*>(mcutil::cache_univ);
            int  targetp      = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];

            // data
            float        exc_energy = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].e[targetp];

            // extract ZA number
            mcutil::UNION_FLAGS flags(buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].flags[targetp]);
            uchar4 za_evap;
            za_evap.x = (unsigned char)flags.deex.z;
            za_evap.y = (unsigned char)flags.deex.a;
            za_evap.z = 0;
            za_evap.w = 0;

            assert(flags.deex.a > 0);

            // initialize flags
            flags.base.fmask = 0u;

            // check target is stable
            if (za_evap.y <= 1)
                flags.base.fmask |= FLAGS::FLAG_IS_STABLE;
            if (exc_energy < MIN_EXC_ENERGY && long_lived_table.longLived(za_evap.x, za_evap.y))
                flags.base.fmask |= FLAGS::FLAG_IS_STABLE;
            __syncthreads();

            // total probability
            float prob_cumul = 0.f;
            float prob;

            // mass
            float mass_nuc   = mass_table[za_evap.x].get(za_evap.y);

            // photon branch probability
            prob        = photon::emissionProbability(za_evap.x, za_evap.y, mass_nuc, exc_energy, false);
            prob_cumul += prob;
            mcutil::cache_univ[CUDA_WARP_SIZE + threadIdx.x] = prob;
            __syncthreads();

            // fission branch probability
            prob = 0.f;
            if (DO_FISSION)
                prob = fission::emissionProbability(za_evap.x, za_evap.y, exc_energy);
            prob_cumul += prob;
            mcutil::cache_univ[CUDA_WARP_SIZE + threadIdx.x + blockDim.x] = prob;
            __syncthreads();

            // evaporation branch probability
            for (int channel = CHANNEL::CHANNEL_NEUTRON; channel < CHANNEL::CHANNEL_2N; ++channel) {
                prob        = emissionProbability((CHANNEL)channel, za_evap.x, za_evap.y, mass_nuc, exc_energy);
                prob_cumul += prob;
                mcutil::cache_univ[CUDA_WARP_SIZE + threadIdx.x + channel * blockDim.x] = prob;
                __syncthreads();
            }

            // select branch
            CHANNEL channel = CHANNEL::CHANNEL_UNKNWON;
            if (prob_cumul <= 0.f)
                flags.base.fmask |= FLAGS::FLAG_CHANNEL_UBREAKUP;
            __syncthreads();

            prob_cumul *= 1.f - curand_uniform(&rand_state[idx]);
            prob_cumul *= 0.999f;  // FP error
            for (int i = CHANNEL::CHANNEL_PHOTON; i < CHANNEL::CHANNEL_2N; ++i) {
                prob_cumul -= mcutil::cache_univ[CUDA_WARP_SIZE + threadIdx.x + i * blockDim.x];
                if (prob_cumul <= 0.f && !(flags.base.fmask & (FLAGS::FLAG_CHANNEL_FOUND | FLAGS::FLAG_CHANNEL_UBREAKUP))) {
                    channel = (CHANNEL)i;
                    flags.base.fmask |= FLAGS::FLAG_CHANNEL_FOUND;
                }
                __syncthreads();
            }

            // exclude photon evaporation branch
            int mask, size, offset; 
            mask = __ballot_sync(0xffffffff, channel == CHANNEL::CHANNEL_PHOTON);
            size = __popc(mask);
            if (size) {
                // index
                int target_from = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];
                if (!lane_idx)
                    offset = buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].pushAtomicWarp(size);
                offset  = __shfl_sync(0xffffffff, offset, 0);
                mask   &= ~(0xffffffff << lane_idx);
                mask    = __popc(mask);
                offset += mask;
                offset %= buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].size;
                if (channel == CHANNEL::CHANNEL_PHOTON) {
                    flags.base.fmask |= FLAGS::FLAG_CHANNEL_PHOTON;

                    // position
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].x[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].x[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].y[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].y[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].z[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].z[target_from];

                    // weight
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].wee[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].wee[target_from];

                    // flag
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].flags[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].flags[target_from];

                    // momentum
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].u[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].u[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].v[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].v[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].w[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].w[target_from];

                    // energy
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].e[offset] = exc_energy;

                    if (BUFFER_HAS_HID) {
                        buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].hid[offset]
                            = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].hid[target_from];
                    }
                }
            }
            __syncthreads();

            // exclude fission branch
            mask = __ballot_sync(0xffffffff, channel == CHANNEL::CHANNEL_FISSION);
            size = __popc(mask);
            if (size) { // index
                int target_from = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];
                if (!lane_idx)
                    offset = buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].pushAtomicWarp(size);
                offset  = __shfl_sync(0xffffffff, offset, 0);
                mask   &= ~(0xffffffff << lane_idx);
                mask    = __popc(mask);
                offset += mask;
                offset %= buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].size;
                if (channel == CHANNEL::CHANNEL_FISSION) {
                    flags.base.fmask |= FLAGS::FLAG_CHANNEL_FISSION;

                    // position
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].x[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].x[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].y[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].y[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].z[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].z[target_from];

                    // weight
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].wee[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].wee[target_from];

                    // flag
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].flags[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].flags[target_from];

                    // momentum
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].u[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].u[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].v[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].v[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].w[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].w[target_from];

                    // energy
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].e[offset] = exc_energy;

                    if (BUFFER_HAS_HID) {
                        buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].hid[offset]
                            = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].hid[target_from];
                    }
                }
            }
            __syncthreads();

            //printf("%d %d %f %d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e \n", za_evap.x, za_evap.y, exc_energy, (int)channel,
            //    mcutil::cache_univ[CUDA_WARP_SIZE + threadIdx.x],
            //    mcutil::cache_univ[CUDA_WARP_SIZE + threadIdx.x + blockDim.x],
            //    mcutil::cache_univ[CUDA_WARP_SIZE + threadIdx.x + blockDim.x * 2],
            //    mcutil::cache_univ[CUDA_WARP_SIZE + threadIdx.x + blockDim.x * 3],
            //    mcutil::cache_univ[CUDA_WARP_SIZE + threadIdx.x + blockDim.x * 4],
            //    mcutil::cache_univ[CUDA_WARP_SIZE + threadIdx.x + blockDim.x * 5],
            //    mcutil::cache_univ[CUDA_WARP_SIZE + threadIdx.x + blockDim.x * 6],
            //    mcutil::cache_univ[CUDA_WARP_SIZE + threadIdx.x + blockDim.x * 7]
            //);

            // sample secondaries
            // cache_univ[32                 :32 +     blockDim.x] -> ZA number of primary remnant & secondary particle
            // cache_univ[32 + 1 * blockDim.x:32 + 2 * blockDim.x] -> primary remnant mass, ground [MeV]
            // cache_univ[32 + 2 * blockDim.x:32 + 3 * blockDim.x] -> secondary remnant mass, ground [MeV]
            // cache_univ[32 + 3 * blockDim.x:32 + 4 * blockDim.x] -> excitation energy of primary  remnant
            // cache_univ[32 + 4 * blockDim.x:32 + 5 * blockDim.x] -> excitation energy of seconary remnant (used in fission)
            // cache_univ[32 + 5 * blockDim.x:32 + 6 * blockDim.x] -> CM momentum X
            // cache_univ[32 + 6 * blockDim.x:32 + 7 * blockDim.x] -> CM momentum Y
            // cache_univ[32 + 7 * blockDim.x:32 + 8 * blockDim.x] -> CM momentum Z

            // initialize shared cache

            uchar4* cache_zaev = reinterpret_cast<uchar4*>(mcutil::cache_univ + CUDA_WARP_SIZE);
            cache_zaev[threadIdx.x] = za_evap;

            mcutil::cache_univ[CUDA_WARP_SIZE +     blockDim.x + threadIdx.x] = mass_nuc;
            mcutil::cache_univ[CUDA_WARP_SIZE + 3 * blockDim.x + threadIdx.x] = exc_energy;

            // initialize momentum vector (for stables)
            mcutil::cache_univ[CUDA_WARP_SIZE + 5 * blockDim.x + threadIdx.x] = 0.f;
            mcutil::cache_univ[CUDA_WARP_SIZE + 6 * blockDim.x + threadIdx.x] = 0.f;
            mcutil::cache_univ[CUDA_WARP_SIZE + 7 * blockDim.x + threadIdx.x] = 0.f;

            // handle particle evaporation
            if (channel >= CHANNEL::CHANNEL_NEUTRON && 
                channel <  CHANNEL::CHANNEL_UNKNWON) {
                emitParticle(&rand_state[idx], channel);
            }
            __syncthreads();
        
            // handle unstable breakup (A < 30)
            if (flags.base.fmask & FLAGS::FLAG_CHANNEL_UBREAKUP && !(flags.base.fmask & FLAGS::FLAG_IS_STABLE) && za_evap.y < MAX_A_BREAKUP) {
                if (breakup::emitParticle(&rand_state[idx]))
                    flags.base.fmask |= FLAGS::FLAG_CHANNEL_FOUND;
            }
        
            __syncthreads();

            // Lorentz boost & write to buffer
            float total_energy = mass_nuc + exc_energy;

            boostAndWrite(mcutil::BUFFER_TYPE::DEEXCITATION, flags, total_energy, false);  // primary remnant
            boostAndWrite(mcutil::BUFFER_TYPE::DEEXCITATION, flags, total_energy, true);   // secondary remnant
        }


        __host__ void deexcitationStep(int block, int thread) {
            __kernel__deexcitationStep <<< block, thread, mcutil::SIZE_SHARED_MEMORY_GLOBAL >>> ();
        }


    }


    namespace Chatterjee {


        __global__ void __kernel__deexcitationStep() {
            int idx      = threadIdx.x + blockDim.x * blockIdx.x;
            int lane_idx = threadIdx.x % CUDA_WARP_SIZE;

            // pull particle data from buffer
            buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].pullShared(blockDim.x);

            // position & direction are not required here
            // cache structure
            // cache_univ[0:2] -> buffer idx, parent
        
            // cache_univ[INTEGRATE_SHARED_MEM_OFFSET                 :INTEGRATE_SHARED_MEM_OFFSET + 2 * blockDim.x] -> de-excitation emission probability (photon, fission)
            // cache_univ[INTEGRATE_SHARED_MEM_OFFSET + 2 * blockDim.x:INTEGRATE_SHARED_MEM_OFFSET + 3 * blockDim.x] -> index of the selected channel
            // cache_univ[INTEGRATE_SHARED_MEM_OFFSET + 3 * blockDim.x:INTEGRATE_SHARED_MEM_OFFSET + 4 * blockDim.x] -> maximum probability (for rejection unity)

            int* cache_univ_i = reinterpret_cast<int*>(mcutil::cache_univ);
            int  targetp      = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];

            // data
            float        exc_energy = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].e[targetp];

            // extract ZA number
            mcutil::UNION_FLAGS flags(buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].flags[targetp]);
            uchar4 za_evap;
            za_evap.x = (unsigned char)flags.deex.z;
            za_evap.y = (unsigned char)flags.deex.a;
            za_evap.z = 0;
            za_evap.w = 0;

            assert(flags.deex.a > 0);

            // initialize flags
            flags.base.fmask = 0u;

            // check target is stable
            if (za_evap.y <= 1)
                flags.base.fmask |= FLAGS::FLAG_IS_STABLE;
            if (exc_energy < MIN_EXC_ENERGY && long_lived_table.longLived(za_evap.x, za_evap.y))
                flags.base.fmask |= FLAGS::FLAG_IS_STABLE;
            __syncthreads();

            // total probability
            float prob;

            // mass
            float mass_nuc   = mass_table[za_evap.x].get(za_evap.y);

            // photon branch probability
            prob = photon::emissionProbability(za_evap.x, za_evap.y, mass_nuc, exc_energy, false);
            mcutil::cache_univ[INTEGRATE_SHARED_MEM_OFFSET + threadIdx.x] = prob;
            __syncthreads();

            // fission branch probability
            prob = 0.f;
            if (DO_FISSION)
                prob = fission::emissionProbability(za_evap.x, za_evap.y, exc_energy);
            mcutil::cache_univ[INTEGRATE_SHARED_MEM_OFFSET + threadIdx.x + blockDim.x] = prob;
            __syncthreads();

            // shared memory
            IntegrateSharedMem* smem = (IntegrateSharedMem*)mcutil::cache_univ;

            // evaporation branch probability (blockwise working group)
            for (int i = 0; i < blockDim.x; ++i) {
                // initialize prob
                if (threadIdx.x < CHANNEL::CHANNEL_2N) {
                    smem->channel_prob[threadIdx.x]     = threadIdx.x < CHANNEL::CHANNEL_NEUTRON 
                        ? mcutil::cache_univ[INTEGRATE_SHARED_MEM_OFFSET + i + blockDim.x * threadIdx.x]
                        : 0.f;
                    smem->channel_prob_max[threadIdx.x] = 0.f;
                }
                __syncthreads();
                for (int channel = CHANNEL::CHANNEL_NEUTRON; channel < CHANNEL::CHANNEL_2N; ++channel) {
                    if (threadIdx.x == i)
                        setSharedParameters(channel, za_evap.x, za_evap.y, mass_nuc, exc_energy);
                    __syncthreads();
                    integrateProbability();
                }
                // select the channel
                if (threadIdx.x == i) {
                    flags = selectChannel(flags);
                }
                __syncthreads();
                /*
                if (threadIdx.x == i) {
                    int channel = evaporation::selectChannel(&flags);
                    if (channel >= CHANNEL::CHANNEL_NEUTRON && channel < CHANNEL::CHANNEL_2N)
                        setChatterjeeSharedParameters(channel, za_evap.x, za_evap.y, mass_nuc, exc_energy);  // reset parameters
                }
                __syncthreads();
                int channel = smem->channel;
                __syncthreads();
                if (channel >= CHANNEL::CHANNEL_NEUTRON && channel < CHANNEL::CHANNEL_2N)
                    evaporation::sampleEmitParticleEnergy();
                __syncthreads();
                */
            }

            // get channel from the shared memory
            int* cache_channel = reinterpret_cast<int*>(mcutil::cache_univ + INTEGRATE_SHARED_MEM_OFFSET + 2 * blockDim.x);

            CHANNEL channel  = (CHANNEL)cache_channel[threadIdx.x];

            // exclude photon evaporation branch
            int mask, size, offset;
            mask = __ballot_sync(0xffffffff, channel == CHANNEL::CHANNEL_PHOTON);
            size = __popc(mask);
            if (size) {
                // index
                int target_from = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];
                if (!lane_idx)
                    offset = buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].pushAtomicWarp(size);
                offset = __shfl_sync(0xffffffff, offset, 0);
                mask &= ~(0xffffffff << lane_idx);
                mask = __popc(mask);
                offset += mask;
                offset %= buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].size;
                if (channel == CHANNEL::CHANNEL_PHOTON) {
                    flags.base.fmask |= FLAGS::FLAG_CHANNEL_PHOTON;

                    // position
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].x[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].x[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].y[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].y[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].z[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].z[target_from];

                    // weight
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].wee[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].wee[target_from];

                    // flag
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].flags[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].flags[target_from];

                    // momentum
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].u[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].u[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].v[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].v[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].w[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].w[target_from];

                    // energy
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].e[offset] = exc_energy;

                    if (BUFFER_HAS_HID) {
                        buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].hid[offset]
                            = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].hid[target_from];
                    }
                }
            }
            __syncthreads();

            // exclude fission branch
            mask = __ballot_sync(0xffffffff, channel == CHANNEL::CHANNEL_FISSION);
            size = __popc(mask);
            if (size) { // index
                int target_from = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];
                if (!lane_idx)
                    offset = buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].pushAtomicWarp(size);
                offset = __shfl_sync(0xffffffff, offset, 0);
                mask &= ~(0xffffffff << lane_idx);
                mask = __popc(mask);
                offset += mask;
                offset %= buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].size;
                if (channel == CHANNEL::CHANNEL_FISSION) {
                    flags.base.fmask |= FLAGS::FLAG_CHANNEL_FISSION;

                    // position
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].x[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].x[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].y[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].y[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].z[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].z[target_from];

                    // weight
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].wee[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].wee[target_from];

                    // flag
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].flags[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].flags[target_from];

                    // momentum
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].u[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].u[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].v[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].v[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].w[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].w[target_from];

                    // energy
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].e[offset] = exc_energy;

                    if (BUFFER_HAS_HID) {
                        buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].hid[offset]
                            = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].hid[target_from];
                    }
                }
            }
            __syncthreads();

            // sample secondaries (before energy sampling)
            // shared memory
            // cache_univ[32                 :32 +      blockDim.x] -> ZA number of primary remnant & secondary particle
            // cache_univ[32 + 1 * blockDim.x:32 +  2 * blockDim.x] -> mass of the parant nucleus, ground [MeV/c^2]
            // cache_univ[32 + 2 * blockDim.x:32 +  3 * blockDim.x] -> mass of the primary remnant, ground [MeV/c^2]
            // cache_univ[32 + 3 * blockDim.x:32 +  4 * blockDim.x] -> excitation energy of the parant nucleus [MeV]
            // cache_univ[32 + 4 * blockDim.x:32 +  5 * blockDim.x] -> rejection unity
            // cache_univ[32 + 5 * blockDim.x:32 +  6 * blockDim.x] -> coulomb barrier [MeV]
            // cache_univ[32 + 6 * blockDim.x:32 +  7 * blockDim.x] -> Chatterjee p
            // cache_univ[32 + 7 * blockDim.x:32 +  8 * blockDim.x] -> Chatterjee landa
            // cache_univ[32 + 8 * blockDim.x:32 +  9 * blockDim.x] -> Chatterjee mu
            // cache_univ[32 + 9 * blockDim.x:32 + 10 * blockDim.x] -> Chatterjee nu

            // sample secondaries (after energy sampling)
            // shared memory
            // cache_univ[32                 :32 +     blockDim.x] -> ZA number of primary remnant & secondary particle
            // cache_univ[32 + 1 * blockDim.x:32 + 2 * blockDim.x] -> primary remnant mass, ground [MeV]
            // cache_univ[32 + 2 * blockDim.x:32 + 3 * blockDim.x] -> secondary remnant mass, ground [MeV]
            // cache_univ[32 + 3 * blockDim.x:32 + 4 * blockDim.x] -> excitation energy of primary remnant
            // cache_univ[32 + 4 * blockDim.x:32 + 5 * blockDim.x] -> excitation energy of seconary remnant (used in fission)
            // cache_univ[32 + 5 * blockDim.x:32 + 6 * blockDim.x] -> CM momentum X
            // cache_univ[32 + 6 * blockDim.x:32 + 7 * blockDim.x] -> CM momentum Y
            // cache_univ[32 + 7 * blockDim.x:32 + 8 * blockDim.x] -> CM momentum Z

            // initialize shared cache

            uchar4* cache_zaev = reinterpret_cast<uchar4*>(mcutil::cache_univ + CUDA_WARP_SIZE);
            cache_zaev[threadIdx.x] = za_evap;

            float   max_prob = mcutil::cache_univ[INTEGRATE_SHARED_MEM_OFFSET + threadIdx.x + 3 * blockDim.x];
            __syncthreads();

            // channel shared -> energy shared

            mcutil::cache_univ[CUDA_WARP_SIZE +     blockDim.x + threadIdx.x] = mass_nuc;
            mcutil::cache_univ[CUDA_WARP_SIZE + 3 * blockDim.x + threadIdx.x] = exc_energy;
            mcutil::cache_univ[CUDA_WARP_SIZE + 4 * blockDim.x + threadIdx.x] = max_prob * 1.05f;  // safety margin of rejection unity (1.05)

            // initialize momentum vector (for stables)
            mcutil::cache_univ[CUDA_WARP_SIZE + 5 * blockDim.x + threadIdx.x] = 0.f;
            mcutil::cache_univ[CUDA_WARP_SIZE + 6 * blockDim.x + threadIdx.x] = 0.f;
            mcutil::cache_univ[CUDA_WARP_SIZE + 7 * blockDim.x + threadIdx.x] = 0.f;

            // handle particle evaporation
            if (channel >= CHANNEL::CHANNEL_NEUTRON &&
                channel < CHANNEL::CHANNEL_UNKNWON) {
                emitParticle(&rand_state[idx], channel);
            }
            __syncthreads();

            // handle unstable breakup (A < 30)
            if (flags.base.fmask & FLAGS::FLAG_CHANNEL_UBREAKUP && !(flags.base.fmask & FLAGS::FLAG_IS_STABLE) && za_evap.y < MAX_A_BREAKUP) {
                if (breakup::emitParticle(&rand_state[idx]))
                    flags.base.fmask |= FLAGS::FLAG_CHANNEL_FOUND;
            }
            __syncthreads();

            // Lorentz boost & write to buffer
            float total_energy = mass_nuc + exc_energy;

            boostAndWrite(mcutil::BUFFER_TYPE::DEEXCITATION, flags, total_energy, false);  // primary remnant
            boostAndWrite(mcutil::BUFFER_TYPE::DEEXCITATION, flags, total_energy, true);   // secondary remnant
        }


        __host__ void deexcitationStep(int block, int thread) {
            __kernel__deexcitationStep <<< block, thread, mcutil::SIZE_SHARED_MEMORY_GLOBAL >>> ();
        }


    }


    namespace Kalbach {


        __global__ void __kernel__deexcitationStep() {
            int idx      = threadIdx.x + blockDim.x * blockIdx.x;
            int lane_idx = threadIdx.x % CUDA_WARP_SIZE;

            // pull particle data from buffer
            buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].pullShared(blockDim.x);

            // position & direction are not required here
            // cache structure
            // cache_univ[0:2] -> buffer idx, parent
        
            // cache_univ[INTEGRATE_SHARED_MEM_OFFSET                 :INTEGRATE_SHARED_MEM_OFFSET + 2 * blockDim.x] -> de-excitation emission probability (photon, fission)
            // cache_univ[INTEGRATE_SHARED_MEM_OFFSET + 2 * blockDim.x:INTEGRATE_SHARED_MEM_OFFSET + 3 * blockDim.x] -> index of the selected channel
            // cache_univ[INTEGRATE_SHARED_MEM_OFFSET + 3 * blockDim.x:INTEGRATE_SHARED_MEM_OFFSET + 4 * blockDim.x] -> maximum probability (for rejection unity)
            
            int* cache_univ_i = reinterpret_cast<int*>(mcutil::cache_univ);
            int  targetp      = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];

            // data
            float        exc_energy = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].e[targetp];

            // extract ZA number
            mcutil::UNION_FLAGS flags(buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].flags[targetp]);
            uchar4 za_evap;
            za_evap.x = (unsigned char)flags.deex.z;
            za_evap.y = (unsigned char)flags.deex.a;
            za_evap.z = 0;
            za_evap.w = 0;

            assert(flags.deex.a > 0);

            // initialize flags
            flags.base.fmask = 0u;

            // check target is stable
            if (za_evap.y <= 1)
                flags.base.fmask |= FLAGS::FLAG_IS_STABLE;
            if (exc_energy < MIN_EXC_ENERGY && long_lived_table.longLived(za_evap.x, za_evap.y))
                flags.base.fmask |= FLAGS::FLAG_IS_STABLE;
            __syncthreads();

            // total probability
            float prob;

            // mass
            float mass_nuc   = mass_table[za_evap.x].get(za_evap.y);

            // photon branch probability
            prob = photon::emissionProbability(za_evap.x, za_evap.y, mass_nuc, exc_energy, false);
            mcutil::cache_univ[INTEGRATE_SHARED_MEM_OFFSET + threadIdx.x] = prob;
            __syncthreads();

            // fission branch probability
            prob = 0.f;
            if (DO_FISSION)
                prob = fission::emissionProbability(za_evap.x, za_evap.y, exc_energy);
            mcutil::cache_univ[INTEGRATE_SHARED_MEM_OFFSET + threadIdx.x + blockDim.x] = prob;
            __syncthreads();

            // shared memory
            IntegrateSharedMem* smem = (IntegrateSharedMem*)mcutil::cache_univ;


            // evaporation branch probability (blockwise working group)
            for (int i = 0; i < blockDim.x; ++i) {
                // initialize prob
                if (threadIdx.x < CHANNEL::CHANNEL_2N) {
                    smem->channel_prob[threadIdx.x] = threadIdx.x < CHANNEL::CHANNEL_NEUTRON
                        ? mcutil::cache_univ[INTEGRATE_SHARED_MEM_OFFSET + i + blockDim.x * threadIdx.x]
                        : 0.f;
                    smem->channel_prob_max[threadIdx.x] = 0.f;
                }
                __syncthreads();
                for (int channel = CHANNEL::CHANNEL_NEUTRON; channel < CHANNEL::CHANNEL_2N; ++channel) {
                    if (threadIdx.x == i)
                        setSharedParameters(channel, za_evap.x, za_evap.y, mass_nuc, exc_energy);
                    __syncthreads();
                    integrateProbability();
                }
                // select the channel
                if (threadIdx.x == i) {
                    flags = selectChannel(flags);
                }
                __syncthreads();
            }

            // get channel from the shared memory
            int* cache_channel = reinterpret_cast<int*>(mcutil::cache_univ + INTEGRATE_SHARED_MEM_OFFSET + 2 * blockDim.x);

            CHANNEL channel = (CHANNEL)cache_channel[threadIdx.x];

            // exclude photon evaporation branch
            int mask, size, offset;
            mask = __ballot_sync(0xffffffff, channel == CHANNEL::CHANNEL_PHOTON);
            size = __popc(mask);
            if (size) {
                // index
                int target_from = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];
                if (!lane_idx)
                    offset = buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].pushAtomicWarp(size);
                offset = __shfl_sync(0xffffffff, offset, 0);
                mask &= ~(0xffffffff << lane_idx);
                mask = __popc(mask);
                offset += mask;
                offset %= buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].size;
                if (channel == CHANNEL::CHANNEL_PHOTON) {
                    flags.base.fmask |= FLAGS::FLAG_CHANNEL_PHOTON;

                    // position
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].x[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].x[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].y[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].y[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].z[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].z[target_from];

                    // weight
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].wee[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].wee[target_from];

                    // flag
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].flags[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].flags[target_from];

                    // momentum
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].u[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].u[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].v[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].v[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].w[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].w[target_from];

                    // energy
                    buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].e[offset] = exc_energy;

                    if (BUFFER_HAS_HID) {
                        buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].hid[offset]
                            = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].hid[target_from];
                    }
                }
            }
            __syncthreads();

            // exclude fission branch
            mask = __ballot_sync(0xffffffff, channel == CHANNEL::CHANNEL_FISSION);
            size = __popc(mask);
            if (size) { // index
                int target_from = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];
                if (!lane_idx)
                    offset = buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].pushAtomicWarp(size);
                offset = __shfl_sync(0xffffffff, offset, 0);
                mask &= ~(0xffffffff << lane_idx);
                mask = __popc(mask);
                offset += mask;
                offset %= buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].size;
                if (channel == CHANNEL::CHANNEL_FISSION) {
                    flags.base.fmask |= FLAGS::FLAG_CHANNEL_FISSION;

                    // position
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].x[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].x[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].y[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].y[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].z[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].z[target_from];

                    // weight
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].wee[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].wee[target_from];

                    // flag
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].flags[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].flags[target_from];

                    // momentum
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].u[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].u[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].v[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].v[target_from];
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].w[offset]
                        = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].w[target_from];

                    // energy
                    buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].e[offset] = exc_energy;

                    if (BUFFER_HAS_HID) {
                        buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].hid[offset]
                            = buffer_catalog[mcutil::BUFFER_TYPE::DEEXCITATION].hid[target_from];
                    }
                }
            }
            __syncthreads();

            // sample secondaries (before energy sampling)
            // shared memory
            // cache_univ[32                  :32 +      blockDim.x] -> ZA number of primary remnant & secondary particle
            // cache_univ[32 + 1  * blockDim.x:32 +  2 * blockDim.x] -> mass of the parant nucleus, ground [MeV/c^2]
            // cache_univ[32 + 2  * blockDim.x:32 +  3 * blockDim.x] -> mass of the primary remnant, ground [MeV/c^2]
            // cache_univ[32 + 3  * blockDim.x:32 +  4 * blockDim.x] -> excitation energy of the parant nucleus [MeV]
            // cache_univ[32 + 4  * blockDim.x:32 +  5 * blockDim.x] -> rejection unity
            // cache_univ[32 + 5  * blockDim.x:32 +  6 * blockDim.x] -> coulomb barrier [MeV]
            // cache_univ[32 + 6  * blockDim.x:32 +  7 * blockDim.x] -> Kalbach p
            // cache_univ[32 + 7  * blockDim.x:32 +  8 * blockDim.x] -> Kalbach lambda
            // cache_univ[32 + 8  * blockDim.x:32 +  9 * blockDim.x] -> Kalbach mu
            // cache_univ[32 + 9  * blockDim.x:32 + 10 * blockDim.x] -> Kalbach nu
            // cache_univ[32 + 10 * blockDim.x:32 + 11 * blockDim.x] -> Kalbach geom

            // sample secondaries (after energy sampling)
            // shared memory
            // cache_univ[32                 :32 +     blockDim.x] -> ZA number of primary remnant & secondary particle
            // cache_univ[32 + 1 * blockDim.x:32 + 2 * blockDim.x] -> primary remnant mass, ground [MeV]
            // cache_univ[32 + 2 * blockDim.x:32 + 3 * blockDim.x] -> secondary remnant mass, ground [MeV]
            // cache_univ[32 + 3 * blockDim.x:32 + 4 * blockDim.x] -> excitation energy of primary remnant
            // cache_univ[32 + 4 * blockDim.x:32 + 5 * blockDim.x] -> excitation energy of seconary remnant (used in fission)
            // cache_univ[32 + 5 * blockDim.x:32 + 6 * blockDim.x] -> CM momentum X
            // cache_univ[32 + 6 * blockDim.x:32 + 7 * blockDim.x] -> CM momentum Y
            // cache_univ[32 + 7 * blockDim.x:32 + 8 * blockDim.x] -> CM momentum Z

            // initialize shared cache

            uchar4* cache_zaev = reinterpret_cast<uchar4*>(mcutil::cache_univ + CUDA_WARP_SIZE);
            cache_zaev[threadIdx.x] = za_evap;

            float   max_prob = mcutil::cache_univ[INTEGRATE_SHARED_MEM_OFFSET + threadIdx.x + 3 * blockDim.x];
            __syncthreads();

            // channel shared -> energy shared

            mcutil::cache_univ[CUDA_WARP_SIZE +     blockDim.x + threadIdx.x] = mass_nuc;
            mcutil::cache_univ[CUDA_WARP_SIZE + 3 * blockDim.x + threadIdx.x] = exc_energy;
            mcutil::cache_univ[CUDA_WARP_SIZE + 4 * blockDim.x + threadIdx.x] = max_prob * 1.05f;  // safety margin of rejection unity (1.05)

            // initialize momentum vector (for stables)
            mcutil::cache_univ[CUDA_WARP_SIZE + 5 * blockDim.x + threadIdx.x] = 0.f;
            mcutil::cache_univ[CUDA_WARP_SIZE + 6 * blockDim.x + threadIdx.x] = 0.f;
            mcutil::cache_univ[CUDA_WARP_SIZE + 7 * blockDim.x + threadIdx.x] = 0.f;

            // handle particle evaporation
            if (channel >= CHANNEL::CHANNEL_NEUTRON &&
                channel < CHANNEL::CHANNEL_UNKNWON) {
                emitParticle(&rand_state[idx], channel);
            }
            __syncthreads();

            // handle unstable breakup (A < 30)
            if (flags.base.fmask & FLAGS::FLAG_CHANNEL_UBREAKUP && !(flags.base.fmask & FLAGS::FLAG_IS_STABLE) && za_evap.y < MAX_A_BREAKUP) {
                if (breakup::emitParticle(&rand_state[idx]))
                    flags.base.fmask |= FLAGS::FLAG_CHANNEL_FOUND;
            }
            __syncthreads();

            // Lorentz boost & write to buffer
            float total_energy = mass_nuc + exc_energy;

            boostAndWrite(mcutil::BUFFER_TYPE::DEEXCITATION, flags, total_energy, false);  // primary remnant
            boostAndWrite(mcutil::BUFFER_TYPE::DEEXCITATION, flags, total_energy, true);   // secondary remnant
        }


        __host__ void deexcitationStep(int block, int thread) {
            __kernel__deexcitationStep <<< block, thread, mcutil::SIZE_SHARED_MEMORY_GLOBAL >>> ();
        }


    }


}