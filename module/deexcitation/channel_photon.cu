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
 * @file    module/deexcitation/channel_photon.cu
 * @brief   Photon evaporation channel
 * @author  CM Lee
 * @date    07/09/2024
 */


#include "channel_photon.cuh"
#include "deexcitation.cuh"

#include "device/shuffle.cuh"

#include <stdio.h>


namespace deexcitation {
    namespace photon {

        
        __device__ float emissionProbability(int z, int a, float mass, float exc_energy, bool store_shared_mem) {
            float emax = mass_table[z].get(a - 1) + constants::MASS_NEUTRON - mass;
            emax = fminf(0.99f * exc_energy, fmaxf(0.f, emax));

            if (z <= 0 || a <= 1 || z == a || exc_energy < MIN_EXC_ENERGY)
                emax = 0.f;

            float eres2 = GDREnergy(a);

            if (exc_energy >= GDRE_FACTOR * eres2)
                emax = 0.f;

            float fstep    = emax / (float)(PHOTON_EVAP_MAX_DE_POINT - 1);

            if (store_shared_mem)
                mcutil::cache_univ[CUDA_WARP_SIZE + threadIdx.x] = fstep;

            // numerical integrate
            
            float wres2    = 0.3f * eres2;
            eres2 = eres2 * eres2;
            wres2 = wres2 * wres2;

            float clevel   = getLevelDensityParameter(a);
            float xsqr     = sqrtf(clevel * exc_energy);
            float egam     = exc_energy;
            float gamma_e2 = egam * egam;
            float gamma_r2 = gamma_e2 * wres2;
            float egdp2    = gamma_e2 - eres2;

            float p0 = expf(-2.f * xsqr) * gamma_r2 * gamma_e2 / (egdp2 * egdp2 + gamma_r2);
            float p1 = 0.f;

            double fcum = 0.f;  // cumulative -> double
            for (int i = 1; i < PHOTON_EVAP_MAX_DE_POINT; ++i) {
                egam    -= fstep;
                gamma_e2 = egam * egam;
                gamma_r2 = gamma_e2 * wres2;
                egdp2    = gamma_e2 - eres2;
                p1       = expf(2.f * (sqrtf(clevel * fabsf(exc_energy - egam)) - xsqr)) 
                    * gamma_r2 * gamma_e2 / (egdp2 * egdp2 + gamma_r2);
                float fseg = p0 + p1;
                if (store_shared_mem)
                    mcutil::cache_univ[CUDA_WARP_SIZE + i * blockDim.x + threadIdx.x] = fseg * fstep * PHOTON_EVAP_NORM * (float)a;
                fcum    += fseg;
                p0       = p1;
                __syncthreads();
            }
            return fcum * fstep * PHOTON_EVAP_NORM * (float)a;
        }


        __device__ float sampleContinuumEnergy(curandState* state, uchar4 zaev, float mass, float exc_energy) {
            // build cumulative probability table
            float ptot  = emissionProbability((int)zaev.x, (int)zaev.y, mass, exc_energy, true);
            float fstep = mcutil::cache_univ[CUDA_WARP_SIZE + threadIdx.x];
            ptot *= 1.f - curand_uniform(state);
            int i;
            for (i = 1; i < PHOTON_EVAP_MAX_DE_POINT; ++i) {
                ptot -= mcutil::cache_univ[CUDA_WARP_SIZE + i * blockDim.x + threadIdx.x];
                if (ptot < 0.f)
                    break;
            }
            __syncthreads();

            float efinal = fminf(exc_energy, fstep * ((float)i - curand_uniform(state)));
            float nden   = deexcitation::getLevelDensityParameter((int)zaev.y);
            return exc_energy < 1.f / nden ? 0.f : efinal;
        }


        __device__ void boostAndWriteGamma(float m0) {
            int* cache_univ_i = reinterpret_cast<int*>(mcutil::cache_univ);
            int  targetp      = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];

            // momentum of initial nucleus
            float3 axis, momentum;
            float  etot;
            axis.x = -buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].u[targetp];  // [GeV/c]
            axis.y = -buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].v[targetp];
            axis.z = -buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].w[targetp];

            float p = norm3df(axis.x, axis.y, axis.z);  // momentum norm [GeV/c]

            // normalize
            if (p > 0.f) {
                axis.x /= p;
                axis.y /= p;
                axis.z /= p;
            }

            // calculate the Lorentz boost vector
            float beta  = p / sqrtf(p * p + m0 * m0 * 1e-6f);  // m0 to GeV/c^2
            float gamma = 1.f / sqrtf(1.f - beta * beta);
            float alpha = (gamma * gamma) / (1.f + gamma);

            momentum.x = -mcutil::cache_univ[CUDA_WARP_SIZE + 5 * blockDim.x + threadIdx.x];  // [MeV/c]
            momentum.y = -mcutil::cache_univ[CUDA_WARP_SIZE + 6 * blockDim.x + threadIdx.x];
            momentum.z = -mcutil::cache_univ[CUDA_WARP_SIZE + 7 * blockDim.x + threadIdx.x];
            etot       = norm3df(momentum.x, momentum.y, momentum.z);

            float bp = momentum.x * axis.x + momentum.y * axis.y * momentum.z * axis.z;
            bp *= beta;

            // boost
            momentum.x += beta * axis.x * (alpha * bp - gamma * etot);
            momentum.y += beta * axis.y * (alpha * bp - gamma * etot);
            momentum.z += beta * axis.z * (alpha * bp - gamma * etot);
            etot        = norm3df(momentum.x, momentum.y, momentum.z);
            if (etot > 0.f) {
                momentum.x /= etot;
                momentum.y /= etot;
                momentum.z /= etot;
            }
            __syncthreads();

            // push secondary gamma
            int target;
            target = buffer_catalog[mcutil::BUFFER_TYPE::PHOTON].pushBulk();
            buffer_catalog[mcutil::BUFFER_TYPE::PHOTON].x[target] 
                = buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].x[targetp];
            buffer_catalog[mcutil::BUFFER_TYPE::PHOTON].y[target] 
                = buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].y[targetp];
            buffer_catalog[mcutil::BUFFER_TYPE::PHOTON].z[target]
                = buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].z[targetp];
            buffer_catalog[mcutil::BUFFER_TYPE::PHOTON].u[target]   = momentum.x;
            buffer_catalog[mcutil::BUFFER_TYPE::PHOTON].v[target]   = momentum.y;
            buffer_catalog[mcutil::BUFFER_TYPE::PHOTON].w[target]   = momentum.z;
            buffer_catalog[mcutil::BUFFER_TYPE::PHOTON].e[target]   = etot;
            buffer_catalog[mcutil::BUFFER_TYPE::PHOTON].wee[target] 
                = buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].wee[targetp];
            if (BUFFER_HAS_HID)
                buffer_catalog[mcutil::BUFFER_TYPE::PHOTON].hid[target] 
                = buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].hid[targetp];
            // ro is not necessary
        }


        __global__ void __kernel__continuumEvaporationStep() {
            int idx      = threadIdx.x + blockDim.x * blockIdx.x;

            // pull particle data from buffer
            buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].pullShared(blockDim.x);

            int* cache_univ_i = reinterpret_cast<int*>(mcutil::cache_univ);
            int  targetp      = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];

            // load data
            float        exc_energy = buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].e[targetp];
            mcutil::UNION_FLAGS flags(buffer_catalog[mcutil::BUFFER_TYPE::PHOTON_EVAP].flags[targetp]);

            // extract ZA number
            uchar4 za_evap;
            za_evap.x = flags.deex.z;
            za_evap.y = flags.deex.a;
            za_evap.z = 0;
            za_evap.w = 0;

            // mass
            float mass_nuc = mass_table[za_evap.x].get(za_evap.y);

            // initialize flags
            flags.base.fmask  = 0u;
            flags.base.fmask |= FLAGS::FLAG_CHANNEL_FOUND;

            // sample remnant energy
            float efinal = sampleContinuumEnergy(&rand_state[idx], za_evap, mass_nuc, exc_energy);

            // sample secondaries
            // cache_univ[32                 :32 +     blockDim.x] -> ZA number of primary remnant & secondary particle
            // cache_univ[32 + 1 * blockDim.x:32 + 2 * blockDim.x] -> primary remnant mass, ground [MeV]
            // cache_univ[32 + 2 * blockDim.x:32 + 3 * blockDim.x] -> secondary remnant mass, ground [MeV] -> 0 in gamma
            // cache_univ[32 + 3 * blockDim.x:32 + 4 * blockDim.x] -> excitation energy of primary  remnant
            // cache_univ[32 + 4 * blockDim.x:32 + 5 * blockDim.x] -> gamma energy
            // cache_univ[32 + 5 * blockDim.x:32 + 6 * blockDim.x] -> CM momentum X
            // cache_univ[32 + 6 * blockDim.x:32 + 7 * blockDim.x] -> CM momentum Y
            // cache_univ[32 + 7 * blockDim.x:32 + 8 * blockDim.x] -> CM momentum Z
            // initialize shared cache

            uchar4* cache_zaev = reinterpret_cast<uchar4*>(mcutil::cache_univ + CUDA_WARP_SIZE);
            cache_zaev[threadIdx.x] = za_evap;

            mcutil::cache_univ[CUDA_WARP_SIZE +     blockDim.x + threadIdx.x] = mass_nuc;
            mcutil::cache_univ[CUDA_WARP_SIZE + 3 * blockDim.x + threadIdx.x] = efinal;

            // sample the momentum of photon (isotropic)
            // momentum == gamma energy in c=1
            float p = exc_energy - efinal;
            mcutil::cache_univ[CUDA_WARP_SIZE + 4 * blockDim.x + threadIdx.x] = p;
            
            // random isotropic direction
            float cost, sint;
            float cosp, sinp;
            float angle;

            // polar
            cost  = 1.f - 2.f * curand_uniform(&rand_state[idx]);
            sint  = sqrtf(fmaxf(0.f, 1.f - cost * cost));
            // azimuthal
            angle = constants::FP32_TWO_PI * curand_uniform(&rand_state[idx]);
            __sincosf(angle, &sinp, &cosp);

            mcutil::cache_univ[CUDA_WARP_SIZE + 5 * blockDim.x + threadIdx.x] = p * sint * cosp;  // X
            mcutil::cache_univ[CUDA_WARP_SIZE + 6 * blockDim.x + threadIdx.x] = p * sint * sinp;  // Y
            mcutil::cache_univ[CUDA_WARP_SIZE + 7 * blockDim.x + threadIdx.x] = p * cost;         // Z

            // Lorentz boost & write to buffer
            float total_energy = mass_nuc + exc_energy;

            boostAndWrite(mcutil::BUFFER_TYPE::PHOTON_EVAP, flags, total_energy, false);  // primary remnant
            boostAndWriteGamma(total_energy);   // secondary gamma
        }


        __host__ void continuumEvaporationStep(int block, int thread) {
            __kernel__continuumEvaporationStep <<< block, thread, mcutil::SIZE_SHARED_MEMORY_PEVAP >>> ();
        }


    }
}