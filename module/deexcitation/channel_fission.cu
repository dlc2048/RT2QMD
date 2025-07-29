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
 * @file    module/deexcitation/channel_fission.cu
 * @brief   Competitive fission channel
 * @author  CM Lee
 * @date    07/09/2024
 */


#include "channel_fission.cuh"
#include "deexcitation.cuh"
#include "device/shuffle.cuh"

#include <stdio.h>


namespace deexcitation {
    namespace fission {


        __device__ float* cameron_spz;
        __device__ float* cameron_spn;
        __device__ float* cameron_sz;
        __device__ float* cameron_sn;
        __device__ float* cameron_pz;
        __device__ float* cameron_pn;


        __device__ float BarashenkovFissionBarrier(int z, int a) {
            int n = a - z;

            // fissibility parameter
            float x = 0.5f * BARASHENKOV_COULOMB / BARASHENKOV_SURFACE * (float)(z * z) / float(a);
            x /= 1.f - BARASHENKOV_K * (float)((n - z) * (n - z)) / (float)(a * a);

            // BFO
            float bfo = BARASHENKOV_SURFACE * powf((float)a, constants::TWO_OVER_THREE);
            bfo *= x <= constants::TWO_OVER_THREE
                ? 0.38f * (0.75f - x)
                : 0.83f * (1.f - x) * (1.f - x) * (1.f - x);

            int d = n - 2 * (n / 2) + z - 2 * (z / 2);

            return bfo + BARASHENKOV_DELTA * d - CameronSpinPairingCorrection(z, n);
        }


        __device__ float emissionProbability(int z, int a, float exc_energy) {
            // check nuclei
            if (z < 16 || a < 65)
                return 0.f;

            // check effective excitation energy
            float exc_eff = exc_energy - fissionPairingCorrection(z, a - z);
            if (exc_eff <= 0.f)
                return 0.f;

            // check maximum kinetic energy
            float max_eke = exc_eff - fissionBarrier(z, a, exc_eff);
            if (max_eke <= 0.f)
                return 0.f;

            // check available energy
            float ucompound = exc_energy - pairingCorrection(z, a - z);
            float ufission  = exc_energy - fissionPairingCorrection(z, a - z);
            if (ucompound < 0.f || ufission < 0.f)
                return 0.f;

            float system_entropy = 2.f * sqrtf(getLevelDensityParameter(a) * ucompound);
            float afission       = getFissionLevelDensityParameter(z, a);

            float cf   = 2.f * sqrtf(afission * max_eke);
            float exp1 = system_entropy      <= 160.f ? expf(-system_entropy)      : 0.f;
            float exp2 = system_entropy - cf <= 160.f ? expf(-system_entropy + cf) : 0.f;

            return (exp1 + (cf - 1.f) * exp2) / (constants::FP32_FOUR_PI * afission);
        }


        __device__ void emitParticle(curandState* state, uchar4 zaev, float m0, float exc_energy) {
          
            // define fission parameters
            float pcorr  = fissionPairingCorrection(zaev.x, zaev.y - zaev.x);
            float u      = fminf(200.f, exc_energy);
            float sigma2 = (zaev.y <= 235) ? 5.6f : 5.6f + 0.096f * (float)(zaev.y - 235); 
            float sigmas = 0.8f * expf(0.00553f * u + 2.1386f);
            float as     = (float)zaev.y * 0.5f;
            float fb     = fissionBarrier(zaev.x, zaev.y, exc_energy);
            float w      = fissionWeight(zaev, u, fb, sigma2, sigmas);

            uchar4 zaev_init = zaev;  // reserve initial state
            float m1, m2;
            float frag_eke, frag_exc;
            while (true) {

                // first fragment
                zaev.y = (unsigned char)sampleMassNumber(state, zaev_init.y, as, sigma2, sigmas, w);
                zaev.x = (unsigned char)sampleAtomicNumber(state, zaev_init.y, zaev_init.x, zaev.y);
                m1     = mass_table[zaev.x].get(zaev.y);

                // second fragment
                zaev.w = zaev_init.y - zaev.y;
                zaev.z = zaev_init.x - zaev.x;
                if (zaev.w == 0 || zaev_init.x < zaev.x || zaev.z > zaev.w)
                    continue;
                m2     = mass_table[zaev.z].get(zaev.w);

                float tmax = m0 + u - m1 - m2 - pcorr;
                if (tmax < 0.f)
                    continue;

                frag_eke = fissionKineticEnergy(
                    state, zaev_init.y, zaev_init.x, zaev.y, zaev.w, tmax, as, sigma2, sigmas, w
                );
                frag_exc = tmax - frag_eke + pcorr;

                if (frag_exc >= 0.f)
                    break;
            }

            // mass cache
            mcutil::cache_univ[CUDA_WARP_SIZE + 1 * blockDim.x + threadIdx.x] = m1;
            mcutil::cache_univ[CUDA_WARP_SIZE + 2 * blockDim.x + threadIdx.x] = m2;

            // divide excitation energy
            mcutil::cache_univ[32 + 3 * blockDim.x + threadIdx.x] = frag_exc * (float)zaev.y / (float)zaev_init.y;
            mcutil::cache_univ[32 + 4 * blockDim.x + threadIdx.x] = frag_exc * (float)zaev.w / (float)zaev_init.y;

            // ZA cache
            uchar4* cache_zaev = reinterpret_cast<uchar4*>(mcutil::cache_univ + CUDA_WARP_SIZE);
            cache_zaev[threadIdx.x] = zaev;

            // momentum norm
            float etot1    = 0.5f * ((m0 - m2) * (m0 + m2) + m1 * m1) / m0;
            float momentum = sqrtf((etot1 - m1) * (etot1 + m1));

            // random isotropic direction
            float cost, sint;
            float cosp, sinp;
            float angle;

            // polar
            cost  = 1.f - 2.f * curand_uniform(state);
            sint  = sqrtf(fmaxf(0.f, 1.f - cost * cost));
            // azimuthal
            angle = constants::FP32_TWO_PI * curand_uniform(state);
            __sincosf(angle, &sinp, &cosp);

            mcutil::cache_univ[CUDA_WARP_SIZE + 5 * blockDim.x + threadIdx.x] = momentum * sint * cosp;  // X
            mcutil::cache_univ[CUDA_WARP_SIZE + 6 * blockDim.x + threadIdx.x] = momentum * sint * sinp;  // Y
            mcutil::cache_univ[CUDA_WARP_SIZE + 7 * blockDim.x + threadIdx.x] = momentum * cost;         // Z

            return;
        }


        __device__ float fissionWeight(uchar4 zaev, float u, float fb, float sigma2, float sigmas) {
            float wa = 0.f;
            float w  = 1001.f;
            float as = (float)zaev.y * 0.5f;
            if (zaev.x >= 82) {
                if (zaev.x >= 90) {
                    wa = (u <= 16.25f)
                        ? 0.5385f  * u - 9.9564f
                        : 0.09197f * u - 2.7003f;
                }
                else if (zaev.x == 89)
                    wa = 0.09197f * u - 1.0808f;
                else {
                    float x = fmaxf(0.f, fb - 7.5f);
                    wa = 0.09197f * (u - x) - 1.0808f;
                }
                wa = expf(wa);

                float x1 = ((float)FISSION_PARAM_A1 - as) / sigma2 * 2.f;
                float x2 = ((float)FISSION_PARAM_A2 - as) / sigma2;
                float fasymasym = 2.f * localExp(x2) + localExp(x1);

                float x3 = (as - FISSION_PARAM_A3) / sigmas;
                float fsyma1a2 = localExp(x3);

                float w1 = fmaxf(1.03f * wa - fasymasym, 1e-4f);
                float w2 = fmaxf(1.f - fsyma1a2 * wa, 1e-4f);

                w = w1 / w2;
            }
            return w;
        }


        __device__ int sampleMassNumber(curandState* state, int a, float as, float sigma2, float sigmas, float w) {
            float c2a = (float)FISSION_PARAM_A2 + 3.72f * sigma2;
            float c2s = as + 3.72f * sigmas;
            float c2  = 0.f;

            if (w > 1000.f)
                c2 = c2s;
            else if (w < 0.001f)
                c2 = c2a;
            else
                c2 = fmaxf(c2a, c2s);

            float c1 = (float)a - c2;
            if (c1 < 30.f) {
                c2 = (float)a - 30.f;
                c1 = 30.f;
            }

            float am1 = (as + (float)FISSION_PARAM_A1) * 0.5f;

            // mass distribution
            float mass1 = massDistribution(as,  a, as, sigma2, sigmas, w);
            float mass2 = massDistribution(am1, a, as, sigma2, sigmas, w);
            float mass3 = massDistribution((float)FISSION_PARAM_A1, a, as, sigma2, sigmas, w);
            float mass4 = massDistribution(FISSION_PARAM_A3, a, as, sigma2, sigmas, w);
            float mass5 = massDistribution((float)FISSION_PARAM_A2, a, as, sigma2, sigmas, w);

            // get maximal value among mass1,...,mass5
            float mass_max = mass1;
            mass_max = fmaxf(mass_max, mass2);
            mass_max = fmaxf(mass_max, mass3);
            mass_max = fmaxf(mass_max, mass4);
            mass_max = fmaxf(mass_max, mass5);

            float xm, pm;
            while (true) {
                xm = c1 + curand_uniform(state) * (c2 - c1);
                pm = massDistribution(xm, a, as, sigma2, sigmas, w);
                if (mass_max * curand_uniform(state) < pm)
                    break;
            }
            return (int)xm;
        }


        __device__ int sampleAtomicNumber(curandState* state, int a, int z, int af) {
            float delta_z;

            if (af >= 134)
                delta_z = -0.45f;
            else if (af <= a - 134)
                delta_z = 0.45f;
            else
                delta_z = -0.45f * ((float)af - (float)a * 0.5f) / (134.f - (float)a * 0.5f);

            float zmean = ((float)af / (float)a) * (float)z + delta_z;
            float zf;
            do {
                zf = curand_normal(state) * FISSION_Z_SIGMA + zmean;
            } while (zf < 1.f || zf >(float)(z - 1) || zf > (float)af);
            return (int)zf;
        }


        __device__ float massDistribution(float x, int a, float as, float sigma2, float sigmas, float w) {
            float y0    = (x - as) / sigmas;
            float xsym  = localExp(y0);

            float y1    = (x - (float)FISSION_PARAM_A1) / sigma2 * 2.f;
            float y2    = (x - (float)FISSION_PARAM_A2) / sigma2;
            float z1    = (x - (float)(a - FISSION_PARAM_A1)) / sigma2 * 2.f;
            float z2    = (x - (float)(a - FISSION_PARAM_A2)) / sigma2;
            float xasym = localExp(y1) + localExp(y2) + 0.5f * (localExp(z1) + localExp(z2));

            float res;
            if (w > 1000.f)
                res = xsym;
            else if (w < 0.001f)
                res = xasym;
            else
                res = w * xsym + xasym;
            return res;
        }


        __device__ float fissionKineticEnergy(curandState* state, int a, int z, int af1, int af2, float tmax, float as, float sigma2, float sigmas, float w) {
            // maximum a
            int afmax = max(af1, af2);

            // weight
            float pas = 0.f;
            if (w <= 1000.f) {
                float x1 = (afmax - (float)FISSION_PARAM_A1) / sigma2 * 2.f;
                float x2 = (afmax - (float)FISSION_PARAM_A2) / sigma2;
                pas = 0.5f * localExp(x1) + localExp(x2);
            }
            float ps  = 0.f;
            if (w >= 0.001f) {
                float xs = (afmax - as) / sigmas;
                ps = w * localExp(xs);
            }
            float psy = pas + ps > 0.f ? ps / (pas + ps) : 0.5f;

            // fission fraction xsy and xas formed in symmetric and asymmetric modes
            float ppas = 2.5f * sigma2;
            float ppsy = w * sigmas;
            float xas  = ppas + ppsy > 0.f ? ppas / (ppas + ppsy) : 0.5f;

            // average kinetic energy
            float eaverage = 0.1071f * (float)(z * z) / powf((float)a, constants::ONE_OVER_THREE) + 22.2f;

            // fission mode
            float taverage_afmax;
            float esigma = 10.f;
            if (curand_uniform(state) > psy) {
                float scale_factor =
                    asymmetricRatio(a, FISSION_PARAM_A1 - 0.39895f * sigma2) * 0.25f +
                    asymmetricRatio(a, FISSION_PARAM_A1 + 0.39895f * sigma2) * 0.25f +
                    asymmetricRatio(a, FISSION_PARAM_A2 - 0.7979f  * sigma2) +
                    asymmetricRatio(a, FISSION_PARAM_A2 + 0.7979f  * sigma2);
                scale_factor  *= sigma2;
                taverage_afmax = (eaverage + 12.5f * (1.f - xas)) * (ppas / scale_factor) *
                    asymmetricRatio(a, (float)afmax);
            }
            else {
                taverage_afmax = (eaverage - 12.5f * (1.f - xas))
                    * symmetricRatio(a, (float)afmax)
                    / symmetricRatio(a, as + 0.7979f * sigmas);
                esigma = 8.f;
            }

            // sample energy
            float eke;
            do {
                eke = curand_normal(state) * esigma + taverage_afmax;
            } while (eke < eaverage - 3.72f * esigma ||
                eke > eaverage + 3.72f * esigma ||
                eke > tmax);
            return eke;
        }


        __global__ void __kernel__fissionStep() {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;

            // pull particle data from buffer
            buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].pullShared(blockDim.x);

            // position & direction are not required here
            // cache structure
            // cache_univ[0:2] -> buffer idx, parent

            int* cache_univ_i = reinterpret_cast<int*>(mcutil::cache_univ);
            int  targetp      = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];

            // load data
            float exc_energy = buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].e[targetp];
            mcutil::UNION_FLAGS flags(buffer_catalog[mcutil::BUFFER_TYPE::COMP_FISSION].flags[targetp]);

            // ZA number

            uchar4 za_evap;
            za_evap.x = flags.deex.z;
            za_evap.y = flags.deex.a;
            za_evap.z = 0;
            za_evap.w = 0;

            // initialize flags
            flags.base.fmask = 0u;
            flags.base.fmask |= FLAGS::FLAG_CHANNEL_FOUND;

            // mass
            float mass_nuc = mass_table[za_evap.x].get(za_evap.y);

            // sample secondaries
            // cache_univ[32                 :32 +     blockDim.x] -> ZA number of primary remnant & secondary particle
            // cache_univ[32 + 1 * blockDim.x:32 + 2 * blockDim.x] -> primary remnant mass, ground [MeV]
            // cache_univ[32 + 2 * blockDim.x:32 + 3 * blockDim.x] -> secondary remnant mass, ground [MeV]
            // cache_univ[32 + 3 * blockDim.x:32 + 4 * blockDim.x] -> excitation energy of primary  remnant
            // cache_univ[32 + 4 * blockDim.x:32 + 5 * blockDim.x] -> excitation energy of seconary remnant (used in fission)
            // cache_univ[32 + 5 * blockDim.x:32 + 6 * blockDim.x] -> CM momentum X
            // cache_univ[32 + 6 * blockDim.x:32 + 7 * blockDim.x] -> CM momentum Y
            // cache_univ[32 + 7 * blockDim.x:32 + 8 * blockDim.x] -> CM momentum Z
            emitParticle(&rand_state[idx], za_evap, mass_nuc, exc_energy);

            // Lorentz boost & write to buffer
            float total_energy = mass_nuc + exc_energy;

            

            boostAndWrite(mcutil::BUFFER_TYPE::COMP_FISSION, flags, total_energy, false);  // primary remnant
            boostAndWrite(mcutil::BUFFER_TYPE::COMP_FISSION, flags, total_energy, true);   // secondary remnant
        }


        __host__ void fissionStep(int block, int thread) {
            __kernel__fissionStep <<< block, thread, mcutil::SIZE_SHARED_MEMORY_GLOBAL >>> ();
        }


        __host__ cudaError_t setCameronSpinPairingCorrections(float* ptr_spz, float* ptr_spn) {
            M_SOAPtrMapper(float*, ptr_spz, cameron_spz);
            M_SOAPtrMapper(float*, ptr_spn, cameron_spn);
            return cudaSuccess;
        }


        __host__ cudaError_t setCameronSpinCorrections(float* ptr_sz, float* ptr_sn) {
            M_SOAPtrMapper(float*, ptr_sz, cameron_sz);
            M_SOAPtrMapper(float*, ptr_sn, cameron_sn);
            return cudaSuccess;
        }


        __host__ cudaError_t setCameronPairingCorrections(float* ptr_pz, float* ptr_pn) {
            M_SOAPtrMapper(float*, ptr_pz, cameron_pz);
            M_SOAPtrMapper(float*, ptr_pn, cameron_pn);
            return cudaSuccess;
        }


    }
}