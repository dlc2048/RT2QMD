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
 * @file    module/deexcitation/channel_evaporation.cu
 * @brief   Evaporation channel
 * @author  CM Lee
 * @date    07/11/2024
 */


#include "device/shuffle.cuh"

#include "channel_evaporation.cuh"
#include "cross_section.cuh"

#include <stdio.h>


namespace deexcitation {


    namespace Dostrovsky {
        namespace evaporation {
            

            __device__ float getAlphaParam(CHANNEL channel, int rz, int ra) {
                float alpha = 1.f;
                switch (channel) {
                case CHANNEL::CHANNEL_NEUTRON:
                    alpha = getAlphaParamNeutron(ra);
                    break;
                case CHANNEL::CHANNEL_PROTON:
                    alpha = 1.f + getAlphaParamHydrogen(rz);
                    break;
                case CHANNEL::CHANNEL_DEUTERON:
                    alpha = 1.f + 0.5f * getAlphaParamHydrogen(rz);
                    break;
                case CHANNEL::CHANNEL_TRITON:
                    alpha = 1.f + constants::ONE_OVER_THREE * getAlphaParamHydrogen(rz);
                    break;
                case CHANNEL::CHANNEL_HELIUM3:
                    alpha = 1.f + constants::FOUR_OVER_THREE * getAlphaParamHelium(rz);
                    break;
                case CHANNEL::CHANNEL_ALPHA:
                    alpha = 1.f + getAlphaParamHelium(rz);
                    break;
                default:
                    break;
                }
                return alpha;
            }


            __device__ float getBetaParam(CHANNEL channel, int rz, int ra) {
                return channel == CHANNEL::CHANNEL_NEUTRON ? getBetaParamNeutron(ra) : 0.f;
            }


            __device__ float emissionProbability(CHANNEL channel, int z, int a, float mass, float exc_energy) {
                int pz = PROJ_Z[channel];  // emission Z
                int pa = PROJ_A[channel];  // emission A
                int rz = z - pz;           // remnant Z
                int ra = a - pa;           // remnant A

                // allowed channel
                if (ra < pa || ra < rz || rz < 0 || (ra == pa && rz < pz) || (ra > 1 && (ra == rz || rz == 0)))
                    return 0.f;

                // available kinetic energy
                mass += exc_energy;  // total mass energy
                float res_mass = mass_table[rz].get(ra);

                float eke_max  = 0.5f * ((mass - res_mass) * (mass + res_mass) + PROJ_M2[channel]) / mass - PROJ_M[channel];
                float elim     = coulombBarrier(channel, rz, ra, 0.f);
                if (mass <= res_mass + PROJ_M[channel] + elim)
                    return 0.f;

                float eke_min   = 0.f;
                if (elim > 0.f) {
                    float res_m = mass - PROJ_M[channel] - elim;
                    eke_min = fmaxf(0.f, 0.5f * ((mass - res_m) * (mass + res_m) + PROJ_M2[channel]) / mass - PROJ_M[channel]);
                }

                if (eke_max <= eke_min)
                    return 0.f;

                float a0             = getLevelDensityParameter(a);
                float a1             = getLevelDensityParameter(ra);
                float system_entropy = 2.f * sqrtf(a0 * exc_energy);
            
                float alpha = getAlphaParam(channel, rz, ra);
                float beta  = getBetaParam(channel, rz, ra);

                float maxea = eke_max * a1;
                float term1 = beta * a1 - 1.5f + maxea;
                float term2 = (2.f * beta * a1 - 3.f) * sqrtf(maxea) + 2.f * maxea;

                float expterm1 = system_entropy <= 160.f ? expf(-system_entropy) : 0.f;
                float expterm2 = 2.f * sqrtf(maxea) - system_entropy;

                expterm2 = expf(fminf(expterm2, 160.f));

                float gfactor = PROJ_S[channel] * alpha * PROJ_M[channel] * PROB_COEFF * powf((float)ra, constants::TWO_OVER_THREE) / (a1 * a1);

                return fmaxf(0.f, (term1 * expterm1 + term2 * expterm2) * gfactor);
            }


            // simple method, Weisskopf evaporation
            __device__ float emitParticleEnergy(curandState* state, CHANNEL channel, int res_z, int res_a, float exc_energy) {
                float mass     = mcutil::cache_univ[CUDA_WARP_SIZE + blockDim.x + threadIdx.x] + exc_energy;
                float res_mass = mass_table[res_z].get(res_a);

                float eke_max  = 0.5f * ((mass - res_mass) * (mass + res_mass) + PROJ_M2[channel]) / mass - PROJ_M[channel];
            
                float elim     = coulombBarrier(channel, res_z, res_a, 0.f);
                if (mass <= res_mass + PROJ_M[channel] + elim)
                    return 0.f;
            
                float eke_min = 0.f;
                if (elim > 0.f) {
                    float res_m = mass - PROJ_M[channel] - elim;
                    eke_min = fmaxf(0.f, 0.5f * ((mass - res_m) * (mass + res_m) + PROJ_M2[channel]) / mass - PROJ_M[channel]);
                }
            
                // rejection unity
                float a1       = getLevelDensityParameter(res_a);
                float xmax     = (sqrtf(0.25f + a1 * eke_max) - 0.5f) / a1;
                float wmax     = xmax * expf(2.f * sqrtf(a1 * (eke_max - xmax)));

                // sample energy
                float x, w;
                do {
                    x = (eke_max - eke_min) * (1.f - curand_uniform(state));
                    w = x * expf(2.f * sqrtf(a1 * (eke_max - x)));
                } while (wmax * curand_uniform(state) > w);
                return res_a > 4 ? x + eke_min : eke_max;
            }



            /*
            __device__ float emitParticleEnergy(curandState* state, CHANNEL channel, int res_z, int res_a, float exc_energy) {
                //uchar4* cache_zaev = reinterpret_cast<uchar4*>(mcutil::cache_univ + CUDA_WARP_SIZE);
                //uchar4  zaev;

                double mass     = (double)mcutil::cache_univ[CUDA_WARP_SIZE + blockDim.x + threadIdx.x] + (double)exc_energy;
                double res_mass = (double)mass_table[res_z].get(res_a);

                // calculate maximum possible kinetic energy
                double eke_max = 0.5f * ((mass - res_mass) * (mass + res_mass) + PROJ_M2[channel]) / mass - PROJ_M[channel];

                // calculate minimum possible kinetic energy
                double elim    = coulombBarrier(channel, res_z, res_a, 0.f);
                double eke_min = 0.f;
                if (elim > 0.f) {
                    double res_m = mass - PROJ_M[channel] - elim;
                    eke_min = fmaxf(0.f, 0.5f * ((mass - res_m) * (mass + res_m) + PROJ_M2[channel]) / mass - PROJ_M[channel]);
                }

                // sample energy
                // Geant4 10.x.x method
                double a1   = getLevelDensityParameter(res_a + PROJ_A[channel]);
                double rb   = 4.f * a1 * eke_max;
                rb = sqrt(rb);

                double pex1 = rb < 160.f ? exp(-rb) : 0.f;

                double rk;
                double frk;
                do {
                    rk = curand_uniform(state);
                    rk = 1.f + 1 / rb * log(rk + (1.f - rk) * pex1);
                    double q1 = 1.f;
                    double q2 = 1.f;
                    if (channel == CHANNEL::CHANNEL_NEUTRON) {
                        float beta = (2.12f / (float)res_a / (float)res_a - 0.05f)
                            / (0.76f + 2.2f * powf((float)res_a, -constants::ONE_OVER_THREE));
                        q1 = 1.f + beta / eke_max;
                        q2 = q1 * sqrtf(q1);
                    }
                    frk = SSQR3 * rk * (q1 - rk * rk) / q2;

                } while (frk < curand_uniform(state));

                return eke_max * (1.f - rk * rk) + eke_min;
            }
            */


            __device__ void emitParticle(curandState* state, CHANNEL channel) {
                uchar4* cache_zaev = reinterpret_cast<uchar4*>(mcutil::cache_univ + CUDA_WARP_SIZE);
                float   exc_energy = mcutil::cache_univ[CUDA_WARP_SIZE + 3 * blockDim.x + threadIdx.x];

                // update ZA of system
                uchar4  zaev;
                zaev.z = (unsigned char)PROJ_Z[channel];
                zaev.w = (unsigned char)PROJ_A[channel];
                zaev.x = cache_zaev[threadIdx.x].x - zaev.z;
                zaev.y = cache_zaev[threadIdx.x].y - zaev.w;

                assert(cache_zaev[threadIdx.x].x >= zaev.z);
                assert(cache_zaev[threadIdx.x].y >= zaev.w);

                // update cache
                cache_zaev[threadIdx.x] = zaev;

                // momentum of emitted particle
                float eke_emit  = emitParticleEnergy(state, channel, (int)zaev.x, (int)zaev.y, exc_energy);

                float mass_emit = PROJ_M[channel];
                float momentum  = eke_emit * (eke_emit + 2.f * mass_emit);  // now it is the square of norm

                // calculate the excitation energy of primary remnant (use double to avoid FP error) 
                float  mass_rem = mass_table[zaev.x].get(zaev.y);
                float exc_rem  =
                    + mcutil::cache_univ[CUDA_WARP_SIZE + blockDim.x + threadIdx.x]  // primary mass
                    + exc_energy  // primary excitation
                    - mass_emit   // emitted mass
                    - eke_emit;   // kinetic energy of emitted particle
                exc_rem = sqrt((exc_rem * exc_rem - momentum)) - mass_rem;
                assert(exc_rem > -0.1f);
                exc_rem = fmaxf(exc_rem, 0.f);

                // update cache
                mcutil::cache_univ[CUDA_WARP_SIZE + 1 * blockDim.x + threadIdx.x] = mass_rem;
                mcutil::cache_univ[CUDA_WARP_SIZE + 2 * blockDim.x + threadIdx.x] = mass_emit;
                mcutil::cache_univ[CUDA_WARP_SIZE + 3 * blockDim.x + threadIdx.x] = exc_rem;
                mcutil::cache_univ[CUDA_WARP_SIZE + 4 * blockDim.x + threadIdx.x] = 0.f;

                // random isotropic direction
                float cost, sint;
                float cosp, sinp;
                float angle;

                // now momentum is norm
                momentum = sqrtf(momentum);

                // polar
                cost  = 1.f - 2.f * curand_uniform(state);
                sint  = sqrtf(fmaxf(0.f, 1.f - cost * cost));
                // azimuthal
                angle = constants::FP32_TWO_PI * curand_uniform(state);
                __sincosf(angle, &sinp, &cosp);

                mcutil::cache_univ[CUDA_WARP_SIZE + 5 * blockDim.x + threadIdx.x] = momentum * sint * cosp;  // X
                mcutil::cache_univ[CUDA_WARP_SIZE + 6 * blockDim.x + threadIdx.x] = momentum * sint * sinp;  // Y
                mcutil::cache_univ[CUDA_WARP_SIZE + 7 * blockDim.x + threadIdx.x] = momentum * cost;         // Z

                // Momentum of primary remnant is (-px, -py, -pz)
            }
        }


    }


    namespace Chatterjee {
        namespace evaporation {


            __device__ float computeProbabilityShared(float k) {
                // shared memory
                Chatterjee::IntegrateSharedMem* smem = (Chatterjee::IntegrateSharedMem*)mcutil::cache_univ;

                double em    = (double)PROJ_M[smem->channel];
                double m     = (double)smem->mass;
                double rm    = (double)smem->res_mass;
                double m_res = sqrt(m * m + em * em - 2.f * m * (em + k));

                float exc_res = m_res - rm;
                float e0      = fmaxf(smem->exc - smem->delta0, 0.f);
                float e1      = fmaxf(exc_res   - smem->delta1, 0.f);
                float erec    = (m * (k + em) - em * em) / m_res - em;
                erec = fmaxf(erec, 0.f);
                float xs      = crossSectionChatterjeeShared(erec);
                float prob    = (float)PROJ_S[smem->channel] * PROJ_M[smem->channel] * PROB_COEFF 
                    * expf(2.f * (sqrtf(smem->a1 * e1) - sqrtf(smem->a0 * e0))) * k * xs;
                if (exc_res < 0.f)
                    prob = 0.f;
                return prob;
            }


            __device__ float computeProbability(int channel, float k, float a0, float a1, float delta0, float delta1) {

                // shared memory, energy sampling
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

                float exc = mcutil::cache_univ[CUDA_WARP_SIZE + 3 * blockDim.x + threadIdx.x];

                double em    = (double)PROJ_M[channel];
                double m     = (double)(
                    mcutil::cache_univ[CUDA_WARP_SIZE + 1 * blockDim.x + threadIdx.x] +
                    exc
                );
                double rm    = (double)mcutil::cache_univ[CUDA_WARP_SIZE + 2 * blockDim.x + threadIdx.x];
                double m_res = sqrt(m * m + em * em - 2.f * m * (em + k));

                float exc_res = m_res - rm;
                float e0      = fmaxf(exc     - delta0, 0.f);
                float e1      = fmaxf(exc_res - delta1, 0.f);
                float erec    = (m * (k + em) - em * em) / m_res - em;
                erec = fmaxf(erec, 0.f);
                float xs      = crossSectionChatterjee(channel, erec);
                float prob    = (float)PROJ_S[channel] * PROJ_M[channel] * PROB_COEFF 
                    * expf(2.f * (sqrtf(a1 * e1) - sqrtf(a0 * e0))) * k * xs;
                if (exc_res < 0.f)
                    prob = 0.f;

                return prob;
            }


            __device__ void integrateProbability() {
                int warp_idx = threadIdx.x %  CUDA_WARP_SIZE;
                int lane_idx = threadIdx.x >> CUDA_WARP_SHIFT;
                int lane_dim = blockDim.x  /  CUDA_WARP_SIZE;
                // shared memory
                Chatterjee::IntegrateSharedMem* smem = (Chatterjee::IntegrateSharedMem*)mcutil::cache_univ;

                __syncthreads();
                bool allowed = smem->is_allowed;
                __syncthreads();
                if (!allowed)
                    return;

                // init
                if (!lane_idx) {
                    smem->redux_r1[warp_idx] = 0.f;  // integrate
                    smem->redux_r2[warp_idx] = 0.f;  // y maximum
                }
                __syncthreads();

                float temp;
                for (int i = 0; i < smem->int_iter; ++i) {
                    // 2nd step, warp reduction
                    if (i && i % smem->int_iter_2nd == 0) {
                        if (!lane_idx) {
                            temp = smem->redux_r1[warp_idx];
                            temp = rsum32_(temp);
                            if (!threadIdx.x)
                                smem->prob += temp;
                            temp = smem->redux_r2[warp_idx];
                            temp = rmax32_(temp);
                            if (!threadIdx.x)
                                smem->prob_max = fmaxf(smem->prob_max, temp);
                            __syncwarp(); // init
                            smem->redux_r1[warp_idx] = 0.f;  // integrate
                            smem->redux_r2[warp_idx] = 0.f;  // y maximum
                        }
                    }
                    __syncthreads();
                        
                    // 1st step, warp reduction
                    float ibin = fmaxf(i * blockDim.x + threadIdx.x, 0.02f);
                    float k    = smem->emin + ibin * smem->edelta;
                    float y    = computeProbabilityShared(k);

                    // maximum y
                    smem->redux_r2[lane_dim * (i % smem->int_iter_2nd) + lane_idx] = rmax32_(y);

                    // trapezoidal
                    if ((i == 0 && !threadIdx.x) || (i == smem->int_iter - 1 && threadIdx.x == blockDim.x - 1))
                        y = y * 0.5f;
                    smem->redux_r1[lane_dim * (i % smem->int_iter_2nd) + lane_idx] = rsum32_(y);

                    __syncthreads();
                }

                if (!lane_idx) {
                    temp = smem->redux_r1[warp_idx];
                    temp = rsum32_(temp);
                    if (!threadIdx.x) {
                        smem->channel_prob[smem->channel]     = (smem->prob + temp) * smem->edelta;
                    }
                    temp = smem->redux_r2[warp_idx];
                    temp = rmax32_(temp);
                    if (!threadIdx.x)
                        smem->channel_prob_max[smem->channel] = fmaxf(smem->prob_max, temp);
                }
                __syncthreads();
            }


            __device__ mcutil::UNION_FLAGS selectChannel(mcutil::UNION_FLAGS flags) {
                // shared memory
                Chatterjee::IntegrateSharedMem* smem = (Chatterjee::IntegrateSharedMem*)mcutil::cache_univ;
                int* cache_channel = reinterpret_cast<int*>(mcutil::cache_univ + INTEGRATE_SHARED_MEM_OFFSET + 2 * blockDim.x);

                // cumulative sum
                float prob_cumul = 0.f;
                for (int i = 0; i < CHANNEL::CHANNEL_2N; ++i) {
                    prob_cumul += smem->channel_prob[i];
                }

                // select branch
                int channel = CHANNEL::CHANNEL_UNKNWON;
                if (prob_cumul <= 0.f) {
                    flags.base.fmask |= FLAGS::FLAG_CHANNEL_UBREAKUP;
                    cache_channel[threadIdx.x] = channel;
                    return flags;
                }

                prob_cumul *= 1.f - curand_uniform(&rand_state[threadIdx.x + blockDim.x * blockIdx.x]);
                prob_cumul *= 0.999f;  // FP error
                for (int i = CHANNEL::CHANNEL_PHOTON; i < CHANNEL::CHANNEL_2N; ++i) {
                    prob_cumul -= smem->channel_prob[i];
                    if (prob_cumul <= 0.f) {
                        channel = i;
                        flags.base.fmask |= FLAGS::FLAG_CHANNEL_FOUND;
                        mcutil::cache_univ[INTEGRATE_SHARED_MEM_OFFSET + threadIdx.x + 3 * blockDim.x] 
                            = smem->channel_prob_max[channel];
                        break;
                    }
                }

                if (!(flags.base.fmask & FLAGS::FLAG_CHANNEL_FOUND))
                    channel = CHANNEL::CHANNEL_UNKNWON;

                cache_channel[threadIdx.x] = channel;
                return flags;
            }


            __device__ float emitParticleEnergy(curandState* state, CHANNEL channel) {

                // shared memory, energy sampling
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

                uchar4* cache_zaev = reinterpret_cast<uchar4*>(mcutil::cache_univ + CUDA_WARP_SIZE);

                float mass     =
                    mcutil::cache_univ[CUDA_WARP_SIZE +     blockDim.x + threadIdx.x] +
                    mcutil::cache_univ[CUDA_WARP_SIZE + 3 * blockDim.x + threadIdx.x];
                float res_mass = mass_table[cache_zaev[threadIdx.x].x].get(cache_zaev[threadIdx.x].y);

                float eke_max = 0.5f * ((mass - res_mass) * (mass + res_mass) + PROJ_M2[channel]) / mass - PROJ_M[channel];
                float elim    = coulombBarrier(channel, cache_zaev[threadIdx.x].x, cache_zaev[threadIdx.x].y, 0.f);

                mcutil::cache_univ[CUDA_WARP_SIZE + 5 * blockDim.x + threadIdx.x] = elim;  // Coulomb barrier

                // Chatterjee parameters
                setChatterjeeParameters(channel, elim, cache_zaev[threadIdx.x].y);

                elim *= COULOMB_RATIO;
                
                if (mass <= res_mass + PROJ_M[channel] + elim)
                    return 0.f;

                float eke_min = 0.f;
                if (elim > 0.f) {
                    float res_m = mass - PROJ_M[channel] - elim;
                    eke_min = fmaxf(0.f, 0.5f * ((mass - res_m) * (mass + res_m) + PROJ_M2[channel]) / mass - PROJ_M[channel]);
                }

                // calculate the nuclear level density and pairing correction
                int   a      = cache_zaev[threadIdx.x].y + cache_zaev[threadIdx.x].w;
                int   z      = cache_zaev[threadIdx.x].x + cache_zaev[threadIdx.x].z;
                float a0     = getLevelDensityParameter(a);
                float a1     = getLevelDensityParameter(cache_zaev[threadIdx.x].y);
                float delta0 = fission::pairingCorrection(z, a - z);
                float delta1 = fission::pairingCorrection(cache_zaev[threadIdx.x].x, cache_zaev[threadIdx.x].y - cache_zaev[threadIdx.x].x);

                // rejection unity
                float x, w;
                float wmax = mcutil::cache_univ[CUDA_WARP_SIZE + 4 * blockDim.x + threadIdx.x];
                do {
                    x = (eke_max - eke_min) * (1.f - curand_uniform(state));
                    w = computeProbability(channel, x + eke_min, a0, a1, delta0, delta1);
                } while (wmax * curand_uniform(state) > w);

                // return x + eke_min;
                return cache_zaev[threadIdx.x].y > 4 ? x + eke_min : eke_max;
            }


            __device__ void emitParticle(curandState* state, CHANNEL channel) {
                uchar4* cache_zaev = reinterpret_cast<uchar4*>(mcutil::cache_univ + CUDA_WARP_SIZE);

                // update ZA of system
                uchar4  zaev;
                zaev.z = (unsigned char)PROJ_Z[channel];
                zaev.w = (unsigned char)PROJ_A[channel];
                zaev.x = cache_zaev[threadIdx.x].x - zaev.z;
                zaev.y = cache_zaev[threadIdx.x].y - zaev.w;

                assert(cache_zaev[threadIdx.x].x >= zaev.z);
                assert(cache_zaev[threadIdx.x].y >= zaev.w);

                // update cache
                cache_zaev[threadIdx.x] = zaev;

                // remnant mass
                mcutil::cache_univ[CUDA_WARP_SIZE + 2 * blockDim.x + threadIdx.x] = mass_table[zaev.x].get(zaev.y);

                // momentum of emitted particle
                float eke_emit = emitParticleEnergy(state, channel);

                float mass_emit = PROJ_M[channel];
                float momentum  = eke_emit * (eke_emit + 2.f * mass_emit);  // now it is the square of norm

                // calculate the excitation energy of primary remnant (use double to avoid FP error) 
                float mass_rem = mass_table[zaev.x].get(zaev.y);
                float exc_rem  =
                    + mcutil::cache_univ[CUDA_WARP_SIZE +     blockDim.x + threadIdx.x]  // primary mass
                    + mcutil::cache_univ[CUDA_WARP_SIZE + 3 * blockDim.x + threadIdx.x]  // primary excitation
                    - mass_emit   // emitted mass
                    - eke_emit;   // kinetic energy of emitted particle
                exc_rem = sqrt((exc_rem * exc_rem - momentum)) - mass_rem;
                assert(exc_rem > -0.1f);
                exc_rem = fmaxf(exc_rem, 0.f);

                // update cache
                mcutil::cache_univ[CUDA_WARP_SIZE + 1 * blockDim.x + threadIdx.x] = mass_rem;
                mcutil::cache_univ[CUDA_WARP_SIZE + 2 * blockDim.x + threadIdx.x] = mass_emit;
                mcutil::cache_univ[CUDA_WARP_SIZE + 3 * blockDim.x + threadIdx.x] = exc_rem;
                mcutil::cache_univ[CUDA_WARP_SIZE + 4 * blockDim.x + threadIdx.x] = 0.f;

                // random isotropic direction
                float cost, sint;
                float cosp, sinp;
                float angle;

                // now momentum is norm
                momentum = sqrtf(momentum);

                // polar
                cost  = 1.f - 2.f * curand_uniform(state);
                sint  = sqrtf(fmaxf(0.f, 1.f - cost * cost));
                // azimuthal
                angle = constants::FP32_TWO_PI * curand_uniform(state);
                __sincosf(angle, &sinp, &cosp);

                mcutil::cache_univ[CUDA_WARP_SIZE + 5 * blockDim.x + threadIdx.x] = momentum * sint * cosp;  // X
                mcutil::cache_univ[CUDA_WARP_SIZE + 6 * blockDim.x + threadIdx.x] = momentum * sint * sinp;  // Y
                mcutil::cache_univ[CUDA_WARP_SIZE + 7 * blockDim.x + threadIdx.x] = momentum * cost;         // Z

                // Momentum of primary remnant is (-px, -py, -pz)
            }


        }
    }


}