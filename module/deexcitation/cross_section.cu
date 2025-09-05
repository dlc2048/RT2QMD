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
 * @file    module/deexcitation/cross_section.cu
 * @brief   Inverse cross-section for evaporation
 * @author  CM Lee
 * @date    08/11/2025
 */

#include "cross_section.cuh"
#include "channel_fission.cuh"


namespace deexcitation {


    namespace Chatterjee {


        __device__ void setSharedParameters(int channel, int z, int a, float mass, float exc_energy) {
            // shared memory
            IntegrateSharedMem* smem = (IntegrateSharedMem*)mcutil::cache_univ;

            int pz = PROJ_Z[channel];  // emission Z
            int pa = PROJ_A[channel];  // emission A
            int rz = z - pz;   // remnant Z
            int ra = a - pa;   // remnant A

            smem->is_allowed = false;   // default false
            smem->prob       = 0.f;     // initialize probability
            smem->prob_max   = 0.f;
            smem->channel    = channel;
            smem->res_a      = ra;
            smem->res_a13    = powf((float)ra, constants::ONE_OVER_THREE);

            // allowed channel (za number)
            if (ra < pa || ra < rz || rz < 0 || (ra == pa && rz < pz) || (ra > 1 && (ra == rz || rz == 0)))
                return;

            // available kinetic energy
            mass += exc_energy;  // total mass energy
            float res_mass = mass_table[rz].get(ra);
            float elim     = coulombBarrier((CHANNEL)channel, rz, ra, 0.f);
            smem->cb = elim;

            elim *= COULOMB_RATIO;
            // energetically accepted
            if (mass <= res_mass + PROJ_M[channel] + elim)
                return;

            smem->emax     = 0.5f * ((mass - res_mass) * (mass + res_mass) + PROJ_M2[channel]) / mass - PROJ_M[channel];
            smem->res_mass = res_mass;
            smem->mass     = mass;
            smem->exc      = exc_energy;

            float emin = 0.f;
            if (elim > 0.f) {
                float res_m = mass - PROJ_M[channel] - elim;
                emin = fmaxf(0.f, 0.5f * ((mass - res_m) * (mass + res_m) + PROJ_M2[channel]) / mass - PROJ_M[channel]);
            }
            smem->emin = emin;

            if (smem->emax <= smem->emin)
                return;

            smem->a0     = getLevelDensityParameter(a);
            smem->a1     = getLevelDensityParameter(ra);
            smem->delta0 = fission::pairingCorrection(z,  a  - z);
            smem->delta1 = fission::pairingCorrection(rz, ra - rz);

            // set Chatterjee parameters
            float cb  = smem->cb;
            float a13 = smem->res_a13;
            float muu = Kalbach::powerParameter(ra, channel);
            if (channel == CHANNEL::CHANNEL_NEUTRON) {
                smem->landa = CJXS_PARAM[3][channel] / a13 + CJXS_PARAM[4][channel];
                smem->mu    = (CJXS_PARAM[5][channel] + CJXS_PARAM[6][channel] * a13) * a13;
                smem->nu    = fabsf((CJXS_PARAM[7][channel] * ra + CJXS_PARAM[8][channel] * a13) 
                    * a13 + CJXS_PARAM[9][channel]);
            }
            else {
                smem->p     = CJXS_PARAM[0][channel] + (CJXS_PARAM[1][channel] + CJXS_PARAM[2][channel] / cb) / cb;
                smem->landa = CJXS_PARAM[3][channel] * ra + CJXS_PARAM[4][channel];
                smem->mu    = CJXS_PARAM[5][channel] * muu;
                smem->nu    = muu * (CJXS_PARAM[7][channel] +(CJXS_PARAM[8][channel] + CJXS_PARAM[9][channel] * cb) * cb);
                smem->q     = smem->landa - smem->nu / cb / cb - 2.f * smem->p * cb;
                smem->r     = smem->mu + 2.f * smem->nu / cb + smem->p * cb * cb;
            }

            // binning
            float edelta = channel == CHANNEL::CHANNEL_NEUTRON ? EDELTA_NEUTRON : EDELTA_CHARGED;
            float xbin   = (smem->emax - smem->emin) / edelta;
            int   ibin   = max((int)xbin, 4);
            int   nbin   = ibin * 5;

            // blockdim, not using smart binning
            int   iter         = (nbin - 1) / blockDim.x + 1;
            smem->int_iter     = iter;
            smem->edelta       = (smem->emax - smem->emin) / (float)(iter * blockDim.x - 1);
            smem->int_iter_2nd = CUDA_WARP_SIZE / (blockDim.x / CUDA_WARP_SIZE);

            smem->is_allowed = true;
            return;
        }


        __device__ void setParameters(int channel, float cb, int res_a) {

            // cache_univ[32 + 6 * blockDim.x:32 +  7 * blockDim.x] -> Chatterjee p
            // cache_univ[32 + 7 * blockDim.x:32 +  8 * blockDim.x] -> Chatterjee landa
            // cache_univ[32 + 8 * blockDim.x:32 +  9 * blockDim.x] -> Chatterjee mu
            // cache_univ[32 + 9 * blockDim.x:32 + 10 * blockDim.x] -> Chatterjee nu

            float res_a13 = powf((float)res_a, constants::ONE_OVER_THREE);
            float muu     = Kalbach::powerParameter(res_a, channel);

            float p     = CJXS_PARAM[0][channel] + (CJXS_PARAM[1][channel] + CJXS_PARAM[2][channel] / cb) / cb;
            float landa = CJXS_PARAM[4][channel];
            float mu    = CJXS_PARAM[5][channel];
            float nu;
            if (channel == CHANNEL::CHANNEL_NEUTRON) {
                landa += CJXS_PARAM[3][channel] / res_a13;
                mu     = (mu + CJXS_PARAM[6][channel] * res_a13) * res_a13;
                nu     = fabsf((CJXS_PARAM[7][channel] * (float)res_a + CJXS_PARAM[8][channel] * res_a13)
                    * res_a13 + CJXS_PARAM[9][channel]);
            }
            else {
                landa += CJXS_PARAM[3][channel] * (float)res_a;
                mu     = mu * muu;
                nu     = muu * (CJXS_PARAM[7][channel] + (CJXS_PARAM[8][channel] + CJXS_PARAM[9][channel] * cb) * cb);
            }
            mcutil::cache_univ[CUDA_WARP_SIZE + 6 * blockDim.x + threadIdx.x] = p;
            mcutil::cache_univ[CUDA_WARP_SIZE + 7 * blockDim.x + threadIdx.x] = landa;
            mcutil::cache_univ[CUDA_WARP_SIZE + 8 * blockDim.x + threadIdx.x] = mu;
            mcutil::cache_univ[CUDA_WARP_SIZE + 9 * blockDim.x + threadIdx.x] = nu;
        }


        __device__ float crossSectionShared(float k) {
            // shared memory
            IntegrateSharedMem* smem = (IntegrateSharedMem*)mcutil::cache_univ;

            k = fminf(k, MAX_ENERGY_CJXS);

            float sig, ji;
            if (smem->channel == CHANNEL::CHANNEL_NEUTRON)
                sig = smem->landa * k + smem->mu + smem->nu / k;
            else {
                ji = fmaxf(k, smem->cb);
                if (k < smem->cb)
                    sig = (smem->p * k + smem->q) * k + smem->r;
                else
                    sig = smem->p * (k - ji) * (k - ji) + smem->landa * k + smem->mu + smem->nu * (2.f - k / ji) / ji;
            }
            return sig;
        }


        __device__ float crossSection(int channel, float k) {

            // cache_univ[32 + 5 * blockDim.x:32 +  6 * blockDim.x] -> coulomb barrier [MeV]
            // cache_univ[32 + 6 * blockDim.x:32 +  7 * blockDim.x] -> Chatterjee p
            // cache_univ[32 + 7 * blockDim.x:32 +  8 * blockDim.x] -> Chatterjee landa
            // cache_univ[32 + 8 * blockDim.x:32 +  9 * blockDim.x] -> Chatterjee mu
            // cache_univ[32 + 9 * blockDim.x:32 + 10 * blockDim.x] -> Chatterjee nu

            float cb    = mcutil::cache_univ[CUDA_WARP_SIZE + 5 * blockDim.x + threadIdx.x];
            float p     = mcutil::cache_univ[CUDA_WARP_SIZE + 6 * blockDim.x + threadIdx.x];
            float landa = mcutil::cache_univ[CUDA_WARP_SIZE + 7 * blockDim.x + threadIdx.x];
            float mu    = mcutil::cache_univ[CUDA_WARP_SIZE + 8 * blockDim.x + threadIdx.x];
            float nu    = mcutil::cache_univ[CUDA_WARP_SIZE + 9 * blockDim.x + threadIdx.x];

            k = fminf(k, MAX_ENERGY_CJXS);

            float q    = landa - nu / cb / cb - 2.f * p * cb;
            float r    = mu + 2.f * nu / cb + p * cb * cb;
            float ji   = fmaxf(k, cb);

            float sig;
            if (channel == CHANNEL::CHANNEL_NEUTRON) {
                sig = landa * k + mu + nu / k;
            }
            else {
                if (k < cb)
                    sig = (p * k + q) * k + r;
                else
                    sig = p * (k - ji) * (k - ji) + landa * k + mu + nu * (2.f - k / ji) / ji;
            }
            return fmaxf(sig, 0.f);
        }


    }


    namespace Kalbach {


        __device__ void setSharedParameters(int channel, int z, int a, float mass, float exc_energy) {
            // shared memory
            IntegrateSharedMem* smem = (IntegrateSharedMem*)mcutil::cache_univ;

            int pz = PROJ_Z[channel];  // emission Z
            int pa = PROJ_A[channel];  // emission A
            int rz = z - pz;   // remnant Z
            int ra = a - pa;   // remnant A

            smem->is_allowed = false;   // default false
            smem->prob       = 0.f;     // initialize probability
            smem->prob_max   = 0.f;
            smem->channel    = channel;
            smem->res_a      = ra;
            smem->res_a13    = powf((float)ra, constants::ONE_OVER_THREE);

            // allowed channel (za number)
            if (ra < pa || ra < rz || rz < 0 || (ra == pa && rz < pz) || (ra > 1 && (ra == rz || rz == 0)))
                return;

            // available kinetic energy
            mass += exc_energy;  // total mass energy
            float res_mass = mass_table[rz].get(ra);
            float elim     = coulombBarrier((CHANNEL)channel, rz, ra, 0.f);

            smem->cb     = channel == CHANNEL::CHANNEL_NEUTRON ? fminf(4.f, 100.f / ra) : elim;
            smem->signor = 1.f;

            elim *= COULOMB_RATIO;
            // energetically accepted
            if (mass <= res_mass + PROJ_M[channel] + elim)
                return;

            smem->emax     = 0.5f * ((mass - res_mass) * (mass + res_mass) + PROJ_M2[channel]) / mass - PROJ_M[channel];
            smem->res_mass = res_mass;
            smem->mass     = mass;
            smem->exc      = exc_energy;

            float emin = 0.f;
            if (elim > 0.f) {
                float res_m = mass - PROJ_M[channel] - elim;
                emin = fmaxf(0.f, 0.5f * ((mass - res_m) * (mass + res_m) + PROJ_M2[channel]) / mass - PROJ_M[channel]);
            }
            smem->emin = emin;

            if (smem->emax <= smem->emin)
                return;

            smem->a0     = getLevelDensityParameter(a);
            smem->a1     = getLevelDensityParameter(ra);
            smem->delta0 = fission::pairingCorrection(z,  a  - z);
            smem->delta1 = fission::pairingCorrection(rz, ra - rz);

            // set Kalbach parameters
            float cb   = smem->cb;
            float a13  = smem->res_a13;
            float muu  = powerParameter(ra, channel);
            smem->p    = KMXS_PARAM[0][channel];
            smem->geom = 1.23f * a13 + KMXS_PARAM[10][channel];
            if (channel == CHANNEL::CHANNEL_NEUTRON) {
                if (ra < 40)
                    smem->signor = 0.7f + (float)ra * 0.0075f;
                else if (ra > 210)
                    smem->signor = 1.f + (float)(ra - 210) * 0.004f;

                smem->lambda = KMXS_PARAM[3][channel] / a13 + KMXS_PARAM[4][channel];
                smem->mu     = (KMXS_PARAM[5][channel] + KMXS_PARAM[6][channel] * a13) * a13;
                smem->nu     = fabsf((KMXS_PARAM[7][channel] * ra + KMXS_PARAM[8][channel] * a13)
                    * a13 + KMXS_PARAM[9][channel]);
            }
            else {  // charged
                if (channel == CHANNEL::CHANNEL_PROTON) {
                    if (ra <= 60)
                        smem->signor = 0.92f;
                    else if (ra < 100)
                        smem->signor = 0.8f + (float)ra * 0.002f;
                }
                smem->p      = smem->p + (KMXS_PARAM[1][channel] + KMXS_PARAM[2][channel] / cb) / cb;
                smem->lambda = KMXS_PARAM[3][channel] * (float)ra + KMXS_PARAM[4][channel];
                smem->mu     = KMXS_PARAM[5][channel] * muu;
                smem->nu     = muu * (KMXS_PARAM[7][channel] + (KMXS_PARAM[8][channel] + KMXS_PARAM[9][channel] * cb) * cb);
            }
            smem->a = smem->lambda - smem->nu / cb / cb - 2.f * smem->p * cb;
            smem->b = smem->mu + 2.f * smem->nu / cb + smem->p * cb * cb;
            float det = smem->a * smem->a - 4.f * smem->p * smem->b;
            smem->ecut = det > 0.f
                ? (sqrtf(det) - smem->a) / (2.f * smem->p)
                : -smem->a / (2.f * smem->p);

            // binning
            float edelta = channel == CHANNEL::CHANNEL_NEUTRON ? EDELTA_NEUTRON : EDELTA_CHARGED;
            float xbin   = (smem->emax - smem->emin) / edelta;
            int   ibin   = max((int)xbin, 4);
            int   nbin   = ibin * 5;

            // blockdim, not using smart binning
            int   iter         = (nbin - 1) / blockDim.x + 1;
            smem->int_iter     = iter;
            smem->edelta       = (smem->emax - smem->emin) / (float)(iter * blockDim.x - 1);
            smem->int_iter_2nd = CUDA_WARP_SIZE / (blockDim.x / CUDA_WARP_SIZE);

            smem->is_allowed = true;
            return;
        }


        __device__ void setParameters(int channel, int res_a) {

            // cache_univ[32 + 5  * blockDim.x:32 +  6 * blockDim.x] -> coulomb barrier [MeV]
            // cache_univ[32 + 6  * blockDim.x:32 +  7 * blockDim.x] -> Kalbach p
            // cache_univ[32 + 7  * blockDim.x:32 +  8 * blockDim.x] -> Kalbach lambda
            // cache_univ[32 + 8  * blockDim.x:32 +  9 * blockDim.x] -> Kalbach mu
            // cache_univ[32 + 9  * blockDim.x:32 + 10 * blockDim.x] -> Kalbach nu
            // cache_univ[32 + 10 * blockDim.x:32 + 11 * blockDim.x] -> Kalbach geom

            float res_a13 = powf((float)res_a, constants::ONE_OVER_THREE);
            float muu     = Kalbach::powerParameter(res_a, channel);
            float cb      = mcutil::cache_univ[CUDA_WARP_SIZE + 5 * blockDim.x + threadIdx.x];

            // geom
            mcutil::cache_univ[CUDA_WARP_SIZE + 10 * blockDim.x + threadIdx.x] = 1.23f * res_a13 + KMXS_PARAM[10][channel];

            float p      = KMXS_PARAM[0][channel] + (KMXS_PARAM[1][channel] + KMXS_PARAM[2][channel] / cb) / cb;
            float lambda = channel == CHANNEL::CHANNEL_NEUTRON ? 1.f / res_a13 : res_a;
            lambda = KMXS_PARAM[3][channel] * lambda + KMXS_PARAM[4][channel];

            float mu     = KMXS_PARAM[5][channel];
            float nu     = KMXS_PARAM[7][channel];

            if (channel == CHANNEL::CHANNEL_NEUTRON) {
                mu = (mu + KMXS_PARAM[6][channel] * res_a13) * res_a13;
                nu = fabsf((nu * res_a + KMXS_PARAM[8][channel] * res_a13)
                    * res_a13 + KMXS_PARAM[9][channel]);
            }
            else {
                mu = muu * mu;
                nu = muu * (nu + (KMXS_PARAM[8][channel] + KMXS_PARAM[9][channel] * cb) * cb);
            }
            
            mcutil::cache_univ[CUDA_WARP_SIZE + 6 * blockDim.x + threadIdx.x] = p;
            mcutil::cache_univ[CUDA_WARP_SIZE + 7 * blockDim.x + threadIdx.x] = lambda;
            mcutil::cache_univ[CUDA_WARP_SIZE + 8 * blockDim.x + threadIdx.x] = mu;
            mcutil::cache_univ[CUDA_WARP_SIZE + 9 * blockDim.x + threadIdx.x] = nu;
        }


        __device__ float crossSectionShared(float k) {
            // shared memory
            IntegrateSharedMem* smem = (IntegrateSharedMem*)mcutil::cache_univ;

            float temp1, temp2;
            float sig  = 0.f;
            float ec   = smem->cb;
            float elab = k * (float)(smem->res_a + PROJ_A[smem->channel]) / (float)smem->res_a;
            if (elab < ec) {
                if (smem->channel == CHANNEL::CHANNEL_NEUTRON) {
                    sig = (smem->lambda * ec + smem->mu + smem->nu / ec) * smem->signor * sqrtf(elab / ec);
                }
                else if (elab >= smem->ecut) {
                    sig = ((smem->p * elab + smem->a) * elab + smem->b) * smem->signor;

                    if (smem->channel == CHANNEL::CHANNEL_PROTON) {
                        temp1 = fminf(3.15f, ec * 0.5f);
                        temp1 = (ec - elab - temp1) * 3.15f / (0.7f * temp1);
                        sig  /= 1.f + expf(temp1);
                    }

                }
            }
            else {
                temp1 = 32.f;
                temp2 = 1.f;

                if (smem->channel != CHANNEL::CHANNEL_NEUTRON) {
                    temp1 = 0.f;
                    temp2 = smem->nu / smem->lambda;
                    temp2 = fminf(temp2, SPILL);
                    if (temp2 >= FLOW) {
                        temp1 = sqrtf(temp2);
                        temp1 = smem->channel == CHANNEL::CHANNEL_PROTON 
                            ? temp1 + 7.f 
                            : temp1 * 1.2f;
                    }
                }

                sig = (smem->lambda * elab + smem->mu + smem->nu / elab) * smem->signor;
                if (temp2 >= FLOW && elab >= temp1) {
                    temp1 = sqrtf((float)PROJ_A[smem->channel] * k);
                    temp1 = smem->geom + 4.573f / temp1;
                    temp1 = constants::FP32_TEN_PI * temp1 * temp1;
                    sig   = fmaxf(sig, temp1);
                }
            }
            return fmaxf(0.f, sig);
        }


        __device__ float crossSection(int channel, float k, float signor) {

            // cache_univ[32                  :32 +      blockDim.x] -> ZA number of primary remnant & secondary particle
            // cache_univ[32 + 5  * blockDim.x:32 +  6 * blockDim.x] -> coulomb barrier [MeV]
            // cache_univ[32 + 6  * blockDim.x:32 +  7 * blockDim.x] -> Kalbach p
            // cache_univ[32 + 7  * blockDim.x:32 +  8 * blockDim.x] -> Kalbach lambda
            // cache_univ[32 + 8  * blockDim.x:32 +  9 * blockDim.x] -> Kalbach mu
            // cache_univ[32 + 9  * blockDim.x:32 + 10 * blockDim.x] -> Kalbach nu
            // cache_univ[32 + 10 * blockDim.x:32 + 11 * blockDim.x] -> Kalbach geom

            uchar4* cache_zaev = reinterpret_cast<uchar4*>(mcutil::cache_univ + CUDA_WARP_SIZE);

            float ec     = mcutil::cache_univ[CUDA_WARP_SIZE + 5  * blockDim.x + threadIdx.x];
            float p      = mcutil::cache_univ[CUDA_WARP_SIZE + 6  * blockDim.x + threadIdx.x];
            float lambda = mcutil::cache_univ[CUDA_WARP_SIZE + 7  * blockDim.x + threadIdx.x];
            float mu     = mcutil::cache_univ[CUDA_WARP_SIZE + 8  * blockDim.x + threadIdx.x];
            float nu     = mcutil::cache_univ[CUDA_WARP_SIZE + 9  * blockDim.x + threadIdx.x];
            float geom   = mcutil::cache_univ[CUDA_WARP_SIZE + 10 * blockDim.x + threadIdx.x];

            float elab   = (float)cache_zaev[threadIdx.x].y;
            elab = k * (elab + (float)PROJ_A[channel]) / elab;

            float a    = lambda - nu / ec / ec - 2.f * p * ec;
            float b    = mu + 2.f * nu / ec + p * ec * ec;
            float det  = a * a - 4.f * p * b;
            float ecut = det > 0.f
                ? (sqrtf(det) - a) / (2.f * p)
                : -a / (2.f * p);

            float sig;
            float temp1, temp2;
            if (elab < ec) {
                if (channel == CHANNEL::CHANNEL_NEUTRON) {
                    sig = (lambda * ec + mu + nu / ec) * signor * sqrtf(elab / ec);
                }
                else if (elab >= ecut) {
                    sig = ((p * elab + a) * elab + b) * signor;

                    if (channel == CHANNEL::CHANNEL_PROTON) {
                        temp1 = fminf(3.15f, ec * 0.5f);
                        temp1 = (ec - elab - temp1) * 3.15f / (0.7f * temp1);
                        sig /= 1.f + expf(temp1);
                    }

                }
            }
            else {
                temp1 = 32.f;
                temp2 = 1.f;

                if (channel != CHANNEL::CHANNEL_NEUTRON) {
                    temp1 = 0.f;
                    temp2 = nu / lambda;
                    temp2 = fminf(temp2, SPILL);
                    if (temp2 >= FLOW) {
                        temp1 = sqrtf(temp2);
                        temp1 = channel == CHANNEL::CHANNEL_PROTON 
                            ? temp1 + 7.f 
                            : temp1 * 1.2f;
                    }
                }

                sig = (lambda * elab + mu + nu / elab) * signor;
                if (temp2 >= FLOW && elab >= temp1) {
                    temp1 = sqrtf((float)PROJ_A[channel] * k);
                    temp1 = geom + 4.573f / temp1;
                    temp1 = constants::FP32_TEN_PI * temp1 * temp1;
                    sig   = fmaxf(sig, temp1);
                }
            }
            return fmaxf(0.f, sig);
        }


    }


}