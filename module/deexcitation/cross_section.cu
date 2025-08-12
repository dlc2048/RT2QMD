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


    __device__ void setChatterjeeSharedParameters(int channel, int z, int a, float mass, float exc_energy) {
        // shared memory
        Chatterjee::IntegrateSharedMem* smem = (Chatterjee::IntegrateSharedMem*)mcutil::cache_univ;

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
        smem->muu = powerParameter(ra, channel);
        if (channel == CHANNEL::CHANNEL_NEUTRON) {
            smem->landa = CJXS_PARAM[3][channel] / a13 + CJXS_PARAM[4][channel];
            smem->mu    = (CJXS_PARAM[5][channel] + CJXS_PARAM[6][channel] * a13) * a13;
            smem->nu    = fabsf((CJXS_PARAM[7][channel] * ra + CJXS_PARAM[8][channel] * a13) 
                * a13 + CJXS_PARAM[9][channel]);
        }
        else {
            smem->p     = CJXS_PARAM[0][channel] + (CJXS_PARAM[1][channel] + CJXS_PARAM[2][channel] / cb) / cb;
            smem->landa = CJXS_PARAM[3][channel] * ra + CJXS_PARAM[4][channel];
            smem->mu    = CJXS_PARAM[5][channel] * smem->muu;
            smem->nu    = smem->muu * (CJXS_PARAM[7][channel] +(CJXS_PARAM[8][channel] + CJXS_PARAM[9][channel] * cb) * cb);
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


    __device__ void setChatterjeeParameters(int channel, float cb, int res_a) {

        // cache_univ[32 + 6 * blockDim.x:32 +  7 * blockDim.x] -> Chatterjee p
        // cache_univ[32 + 7 * blockDim.x:32 +  8 * blockDim.x] -> Chatterjee landa
        // cache_univ[32 + 8 * blockDim.x:32 +  9 * blockDim.x] -> Chatterjee mu
        // cache_univ[32 + 9 * blockDim.x:32 + 10 * blockDim.x] -> Chatterjee nu

        float res_a13 = powf((float)res_a, constants::ONE_OVER_THREE);
        float muu     = powerParameter(res_a, channel);

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


    __device__ float crossSectionChatterjeeShared(float k) {
        // shared memory
        Chatterjee::IntegrateSharedMem* smem = (Chatterjee::IntegrateSharedMem*)mcutil::cache_univ;

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


    __device__ float crossSectionChatterjee(int channel, float k) {

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