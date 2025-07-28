/**
 * @file    module/hadron/xs_dev.cuh
 * @brief   Nucleus-Nucleus cross section, optimized for device (G4ComponentGGNuclNuclXsc)
 * @author  CM Lee
 * @date    06/25/2024
 */

#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include <stdio.h>

#include "nucleus.cuh"


namespace Hadron {


    /**
    * @brief Compute coulomb barrier ratio. Two particles cannot collide if ratio is 0
    * @param zs  Charge number of projectile x charge number of target
    * @param mp  Mass of projectile [GeV/c^2]
    * @param mt  Mass of target [GeV/c^2]
    * @param eke Kinetic energy of projectile in lab system [GeV]
    * @param md  Minimum distance for touching, radius of target + radius of projectile [fm]
    *
    * @return Coulomb barrier ratio
    */
    __inline__ __device__ float coulombBarrier(int zs, float mp, float mt, float eke, float md) {
        float tot_mass = mp + mt;
        float tot_tcm  = sqrtf(tot_mass * tot_mass + 2.f * eke * mt) - tot_mass;
        float bc = constants::FP32_FSC_HBARC * (float)zs * 0.5f / md;

        return tot_tcm <= bc ? 0.f : 1.f - bc / tot_tcm;
    }


    /**
    * @brief Calculate the scattering XS of NN interaction
    * @param tp  Type of projectile (proton if true)
    * @param tt  Type of target (proton if true)
    * @param eke Kinetic energy of projectile in lab system [GeV]
    *
    * @return Cross-section of NN interaction [millibarn]
    */
    __inline__ __device__ float xsNucleonNucleon(bool tp, bool tt, float eke) {
        float pmass = tp ? constants::MASS_PROTON_GEV : constants::MASS_NEUTRON_GEV;
        float plab  = sqrtf(eke * (eke + 2.f * pmass));
        float xs    = 0.f;

        if (tp == tt) {  // nn or pp (available up to 1 GeV)
            if (plab < 0.73f)
                xs = 23.f + 50.f * powf(logf(0.73f / plab), 3.5f);
            else if (plab < 1.05f) {
                xs = logf(plab / 0.73f);
                xs = 23.f + 40.f * xs * xs;
            }
            else
                xs = 39.f + 75.f * (plab - 1.2f) / (plab * plab * plab + 0.15f);
        }
        else {  // np or pn (available up to 1 GeV)
            if (plab < 0.02f)
                xs  = 4.1e3f + 30.f * expf(logf(logf(1.3f / plab)) * 3.6f);
            else if (plab < 0.8f) {
                xs  = logf(plab / 1.3f);
                xs *= xs;
                xs *= xs;  // log(plab / 1.3)^4
                xs  = 33.f + 30.f * xs;
            }
            else if (plab < 1.4f) {
                xs = logf(plab / 0.95f);
                xs = 33.f + 30.f * xs * xs;
            }
            else
                xs = 33.3f + 20.8f * (plab * plab - 1.35f) / (powf(plab, 2.5f) + 0.95f);
        }

        if (tp && tt)  // proton-proton Coulomb barrier
            xs *= coulombBarrier(1, constants::MASS_PROTON_GEV, constants::MASS_PROTON_GEV, eke, 1.79f);
        return xs;
    }


    /**
    * @brief Calculate the inelastic XS of nuclei-nuclei collision
    * @param zap ZA number of projectile
    * @param zat ZA number of target
    * @param cb  Coulomb barrier
    * @param spp Proton-proton scattering XS [millibarn]
    * @param spn Proton-neutron scattering XS [millibarn]
    * @param ns  Collisional area [mb]
    *
    * @return Cross-section of nuclei-nuclei collision [millibarn]
    */
    __inline__ __device__ float xsNucleiNuclei(uchar2 zap, uchar2 zat, float cb, float spp, float spn, float ns) {
        int np = (int)(zap.y - zap.x);
        int nt = (int)(zat.y - zat.x);
        int zp = (int)zap.x;
        int zt = (int)zat.x;

        float xs = 0.f;
        if (cb > 0.f) {
            xs  = (float)(zp * zt + np * nt) * spp;
            xs += (float)(zp * nt + np * zt) * spn;
            xs  = ns * logf(1.f + 2.4f * xs / ns) * cb / 2.4f;
        }
        return xs;
    }


    __inline__ __device__ float calculateNNAngularSlope(float plab, bool iso) {
        float b;
        if (iso) {
            if (plab <= 2.f) {
                plab = powf(plab, 8.f);
                return 5.5f * plab / (7.7f + plab);
            }
            else
                return (5.34f + 0.67f * (plab - 2.f));
        }
        else {
            if (plab < 0.8f) {
                b = (7.16f - 1.63f * plab);
                return b / (1.f + expf(-(plab - 0.45f) / 0.05f));
            }
            else if (plab < 1.1f)
                return (9.87f - 4.88f * plab);
            else
                return (3.68f + 0.76f * plab);
        }
        assert(false);
        return 0.f;
    }


}