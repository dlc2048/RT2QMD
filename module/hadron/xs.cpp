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
 * @file    module/hadron/xs.cpp
 * @brief   Nucleus-Nucleus cross section (G4ComponentGGNuclNuclXsc)
 * @author  CM Lee
 * @date    06/11/2024
 */


#include "xs.hpp"


namespace Hadron {


    namespace Host {


        double coulombBarrier(int zp, int zt, double mp, double mt, double eke, double md) {
            double zs       = (double)zp * (double)zt;
            double tot_mass = mp + mt;
            double tot_tcm  = std::sqrt(tot_mass * tot_mass + 2.0 * eke * mt) - tot_mass;
            double bc       = constants::FSC * constants::HBARC * 1e10 * zs * 0.5 / md;

            return tot_tcm <= bc ? 0.0 : 1.0 - bc / tot_tcm;
        }


        double xsNucleonNucleon(bool tp, bool tt, double eke) {
            double pmass = tp ? constants::MASS_PROTON : constants::MASS_NEUTRON;
            pmass *= 1e-3;  // to GeV
            double plab  = std::sqrt(eke * (eke + 2.0 * pmass));
            double xs    = 0.0;

            if (tp == tt) {  // nn or pp (available up to 1 GeV)
                if (plab < 0.73)
                    xs = 23.0 + 50.0 * std::pow(std::log(0.73 / plab), 3.5);
                else if (plab < 1.05) {
                    xs = std::log(plab / 0.73);
                    xs = 23.0 + 40.0 * xs * xs;
                }
                else
                    xs = 39.0 + 75.0 * (plab - 1.2) / (plab * plab * plab + 0.15);
            }
            else {  // np or pn (available up to 1 GeV)
                if (plab < 0.02)
                    xs = 4.1e3 + 30.0 * std::exp(std::log(std::log(1.3 / plab)) * 3.6);
                else if (plab < 0.8) {
                    xs  = std::log(plab / 1.3);
                    xs *= xs;
                    xs *= xs;  // log(plab / 1.3)^4
                    xs  = 33.0 + 30.0 * xs;
                }
                else if (plab < 1.4) {
                    xs = std::log(plab / 0.95);
                    xs = 33.0 + 30.0 * xs * xs;
                }
                else
                    xs = 33.3 + 20.8 * (plab * plab - 1.35) / (std::pow(plab, 2.5) + 0.95);
            }

            if (tp && tt)  // proton-proton Coulomb barrier
                xs *= coulombBarrier(1, 1, constants::MASS_PROTON * 1e-3, constants::MASS_PROTON * 1e-3, eke, 1.79);

            return xs;
        }


        double2 xsNucleiNuclei(int zap, int zat, double eke, double mp, double mt) {
            int zp = physics::getZnumberFromZA(zap);
            int ap = physics::getAnumberFromZA(zap);
            int np = ap - zp;
            int zt = physics::getZnumberFromZA(zat);
            int at = physics::getAnumberFromZA(zat);
            int nt = at - zt;

            double xs = 0.0, ns;
            double rp = (double)Nucleus::nuclearRadius({ (unsigned char)zp, (unsigned char)ap });
            double rt = (double)Nucleus::nuclearRadius({ (unsigned char)zt, (unsigned char)at });
            double cb = coulombBarrier(zp, zt, mp, mt, eke, rp + rt);

            double total     = 0.0;
            double inelastic = 0.0;
            double elastic   = 0.0;
            if (cb > 0.0) {
                eke /= (double)ap;  // kinetic energy per nucleon
                xs   = (double)(zp * zt + np * nt) * xsNucleonNucleon(true, true, eke);
                xs  += (double)(zp * nt + zt * np) * xsNucleonNucleon(false, true, eke);
                ns   = constants::FP64_TWE_PI * (rp * rp + rt * rt);      // [fm^2] to [mb]

                total     = ns * std::log(1.0 + xs / ns) * cb;              // total XS
                inelastic = ns * std::log(1.0 + 2.4 * xs / ns) * cb / 2.4;  // inelastic XS
                elastic   = std::max(0.0, total - inelastic);
            }
            return { inelastic, elastic };
        }


    }


}