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
 * @file    module/physics/physics.cpp
 * @brief   Methods related to physics
 * @author  CM Lee
 * @date    04/01/2024
 */


#include "physics.hpp"


namespace physics {

    double coulombCorrection(int z) {
        double asq, fcoulcp;
        asq = (constants::FSC * (double)z);
        asq *= asq;

        fcoulcp = 0.0083 - 0.002 * asq;
        fcoulcp = fcoulcp * asq - 0.0369;
        fcoulcp = fcoulcp * asq + 0.20206 + 1.e0 / (1.e0 + asq);

        return asq * fcoulcp;
    }


    double xsifp(int z) {
        double lradp = z > 4
            ? log(1194.0 * pow((double)z, -2.0 / 3.0))
            : constants::RADIATION_LOGARITHM_PRIM[z - 1];
        double lrad = z > 4
            ? log(184.15 * pow((double)z, -1.0 / 3.0))
            : constants::RADIATION_LOGARITHM[z - 1];
        return lradp / (lrad - coulombCorrection(z));
    }


    double DensityEffect::_GAS_DENSITY_THRES = 0.1;  // default 0.1 (g/cm3)


    DensityEffect::DensityEffect(
        double mean_ie,
        double density,
        double plasma_frequency
    ) :
        _m(3.e0),
        _c(-2.e0 * log(mean_ie / plasma_frequency) - 1),
        _a(0.e0),
        _x(make_double2(0, 0)) {
        if (density > DensityEffect::_GAS_DENSITY_THRES) {  // solid and liquids
            if (mean_ie < 1e-4) {  // low Z material
                _x.y = 2.0;
                _x.x = (_c > -3.681) ? 0.2 : -0.326 * _c - 1.0;
            }
            else {  // high Z material
                _x.y = 3.0;
                _x.x = (_c > -5.215) ? 0.2 : -0.326 * _c - 1.5;
            }
        }
        else {  // gas
            if (_c > -10)
                _x = make_double2(1.6, 4.0);
            else if (_c > -10.5)
                _x = make_double2(1.7, 4.0);
            else if (_c > -11.0)
                _x = make_double2(1.8, 4.0);
            else if (_c > -11.5)
                _x = make_double2(1.9, 4.0);
            else if (_c > -12.25)
                _x = make_double2(2.0, 4.0);
            else if (_c > -13.804)
                _x = make_double2(2.0, 5.0);
            else
                _x = make_double2(
                    -0.326 * _c - 2.5,
                    5.0
                );
        }
        _a = -_c - 2.e0 * log(10.e0) * _x.x;
        _a /= pow((_x.y - _x.x), 3.0);
    }


    double DensityEffect::get(double energy) {
        double x = energy * (double)constants::MASS_ELECTRON_I;
        double log10 = log(10.e0);
        x = sqrt(x * x - 1.e0);
        x = log(x) / log10;

        double delta;
        if (x < _x.x)
            delta = 0.e0;
        else if (x < _x.y) {
            delta = 2.e0 * log10 * x + _c;
            delta += _a * pow((_x.y - x), _m);
        }
        else {
            delta = 2.e0 * log10 * x + _c;
        }
        return delta;
    }


}