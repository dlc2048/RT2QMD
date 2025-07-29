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
 * @file    module/hadron/nucleus.cuh
 * @brief   Nuclear radius, density and mass (G4NuclearRadii, G4QMDNucleus)
 * @author  CM Lee
 * @date    02/14/2024
 */

#pragma once

#include <cuda_runtime.h>
#include <assert.h>

#include "physics/constants.cuh"

namespace Nucleus {


    /**
    * @brief Get the explicit radius for light nucleii
    * @param za Charge and mass number of nuclei
    *
    * @return Radius of nuclei [fm]
    */
    __inline__ __host__ __device__ float explicitNuclearRadius(uchar2 za) {
        float r = -1.f;
        switch (za.x) {
        case 0:  // neutron
            r = 0.895f;
            break;
        case 1:
            if (za.y == 1)
                r = 0.895f;
            else if (za.y == 2)
                r = 2.13f;
            else if (za.y == 3)
                r = 1.80f;
            break;
        case 2:
            if (za.y == 3)
                r = 1.96f;
            else if (za.y == 4)
                r = 1.68f;
            break;
        case 3:
            if (za.y == 7)
                r = 2.40f;
            break;
        case 4:
            if (za.y == 9)
                r = 2.51f;
            break;
        }
        return r;
    }


    /**
    * @brief Get the nuclear radius
    * @param za Charge and mass number of nuclei
    *
    * @return Radius of nuclei [fm]
    */
    __inline__ __host__ __device__ float nuclearRadius(uchar2 za) {
        float r = explicitNuclearRadius(za);
        float y = 1.1f;
        float x;
        if (r < 0.f) {
            x = za.y <= 50 ? 0.33333333f : 0.27f;
            x = powf((float)za.y, x);
            if (za.y <= 50) {
                if (za.y <= 15)
                    y = 1.26f;
                else if (za.y <= 20)
                    y = 1.19f;
                else if (za.y <= 30)
                    y = 1.12f;
                r = y * (x - 1.f / x);
            }
            else
                r = x;
        }
        return r;
    }


    /**
    * @brief Nuclear mass table, device side
    */
    typedef struct MassTable {
        float* mass;  //! @brief mass table
        int    z;     //! @brief Z number
        int    amin;  //! @brief Minimum A number
        int    amax;  //! @brief Maximum A number


        /**
        * @brief Get nuclear mass for given Z and A
        * @param a Atomic mass number
        *
        * @return Nuclear mass [MeV/c^2]
        */
        __inline__ __device__ float get(int a);


    } MassTable;


    /**
    * @brief Weizsaecker's semi-empirical mass formula
    * @param a   Mass number
    * @param z   Atomic number
    *
    * @return Nuclear mass [MeV/c^2]
    */
    __inline__ __host__ __device__ float getWeizsaeckerMass(const int a, const int z) {
        const int   np = (a - z) % 2;  // Neutron pairing
        const int   zp = z % 2;        // Proton pairing
        const float fa = (float)a;
        const float fz = (float)z;
        float binding =
            - 15.67f * fa                                         // Nuclear volume
            + 17.23f * powf(fa, 0.66666667f)                      // Surface energy
            + 93.15f * ((fa / 2.f - fz) * (fa / 2.f - fz)) / fa   // Asymmetry
            + 0.6984523f * fz * fz * powf(fa, -0.33333333f);      // Coulomb
        if (np == zp)
            binding += (float)(np + zp - 1) * 12.f / sqrtf(fa);   // Pairing
        return fz * constants::MASS_PROTON + (float)(a - z) * constants::MASS_NEUTRON + binding;
    }


    __device__ float MassTable::get(int a) {
        float mass = -1.f;
        if (a >= this->amin && a < this->amax)
            mass = this->mass[a - amin];

        if (mass < 0.f)
            mass = getWeizsaeckerMass(a, this->z);  
        return mass;
    }


    typedef struct LongLivedNucleiTable {
        int* offset;
        int* mass_number;


        __inline__ __device__ bool longLived(int z, int a);


    } LongLivedNucleiTable;


    __device__ bool LongLivedNucleiTable::longLived(int z, int a) {
        int pos = this->offset[z];
        bool long_lived = false;
        while (true) {
            int atable = this->mass_number[pos];
            if (a == atable) {
                long_lived = true;
                break;
            }
            else if (a < atable)
                break;

            pos++;
        }
        return long_lived;
    }


}   