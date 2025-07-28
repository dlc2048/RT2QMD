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
 * @file    mcutil/device/algorithm.cuh
 * @brief   Device side interpolation, sampling and integral algorithms
 * @author  CM Lee
 * @date    05/23/2023
 */


#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <assert.h>


namespace mcutil {


    /**
    * @brief Simple one dimensional alias table, device side
    */
    typedef struct DeviceAliasData {
        int    dim;    //! @brief Length of alias table
        int*   alias;  //! @brief Alias table index
        float* prob;   //! @brief Alias table probability


        __host__ DeviceAliasData();


        __host__ void free();


        /**
        * @brief Sample alias index from table
        * @param state XORWOW PRNG state
        *
        * @return Alias index [0,dim)
        */
        __inline__ __device__ int sample(curandState* state);


    } DeviceAliasData;

     
    /**
    * @brief Simple one dimensional alias table with index map, device side
    */
    typedef struct DeviceAliasDataMap {
        DeviceAliasData table;
        int*            map;


        __host__ DeviceAliasDataMap();


        __host__ void free();


        /**
        * @brief Sample alias index map from table
        * @param state XORWOW PRNG state
        *
        * @return Alias index [0,dim)
        */
        __inline__ __device__ int sample(curandState* state);


    } DeviceAliasDataMap;


    /**
    * @brief Two dimensional EGS-style alias table, device side
    */
    typedef struct DeviceAliasDataEGS {
        int2   dim;    //! @brief Alias table dimension {group, domain}
        float* xdata;  //! @brief Alias table domain
        float* fdata;  //! @brief Alias table value
        int*   idata;  //! @brief Alias table index
        float* wdata;  //! @brief Alias probability


        __host__ DeviceAliasDataEGS();


        __host__ void free();


        /**
        * @brief Sample alias index from table.
        *        In case of all group share same domain
        * @param state XORWOW PRNG state
        * @param g     Sampling group
        *
        * @return Function range of alias table
        */
        __inline__ __device__ float sample(curandState* state, int g);


        /**
        * @brief Sample alias index from table.
        *        In case of all group have their own domain
        * @param state XORWOW PRNG state
        * @param g     Sampling group
        *
        * @return Function range of alias table
        */
        __inline__ __device__ float sample2D(curandState* state, int g);


    } DeviceAliasDataEGS;


    /**
    * @brief Ray-box intersection test
    * @param vmin Bottom-left corner of box {xmin, ymin, zmin}
    * @param vmax Top-right corner of box {xmax, ymax, zmax}
    * @param rpos Initial ray position {x, y, z}
    * @param rdir Initial ray direction {u, v, w}
    *
    * @return Distance between ray and box if positive, ray miss elsewhere
    */
    __inline__ __device__ float rayBoxDistance(float3 vmin, float3 vmax, float3 rpos, float3 rdir);


    /**
    * @brief Ray-sphere intersection test, note that the center of sphere is origin (0,0,0)
    * @param rpos   Initial ray position {x, y, z}
    * @param rdir   Initial ray direction {u, v, w}
    * @param radius Radius of sphere
    */
    __inline__ __device__ bool  raySphereIntersection(float3 rpos, float3 rdir, float radius);


    __inline__ __device__ float gammaDistribution(curandState* state, float a, float b);


#ifdef __CUDACC__


    __inline__ __device__ int DeviceAliasData::sample(curandState* state) {
        float rand, aj;
        int j;

        rand = curand_uniform(state);

        aj = rand * (float)(dim);
        j = (int)aj;
        aj -= (float)j;

        if (j == dim)
            j -= 1;

        if (aj > prob[j])
            j = alias[j];

        return j;
    }


    __inline__ __device__ int DeviceAliasDataMap::sample(curandState* state) {
        return this->map[this->table.sample(state)];
    }


    __inline__ __device__ float DeviceAliasDataEGS::sample(curandState* state, int g) {
        float rand1, rand2, aj;
        float x, dx, a, rnno1, _sample;
        int j, idx;

        rand1 = 1.f - curand_uniform(state);
        rand2 = 1.f - curand_uniform(state);
        aj  = rand1 * (float)dim.y;
        j   = (int)aj;
        aj -= (float)j;
        j = min(j, dim.y - 1);

        // Bound check (DEBUG)
        assert(g < dim.x);
        assert(j < dim.y);

        idx = g * dim.y + j;
        if (aj > wdata[idx])
            j = idata[idx];

        x = xdata[j];
        dx = xdata[j + 1] - x;

        idx = g * (dim.y + 1) + j;
        if (fdata[idx] > 0) {
            a = fdata[idx + 1] / fdata[idx] - 1.f;
            if (abs(a) < 0.2f) {
                rnno1 = 0.5f * (1.0f - rand2) * a;
                _sample = x + rand2 * dx * (1.f + rnno1 * (1.f - rand2 * a));
            }
            else
                _sample = x - dx / a * (1.f - sqrtf(1.f + rand2 * a * (2.f + a)));
        }
        else
            _sample = x + dx * sqrtf(rand2);
        return _sample;
    }


    __inline__ __device__ float DeviceAliasDataEGS::sample2D(curandState* state, int g) {
        float rand1, rand2, aj;
        float x, dx, a, rnno1, _sample;
        int j, idx;

        rand1 = 1.f - curand_uniform(state);
        rand2 = 1.f - curand_uniform(state);
        aj  = rand1 * (float)dim.y;
        j   = (int)aj;
        aj -= (float)j;
        j = min(j, dim.y - 1);

        // Bound check (DEBUG)
        assert(g < dim.x);
        assert(j < dim.y);

        idx = g * dim.y + j;
        if (aj > wdata[idx])
            j = idata[idx];

        idx = g * (dim.y + 1) + j;
        x = xdata[idx];
        dx = xdata[idx + 1] - x;

        if (fdata[idx] > 0) {
            a = fdata[idx + 1] / fdata[idx] - 1.f;
            if (abs(a) < 0.2f) {
                rnno1 = 0.5f * (1.0f - rand2) * a;
                _sample = x + rand2 * dx * (1.f + rnno1 * (1.f - rand2 * a));
            }
            else
                _sample = x - dx / a * (1.f - sqrtf(1.f + rand2 * a * (2.f + a)));
        }
        else
            _sample = x + dx * sqrtf(rand2);
        return _sample;
    }


    __inline__ __device__ float rayBoxDistance(float3 vmin, float3 vmax, float3 rpos, float3 rdir) {
        float t[8];

        t[0] = (vmin.x - rpos.x) / rdir.x;
        t[1] = (vmax.x - rpos.x) / rdir.x;
        t[2] = (vmin.y - rpos.y) / rdir.y;
        t[3] = (vmax.y - rpos.y) / rdir.y;
        t[4] = (vmin.z - rpos.z) / rdir.z;
        t[5] = (vmax.z - rpos.z) / rdir.z;

        t[6] = fmaxf(fmaxf(fminf(t[0], t[1]), fminf(t[2], t[3])), fminf(t[4], t[5]));
        t[7] = fminf(fminf(fmaxf(t[0], t[1]), fmaxf(t[2], t[3])), fmaxf(t[4], t[5]));

        return (t[7] < 0 || t[6] > t[7]) ? 1e10f : t[6];
    }


    __inline__ __device__ bool  raySphereIntersection(float3 rpos, float3 rdir, float radius) {
        float dt = 0.f;
        dt -= rpos.x * rdir.x;
        dt -= rpos.y * rdir.y;
        dt -= rpos.z * rdir.z;

        // MD
        float temp;
        float rmin = 0.f;
        temp  = rpos.x + rdir.x * dt;
        rmin += temp * temp;
        temp  = rpos.y + rdir.y * dt;
        rmin += temp * temp;
        temp  = rpos.z + rdir.z * dt;
        rmin += temp * temp;
        rmin  = sqrtf(rmin);

        return rmin < radius;
    }


    __inline__ __device__ float gammaDistribution(curandState* state, float a, float b) {
        float coeff = 1.f;
        if (a < 1) {
            coeff = powf(curand_uniform(state), 1.f / a);
            a    += 1.f;
        }
        
        float d = a - 1.f / 3.f;
        float c = 1.f / 3.f / sqrtf(d);

        float x, v, u;
        while (true) {
            do {
                x = curand_normal(state);
                v = 1.f + c * x;
            } while (v <= 0.f);

            v = v * v * v;
            u = curand_uniform(state);
            x = x * x;
            if (u < 1.f - 0.0331 * x * x)
                break;
            if (logf(u) < 0.5f * x + d * (1.f - v + logf(v)))
                break;
        }
        return b * d * v * coeff;
    }


#endif


}
