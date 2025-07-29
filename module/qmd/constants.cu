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
 * @file    module/qmd/constants.cu
 * @brief   QMD constants
 * @author  CM Lee
 * @date    02/14/2024
 */


#include "constants.cuh"


namespace RT2QMD {
    namespace constants {


        __constant__ float H_SKYRME_C0;
        __constant__ float H_SKYRME_C3;
        __constant__ float H_SYMMETRY;
        __constant__ float H_COULOMB;

        __constant__ float D_SKYRME_C0;
        __constant__ float D_SKYRME_C0_S;
        __constant__ float D_COULOMB;

        __constant__ float CLUSTER_CPF2;

        __constant__ float G_SKYRME_C0;
        __constant__ float G_SKYRME_C3;
        __constant__ float G_SYMMETRY;

        __constant__ float GS_CD;
        __constant__ float GS_C0;
        __constant__ float GS_C3;
        __constant__ float GS_CS;
        __constant__ float GS_CL;


        cudaError_t setSymbolHCoeffs(float h_skyrme_c0, float h_skyrme_c3, float h_symmetry, float h_coulomb) {
            cudaError_t res;
            res = cudaMemcpyToSymbol(H_SKYRME_C0, &h_skyrme_c0,
                sizeof(float));
            if (res != cudaSuccess)
                return res;
            res = cudaMemcpyToSymbol(H_SKYRME_C3, &h_skyrme_c3,
                sizeof(float));
            if (res != cudaSuccess)
                return res;
            res = cudaMemcpyToSymbol(H_SYMMETRY, &h_symmetry,
                sizeof(float));
            if (res != cudaSuccess)
                return res;
            return cudaMemcpyToSymbol(H_COULOMB, &h_coulomb, 
                sizeof(float));
        }


        cudaError_t setSymbolHDistance(float d_skyrme_c0, float d_skyrme_c0_s, float d_coulomb) {
            cudaError_t res;
            res = cudaMemcpyToSymbol(D_SKYRME_C0, &d_skyrme_c0,
                sizeof(float));
            if (res != cudaSuccess)
                return res;
            res = cudaMemcpyToSymbol(D_SKYRME_C0_S, &d_skyrme_c0_s,
                sizeof(float));
            if (res != cudaSuccess)
                return res;
            return cudaMemcpyToSymbol(D_COULOMB, &d_coulomb,
                sizeof(float));
        }


        cudaError_t setSymbolClusterCoeffs(float cpf2) {
            return cudaMemcpyToSymbol(CLUSTER_CPF2, &cpf2,
                sizeof(float));
        }


        cudaError_t setSymbolHGradient(float g_skyrme_c0, float g_skyrme_c3, float g_symmetry) {
            cudaError_t res;
            res = cudaMemcpyToSymbol(G_SKYRME_C0, &g_skyrme_c0,
                sizeof(float));
            if (res != cudaSuccess)
                return res;
            res = cudaMemcpyToSymbol(G_SKYRME_C3, &g_skyrme_c3,
                sizeof(float));
            if (res != cudaSuccess)
                return res;
            return cudaMemcpyToSymbol(G_SYMMETRY, &g_symmetry,
                sizeof(float));
        }


        cudaError_t setSymbolGroundNucleusCoeffs(float gs_cd, float gs_c0, float gs_c3, float gs_cs, float gs_cl) {
            cudaError_t res;
            res = cudaMemcpyToSymbol(GS_CD, &gs_cd,
                sizeof(float));
            if (res != cudaSuccess)
                return res;
            res = cudaMemcpyToSymbol(GS_C0, &gs_c0,
                sizeof(float));
            if (res != cudaSuccess)
                return res;
            res = cudaMemcpyToSymbol(GS_C3, &gs_c3,
                sizeof(float));
            if (res != cudaSuccess)
                return res;
            res = cudaMemcpyToSymbol(GS_CS, &gs_cs,
                sizeof(float));
            if (res != cudaSuccess)
                return res;
            return cudaMemcpyToSymbol(GS_CL, &gs_cl,
                sizeof(float));
        }


    }
}