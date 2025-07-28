#pragma once

#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>


namespace geo {


    constexpr int MISS_MATERIAL_INDICATOR   = -1;
    constexpr int VACUUM_MATERIAL_INDICATOR = -10;


    typedef struct PhaseSpace {
        int          type;
        unsigned int flags;
        int          hid;
        float3 position;
        float3 direction;
        float  weight;
        float  energy;
        
        __inline__ __device__ void getSecondaryVector(float4* trigon);
        __inline__ __device__ void getRelativeVector(float3* vector);
    } PhaseSpace;


    typedef struct PhaseSpacePid {  // To used in phase space I/O
        int    pid;
        float3 position;
        float3 direction;
        float  weight;
        float  energy;
    } PhaseSpacePid;


    typedef struct DeviceTracerHandle {
        float*   x;  //!< @brief Particle initial x position (cm)
        float*   y;  //!< @brief Particle initial y position (cm)
        float*   z;  //!< @brief Particle initial z position (cm)
        float*   u;
        float*   v;
        float*   w;
        float*   e;
        float*   wee;
        float*   track;
        ushort2* regions;  // { region old, region new }
        float*   nu;       // surface normal vector x
        float*   nv;       // surface normal vector y
        float*   nw;       // surface normal vector z
        unsigned int* flags;
        unsigned int* hid;
    } DeviceTracerHandle;


    //__inline__ __device__ void sampleAzhimuthalAngle(curandState* state, float4* trigon);

    /**
    *
    * @brief Sample uniform azimuthal angle sine and cosine
    * @param state XORWOW PRNG state
    *
    * @return Azimuthal angle sine and cosine (sinp, cosp)
    */
    __inline__ __device__ float2 sampleAzimuthalAngle(curandState* state);


    /**
    *
    * @brief Rotate direction vector by arbitrary horizontal and azimuthal angles
    * @param vector Direction vector (u, v, w)
    * @param hori   Polar angle sine and cosine (sint, cost)
    * @param azim   Azimuthal angle sine and cosine (sinp, cosp)
    *
    * @return Normalized direction vector after rotation (u, v, w)
    */
    __inline__ __device__ float3 rotateVector(float3 vector, float2 polar, float2 azim);


#ifdef __CUDACC__


    __device__ void PhaseSpace::getSecondaryVector(float4* trigon) {
        float us, vs, sinpsi, sindel, cosdel;
        sinpsi = direction.x * direction.x + direction.y * direction.y;

        // small polar angle, no need to rotate
        if (sinpsi < 10e-10f) {
            direction.x = trigon->y * trigon->z;
            direction.y = trigon->y * trigon->w;
            direction.z *= trigon->x;
        }
        else {
            sinpsi = sqrtf(sinpsi);
            us = trigon->y * trigon->z;
            vs = trigon->y * trigon->w;
            sindel = direction.y / sinpsi;
            cosdel = direction.x / sinpsi;

            direction.x = direction.z * cosdel * us - sindel * vs + direction.x * trigon->x;
            direction.y = direction.z * sindel * us + cosdel * vs + direction.y * trigon->x;
            direction.z = -sinpsi * us + direction.z * trigon->x;
        }

        // normalization
        us = norm3df(direction.x, direction.y, direction.z);
        direction.x /= us;
        direction.y /= us;
        direction.z /= us;
    }


    __device__ void PhaseSpace::getRelativeVector(float3* vector) {
        float sint2, cphi0, sphi0, u2p;
        sint2 = direction.x * direction.x + direction.y * direction.y;
        // small polar angle, no need to rotate
        if (sint2 > 1e-10f) {
            sint2 = sqrtf(sint2);
            cphi0 = direction.x / sint2;
            sphi0 = direction.y / sint2;

            u2p = direction.z * vector->x + sint2 * vector->z;
            vector->z = direction.z * vector->z - sint2 * vector->x;
            vector->x = u2p * cphi0 - vector->y * sphi0;
            vector->y = u2p * sphi0 + vector->y * cphi0;
        }

        // normalization
        u2p = norm3df(vector->x, vector->y, vector->z);
        vector->x /= u2p;
        vector->y /= u2p;
        vector->z /= u2p;
    }

    // __device__ void sampleAzhimuthalAngle(curandState* state, float4* trigon);

    /*
    __device__ void sampleAzhimuthalAngle(curandState* state, float4* trigon) {
        float phi = CURAND_2PI * (1.f - curand_uniform(state));
        sincosf(phi, &trigon->w, &trigon->z);
    }
    */

    __device__ float2 sampleAzimuthalAngle(curandState* state) {
        float  phi;
        float2 azim;
        phi = CURAND_2PI * (1.f - curand_uniform(state));
        sincosf(phi, &azim.x, &azim.y);
        return azim;
    }


    __device__ float3 rotateVector(float3 vector, float2 polar, float2 azim) {
        float us, vs, sinpsi, sindel, cosdel;
        sinpsi = vector.x * vector.x + vector.y * vector.y;

        // Small polar angle, no need to rotate
        if (sinpsi < 1e-10f) {
            vector.x  = polar.x * azim.y;
            vector.y  = polar.x * azim.x;
            vector.z *= polar.y;
        }
        else {
            sinpsi = sqrtf(sinpsi);
            us = polar.x * azim.y;
            vs = polar.x * azim.x;
            sindel = vector.y / sinpsi;
            cosdel = vector.x / sinpsi;
            
            vector.x = vector.z * cosdel * us - sindel * vs + vector.x * polar.y;
            vector.y = vector.z * sindel * us + cosdel * vs + vector.y * polar.y;
            vector.z = -sinpsi * us + vector.z * polar.y;
        }

        // Normalization
        us = norm3df(vector.x, vector.y, vector.z);
        vector.x /= us;
        vector.y /= us;
        vector.z /= us;

        return vector;
    }


#endif
    

}