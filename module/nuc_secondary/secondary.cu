
#include "secondary.cuh"
#include "device/shuffle.cuh"
#include "device/assert.cuh"

#include <stdio.h>


namespace Hadron {


    __global__ void __kernel__secondaryStep() {
        int lane_idx = threadIdx.x % CUDA_WARP_SIZE;

        // pull particle data from buffer
        buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].pullShared(blockDim.x);

        // shared cache
        int* cache_univ_i  = reinterpret_cast<int*>(mcutil::cache_univ);
        int  targetp       = (cache_univ_i[0] + threadIdx.x) % cache_univ_i[1];

        // data
        mcutil::UNION_FLAGS flags(buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].flags[targetp]);

        assert(flags.deex.a > 0);

        // momentum
        float3 dir;
        dir.x = buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].u[targetp];  // [GeV/c]
        dir.y = buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].v[targetp];
        dir.z = buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].w[targetp];

        // position and weight
        float3 pos;
        pos.x = buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].x[targetp];
        pos.y = buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].y[targetp];
        pos.z = buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].z[targetp];

        // DEBUG phasespace
        assertNAN(dir.x);
        assertNAN(dir.y);
        assertNAN(dir.z);
        assertNAN(pos.x);
        assertNAN(pos.y);
        assertNAN(pos.z);

        float weight = buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].wee[targetp];

        unsigned int hid;
        if (BUFFER_HAS_HID)
            hid = buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].hid[targetp];

        int  mask, size, offset;
        bool current_target;

        // case 0, double channel (2n, 2p), recursive
        current_target = 
            (flags.deex.z == 2 && flags.deex.a == 2) || 
            (flags.deex.z == 0 && flags.deex.a == 2);

        mask = __ballot_sync(0xffffffff, current_target);
        size = __popc(mask);
        if (size) {
            if (!lane_idx)
                offset = buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].pushAtomicWarp(size);
            offset  = __shfl_sync(0xffffffff, offset, 0);
            mask   &= ~(0xffffffff << lane_idx);
            mask    = __popc(mask);
            offset += mask;
            offset %= buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].size;
            if (current_target) {

                flags.deex.z /= 2;
                flags.deex.a /= 2;
                dir.x *= 0.5f;  // split momentum
                dir.y *= 0.5f;
                dir.z *= 0.5f;

                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].x[offset]     = pos.x;
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].y[offset]     = pos.y;
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].z[offset]     = pos.z;
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].u[offset]     = dir.x;  
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].v[offset]     = dir.y;
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].w[offset]     = dir.z;
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].wee[offset]   = weight;
                buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].flags[offset] = flags.astype<unsigned int>();
                if (BUFFER_HAS_HID)
                    buffer_catalog[mcutil::BUFFER_TYPE::NUC_SECONDARY].hid[offset] = hid;
            }
            
        }

        float np = norm3df(dir.x, dir.y, dir.z);  // momentum norm
        dir.x /= np;
        dir.y /= np;
        dir.z /= np;  // now direction vector

        // mass
        float m;
        m = mass_table[flags.deex.z].get(flags.deex.a) * 1e-3f;  // [GeV/c^2]

        // kinetic energy
        float eke = sqrtf(m * m + np * np) - m; // kinetic energy [GeV]
        eke *= 1e3f;  // [MeV]
        eke /= (float)flags.deex.a;  // per nucleon

        assertNAN(eke);
        assert(eke >= -0.1f);  // 0.1 MeV error
        eke  = fmaxf(0.f, eke);  // avoid FP error

        // case 1, neutron
        current_target = flags.deex.z == 0 && flags.deex.a == 1;

        mask = __ballot_sync(0xffffffff, current_target);
        size = __popc(mask);
        if (size) {
            if (!lane_idx)
                offset = buffer_catalog[mcutil::BUFFER_TYPE::GNEUTRON].pushAtomicWarp(size);
            offset  = __shfl_sync(0xffffffff, offset, 0);
            mask   &= ~(0xffffffff << lane_idx);
            mask    = __popc(mask);
            offset += mask;
            offset %= buffer_catalog[mcutil::BUFFER_TYPE::GNEUTRON].size;
            if (current_target) {
                buffer_catalog[mcutil::BUFFER_TYPE::GNEUTRON].x[offset]     = pos.x;
                buffer_catalog[mcutil::BUFFER_TYPE::GNEUTRON].y[offset]     = pos.y;
                buffer_catalog[mcutil::BUFFER_TYPE::GNEUTRON].z[offset]     = pos.z;
                buffer_catalog[mcutil::BUFFER_TYPE::GNEUTRON].u[offset]     = dir.x;
                buffer_catalog[mcutil::BUFFER_TYPE::GNEUTRON].v[offset]     = dir.y;
                buffer_catalog[mcutil::BUFFER_TYPE::GNEUTRON].w[offset]     = dir.z;
                buffer_catalog[mcutil::BUFFER_TYPE::GNEUTRON].e[offset]     = eke;
                buffer_catalog[mcutil::BUFFER_TYPE::GNEUTRON].wee[offset]   = weight;
                buffer_catalog[mcutil::BUFFER_TYPE::GNEUTRON].flags[offset] = flags.astype<unsigned int>();
                if (BUFFER_HAS_HID)
                    buffer_catalog[mcutil::BUFFER_TYPE::GNEUTRON].hid[offset] = hid;
            }
        }

        // case 2, generic ion (transportable)
        current_target = 
            (flags.deex.z > 0) &&
            (flags.deex.z <= Projectile::TABLE_MAX_Z) &&
            (flags.deex.a <= Projectile::TABLE_MAX_A);

        // get iid
        int iid = PROJSOA1D::getIonIndex(flags.deex.z, flags.deex.a);
        
        // initialize genion flags
        mcutil::UNION_FLAGS flags_genion(flags);

        flags_genion.genion.fmask   = 0u;
        flags_genion.genion.ion_idx = iid;

        mask = __ballot_sync(0xffffffff, current_target && iid >= 0);
        size = __popc(mask);
        if (size) {
            if (!lane_idx)
                offset = buffer_catalog[mcutil::BUFFER_TYPE::GENION].pushAtomicWarp(size);
            offset  = __shfl_sync(0xffffffff, offset, 0);
            mask   &= ~(0xffffffff << lane_idx);
            mask    = __popc(mask);
            offset += mask;
            offset %= buffer_catalog[mcutil::BUFFER_TYPE::GENION].size;
            if (current_target) {
                buffer_catalog[mcutil::BUFFER_TYPE::GENION].x[offset]     = pos.x;
                buffer_catalog[mcutil::BUFFER_TYPE::GENION].y[offset]     = pos.y;
                buffer_catalog[mcutil::BUFFER_TYPE::GENION].z[offset]     = pos.z;
                buffer_catalog[mcutil::BUFFER_TYPE::GENION].u[offset]     = dir.x;
                buffer_catalog[mcutil::BUFFER_TYPE::GENION].v[offset]     = dir.y;
                buffer_catalog[mcutil::BUFFER_TYPE::GENION].w[offset]     = dir.z;
                buffer_catalog[mcutil::BUFFER_TYPE::GENION].e[offset]     = eke;
                buffer_catalog[mcutil::BUFFER_TYPE::GENION].wee[offset]   = weight;
                buffer_catalog[mcutil::BUFFER_TYPE::GENION].flags[offset] = flags_genion.astype<unsigned int>();
                if (BUFFER_HAS_HID)
                    buffer_catalog[mcutil::BUFFER_TYPE::GENION].hid[offset] = hid;
            }
        }

        // case 3, generic ion (local depo)
        int mat_id     = region_mat_table[flags.base.region];
        current_target = flags.deex.z > Projectile::TABLE_MAX_Z || flags.deex.a > Projectile::TABLE_MAX_A;
        eke           *= (float)flags.deex.a;
        
        // Depo / Dose
        for (int i = 0; i < tally_catalog.n_mesh_density; ++i) {  // mesh density
            if (current_target && tally_catalog.mesh_density[i]->isIonTarget())
                tally_catalog.mesh_density[i]->append(
                    pos,
                    eke,
                    1.f,  // RBE future
                    weight,
                    MATSOA1D::density[mat_id]
                );
            __syncthreads();
        }
        for (int i = 0; i < tally_catalog.n_density; ++i) {  // density
            if (current_target && tally_catalog.density[i]->isIonTarget())
                tally_catalog.density[i]->append(
                    flags.base.region,
                    eke,
                    1.f,  // RBE future
                    weight,
                    MATSOA1D::density[mat_id]
                );
            __syncthreads();
        }

        // LET
        for (int i = 0; i < tally_catalog.n_mesh_letd; ++i) {
            if (current_target && tally_catalog.mesh_letd[i]->isIonTarget())
                tally_catalog.mesh_letd[i]->append(
                    pos,
                    eke,
                    eke * constants::LET_RANGE_UNDEFINED_I,
                    weight
                );
            __syncthreads();
        }
        for (int i = 0; i < tally_catalog.n_letd; ++i) {
            if (current_target && tally_catalog.letd[i]->isIonTarget())
                tally_catalog.letd[i]->append(
                    flags.base.region,
                    eke,
                    eke * constants::LET_RANGE_UNDEFINED_I,
                    weight
                );
            __syncthreads();
        }

        // PHD
        for (int i = 0; i < tally_catalog.n_detector; ++i) {  // detector
            if (current_target && tally_catalog.detector[i]->isIonTarget())
                tally_catalog.detector[i]->append(flags.base.region, weight, eke, hid);
            __syncthreads();
        }
        
        return;
    }


    __host__ void secondaryStep(int block, int thread) {
        __kernel__secondaryStep <<< block, thread, mcutil::SIZE_SHARED_MEMORY_GLOBAL >>> ();
    }


    namespace MATSOA1D {


        __device__ float* density;


        __host__ cudaError_t setDensity(CUdeviceptr deviceptr) {
            M_SOASymbolMapper(float*, deviceptr, density);
            return cudaSuccess;
        }


    }


    namespace PROJSOA1D {

        __constant__ int PROJECTILE_TABLE_SIZE;

        // projectile
        __device__ int     offset[Hadron::Projectile::TABLE_MAX_Z];
        __device__ uchar2* za;
        __device__ float*  mass;
        __device__ float*  mass_u;
        __device__ float*  mass_ratio;
        __device__ int*    spin;


        __host__ cudaError_t setTableSize(int size) {
            return cudaMemcpyToSymbol(PROJECTILE_TABLE_SIZE, &size, sizeof(int));
        }


        __host__ cudaError_t setOffset(int* offset_ptr) {
            return cudaMemcpyToSymbol(offset[0], offset_ptr, sizeof(int) * Hadron::Projectile::TABLE_MAX_Z);
        }


        __host__ cudaError_t setTable(uchar2* za_ptr, float* mass_ptr, float* mass_u_ptr, float* mass_ratio_ptr, int* spin_ptr) {
            M_SOAPtrMapper(uchar2*, za_ptr,         za);
            M_SOAPtrMapper(float*,  mass_ptr,       mass);
            M_SOAPtrMapper(float*,  mass_u_ptr,     mass_u);
            M_SOAPtrMapper(float*,  mass_ratio_ptr, mass_ratio);
            M_SOAPtrMapper(int*,    spin_ptr,       spin);
            return cudaSuccess;
        }


    }


    __device__ float* NGROUP;

    __device__ int* region_mat_table;

    __device__ tally::DeviceHandle tally_catalog;

    __constant__ bool BUFFER_HAS_HID;
    __device__ mcutil::RingBuffer* buffer_catalog; 

    __device__ curandState* rand_state;

    __device__ Nucleus::MassTable* mass_table;


    __host__ cudaError_t setNeutronGroupHandle(CUdeviceptr handle) {
        M_SOASymbolMapper(float*, handle, NGROUP);
        return cudaSuccess;
    }


    __host__ cudaError_t setRegionMaterialTable(CUdeviceptr table) {
        M_SOASymbolMapper(int*, table, region_mat_table);
        return cudaSuccess;
    }


    __host__ cudaError_t setTallyHandle(const tally::DeviceHandle& handle) {
        M_SOAPtrMapper(tally::DeviceHandle, handle, tally_catalog);
        return cudaSuccess;
    }


    __host__ cudaError_t setBufferHandle(CUdeviceptr handle, bool has_hid) {
        M_SOASymbolMapper(mcutil::RingBuffer*, handle, buffer_catalog);
        M_SOAPtrMapper(bool, has_hid, BUFFER_HAS_HID);
        return cudaSuccess;
    }


    __host__ cudaError_t setPrngHandle(CUdeviceptr handle) {
        M_SOASymbolMapper(curandState*, handle, rand_state);
        return cudaSuccess;
    }


    __host__ cudaError_t setMassTableHandle(CUdeviceptr handle) {
        M_SOASymbolMapper(Nucleus::MassTable*, handle, mass_table);
        return cudaSuccess;
    }


}