
#include "auxiliary.cuh"

#include "device/shuffle.cuh"


namespace auxiliary {


    __constant__ bool BUFFER_HAS_HID;

    __device__ mcutil::RingBuffer* buffer_catalog;
    __device__ geo::DeviceTracerHandle* tracer_handle;

    __device__ mcutil::RingBuffer* qmd_problems;

    __device__   float* NGROUP;

    __device__ DeviceEvent* event_list;
    __device__ mcutil::DeviceAliasData* event_ptable;

    __device__ curandState* rand_state;


    __host__ void initPhaseSpace(int block, int thread, mcutil::BUFFER_TYPE bid, int hid) {
        __kernel__initPhaseSpace <<< block, thread >>> (bid, hid);
    }


    __global__ void __kernel__initPhaseSpace(mcutil::BUFFER_TYPE bid, int hid) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int target;

        uchar4  zapt;

        mcutil::UNION_FLAGS flags(0x0u);
        flags.base.region = (unsigned short)0;

        // sample event
        int event_id = event_ptable->sample(&rand_state[idx]);
        DeviceEvent esample = event_list[event_id];

        float ene = esample.energy;

        switch (bid) {
        case mcutil::BUFFER_TYPE::QMD:
            zapt.x = (unsigned char)(esample.aux1 / 1000);
            zapt.y = (unsigned char)(esample.aux1 % 1000);
            zapt.z = (unsigned char)(esample.aux2 / 1000);
            zapt.w = (unsigned char)(esample.aux2 % 1000);
            ene   *= 1e-3f;  // MeV/u -> GeV/u
            break;
        case mcutil::BUFFER_TYPE::DEEXCITATION:
            flags.deex.z = (unsigned char)(esample.aux1 / 1000);
            flags.deex.a = (unsigned char)(esample.aux1 % 1000);
            break;
        default:
            assert(false);
            break;
        }

        if (bid == mcutil::BUFFER_TYPE::QMD) {
            int dim  = (esample.aux1 % 1000) + (esample.aux2 % 1000);
            int sbid = dim / CUDA_WARP_SIZE;
            if (dim % CUDA_WARP_SIZE)
                sbid++;
            target = qmd_problems[sbid].pushBulk();
            qmd_problems[sbid].x[target]     = esample.pos.x;
            qmd_problems[sbid].y[target]     = esample.pos.y;
            qmd_problems[sbid].z[target]     = esample.pos.z;
            qmd_problems[sbid].u[target]     = esample.dir.x;
            qmd_problems[sbid].v[target]     = esample.dir.y;
            qmd_problems[sbid].w[target]     = esample.dir.z;
            qmd_problems[sbid].e[target]     = ene;
            qmd_problems[sbid].wee[target]   = 1.f;
            qmd_problems[sbid].flags[target] = flags.astype<unsigned int>();
            qmd_problems[sbid].za[target]    = zapt;

            if (BUFFER_HAS_HID)
                qmd_problems[sbid].hid[target] = hid + idx;
        }
        else {
            target = buffer_catalog[bid].pushBulk();
            buffer_catalog[bid].x[target]     = esample.pos.x;
            buffer_catalog[bid].y[target]     = esample.pos.y;
            buffer_catalog[bid].z[target]     = esample.pos.z;
            buffer_catalog[bid].u[target]     = esample.dir.x;
            buffer_catalog[bid].v[target]     = esample.dir.y;
            buffer_catalog[bid].w[target]     = esample.dir.z;
            buffer_catalog[bid].e[target]     = ene;
            buffer_catalog[bid].wee[target]   = 1.f;
            buffer_catalog[bid].flags[target] = flags.astype<unsigned int>();

            if (BUFFER_HAS_HID)
                buffer_catalog[bid].hid[target] = hid + idx;

        }
    }


    __host__ void dummyOptixLaunch(int block, int thread, int region) {
        __kernel__dummyOptixLaunch <<< block, thread >>> (region);
    }


    __global__ void __kernel__dummyOptixLaunch(int region) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        ushort2 regs;
        regs.x = (unsigned short)region;
        regs.y = (unsigned short)region;
        tracer_handle->regions[idx] = regs;
        tracer_handle->track[idx]   = 1e20f;  // infinity
    }


    __host__ cudaError_t setBufferHandle(CUdeviceptr handle, bool has_hid) {
        M_SOASymbolMapper(mcutil::RingBuffer*, handle, buffer_catalog);
        M_SOAPtrMapper(bool, has_hid, BUFFER_HAS_HID);
        return cudaSuccess;
    }


    __host__ cudaError_t setTracerHandle(CUdeviceptr handle) {
        M_SOASymbolMapper(geo::DeviceTracerHandle*, handle, tracer_handle);
        return cudaSuccess;
    }


    __host__ cudaError_t setQMDBuffer(CUdeviceptr ptr_qmd_buffer) {
        M_SOASymbolMapper(mcutil::RingBuffer*, ptr_qmd_buffer, qmd_problems);
        return cudaSuccess;
    }


    __host__ cudaError_t setPrngHandle(CUdeviceptr handle) {
        M_SOASymbolMapper(curandState*, handle, rand_state);
        return cudaSuccess;
    }


    __host__ cudaError_t setNeutronGroupHandle(CUdeviceptr handle) {
        M_SOASymbolMapper(float*, handle, NGROUP);
        return cudaSuccess;
    }


    __host__ cudaError_t setEventListPtr(DeviceEvent* ptr_event_list, mcutil::DeviceAliasData* ptr_event_ptable) {
        M_SOAPtrMapper(DeviceEvent*, ptr_event_list, event_list);
        M_SOAPtrMapper(mcutil::DeviceAliasData**, ptr_event_ptable, event_ptable);
        return cudaSuccess;
    }


}