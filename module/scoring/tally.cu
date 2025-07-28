
#include "tally.cuh"

#include "device/memory.cuh"


namespace tally {


#ifndef RT2QMD_STANDALONE


    __host__ void __host__deviceGetPhaseSpaceSaturation(int thread, DevicePhaseSpace** buffer, float* saturation) {
        __device__deviceGetPhaseSpaceSaturation <<< 1, thread, mcutil::SIZE_SHARED_MEMORY_GLOBAL >>> (buffer, saturation);
    }


    __global__ void __device__deviceGetPhaseSpaceSaturation(DevicePhaseSpace** ps, float* saturation) {
        saturation[threadIdx.x] = ps[threadIdx.x]->buffer.getBufferSaturation();
        __syncthreads();
    }


    __host__ void __host__deviceInitDetector(int block, int thread, DeviceDetector* det, int det_size) {
        __device__deviceInitDetector <<< block, thread, mcutil::SIZE_SHARED_MEMORY_GLOBAL >>> (det, det_size);
    }


    __global__ void __device__deviceInitDetector(DeviceDetector* det, int det_size) {
        int idx      = threadIdx.x + blockDim.x * blockIdx.x;
        int dim      = blockDim.x * gridDim.x;
        int max_iter = det_size /  dim;
        for (int i = 0; i < max_iter; ++i) {
            if (idx + i * dim < det_size) {
                det->data_hid_1d[idx + i * dim] = 0.0;
                det->prim_weight[idx + i * dim] = 0.f;
            }
            __syncthreads();
        }
    }


    __host__ void __host__deviceProcessDetector(int block, int thread, DeviceDetector* det, int det_size) {
        __device__deviceProcessDetector <<< block, thread, mcutil::SIZE_SHARED_MEMORY_GLOBAL >>> (det, det_size);
    }


    __global__ void __device__deviceProcessDetector(DeviceDetector* det, int det_size) {
        int idx      = threadIdx.x + blockDim.x * blockIdx.x;
        int dim      = blockDim.x * gridDim.x;
        int max_iter = det_size / dim;
        for (int i = 0; i < max_iter; ++i) {
            if (idx + i * dim < det_size) {
                float  signal  = det->data_hid_1d[idx + i * dim];
                double tweight = det->prim_weight[idx + i * dim];
                signal /= tweight;  // normalize
                float  fwhm    = det->a + det->b * sqrtf(signal + det->c * signal * signal);
                signal += fwhm * 0.42466f * curand_normal(&rand_state[idx]);
                int    ebin    = det->getBinIndex(signal);
                if (ebin < det->nebin && ebin >= 0) {
                    atomicAdd(&det->data[ebin].x, tweight);
                    atomicAdd(&det->data[ebin].y, tweight * tweight);  // var
                }
            }
            __syncthreads();
        }
    }


#endif


    __device__ curandState* rand_state;


    __host__ cudaError_t setPrngHandle(CUdeviceptr handle) {
        M_SOASymbolMapper(curandState*, handle, rand_state);
        return cudaSuccess;
    }


}
