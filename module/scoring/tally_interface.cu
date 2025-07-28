
#include "tally_interface.cuh"


namespace tally {


    __global__ void __kernel__buildUncertaintyCOOSparse(
        float* __restrict__ dense_unc_ptr,
        int*   __restrict__ coo_index_ptr,
        float* __restrict__ coo_unc_ptr,
        int    nnz
    ) {
        int idx    = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = gridDim.x  * blockDim.x;

        for (int iter = 0; iter * stride < nnz; ++iter) {
            int from_ci = idx + stride * iter;
            int from_di;
            if (from_ci < nnz) {
                from_di              = coo_index_ptr[from_ci];
                coo_unc_ptr[from_ci] = dense_unc_ptr[from_di];
            }
            __syncthreads();
        }

        return;
    }


    __host__ void __host__buildUncertaintyCOOSparse(
        float* dense_unc_ptr,
        int*   coo_index_ptr,
        float* coo_unc_ptr,
        int    nnz,
        int    block,
        int    thread
    ) {
        __kernel__buildUncertaintyCOOSparse <<< block, thread >>> (dense_unc_ptr, coo_index_ptr, coo_unc_ptr, nnz);
    }


}