#pragma once

#include <stdio.h>

#include <cuda_runtime.h>
#include <assert.h>


namespace tally {

    
    // build sparse uncertainty matrix from data sparse coo index
    __global__ void __kernel__buildUncertaintyCOOSparse(
        float* __restrict__ dense_unc_ptr,
        int*   __restrict__ coo_index_ptr,
        float* __restrict__ coo_unc_ptr,
        int    nnz
    );


    __host__ void __host__buildUncertaintyCOOSparse(
        float* dense_unc_ptr,
        int*   coo_index_ptr,
        float* coo_unc_ptr,
        int    nnz,
        int    block,
        int    thread
    );


}