
#include "aux_score.cuh"


namespace tally {


    __global__ void __kernel__appendYieldFromBuffer(
        mcutil::RingBuffer*    buffer,
        mcutil::BUFFER_TYPE    btype,
        DeviceSecYield**       yield_list,
        int                    n_yield
    ) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        for (unsigned long long int bpos = buffer->tail; bpos < buffer->head; bpos += blockDim.x * gridDim.x) {
            int buffer_idx = (bpos + idx) % buffer->size;
            float mu  = buffer->w[buffer_idx];
            float eke = buffer->e[buffer_idx];
            float wee = buffer->wee[buffer_idx];
            mcutil::UNION_FLAGS flags(buffer->flags[buffer_idx]);
            for (int i = 0; i < n_yield; ++i) {
                bool is_ion_target = btype != mcutil::BUFFER_TYPE::GENION || yield_list[i]->isIonTarget(flags.genion.ion_idx);
                if (is_ion_target && bpos + idx < buffer->head)
                    yield_list[i]->append(mu, wee, yield_list[i]->getBinIndex(eke));
                __syncthreads();
            }
        }
    }


    __host__ void appendYieldFromBuffer(int block, int thread, mcutil::RingBuffer* buffer, mcutil::BUFFER_TYPE btype, DeviceYieldHandle* handle_ptr) {

        if (!handle_ptr->n_yield)
            return;

        __kernel__appendYieldFromBuffer <<< block, thread >>> (&buffer[btype], btype, handle_ptr->yield, handle_ptr->n_yield);

        return;
    }


}