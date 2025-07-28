
#include "aux_score.cuh"


namespace tally {


    __global__ void __kernel__appendYieldFromBuffer(
        mcutil::RingBuffer*    buffer,
        mcutil::BUFFER_TYPE    btype,
        unsigned long long int from, 
        unsigned long long int to, 
        DeviceSecYield**       yield_list,
        int                    n_yield
    ) {
        int idx        = threadIdx.x + blockDim.x * blockIdx.x;
        int buffer_idx = (from + idx) % buffer->size;

        float mu  = buffer->w[buffer_idx];
        float eke = buffer->e[buffer_idx];
        float wee = buffer->wee[buffer_idx];

        mcutil::UNION_FLAGS flags(buffer->flags[buffer_idx]);

        for (int i = 0; i < n_yield; ++i) {
            bool is_ion_target = btype != mcutil::BUFFER_TYPE::GENION || yield_list[i]->isIonTarget(flags.genion.ion_idx);
            if (is_ion_target && from + idx < to)
                yield_list[i]->append(mu, wee, yield_list[i]->getBinIndex(eke));
            __syncthreads();
        }
    }


    __host__ void appendYieldFromBuffer(int block, int thread, mcutil::RingBuffer* buffer, mcutil::BUFFER_TYPE btype, DeviceYieldHandle* handle_ptr) {
        mcutil::RingBuffer buffer_host;

        if (!handle_ptr->n_yield)
            return;

        cudaMemcpy(&buffer_host, &buffer[btype],
            sizeof(mcutil::RingBuffer), cudaMemcpyDeviceToHost);

        unsigned long long int from = buffer_host.tail;
        unsigned long long int to   = buffer_host.head;
        unsigned long long int size = block * thread;
        
        for (size_t i = from; i < to; i += size) {
            __kernel__appendYieldFromBuffer <<< block, thread >>> (&buffer[btype], btype, i, to, handle_ptr->yield, handle_ptr->n_yield);
        }
        return;
    }


}