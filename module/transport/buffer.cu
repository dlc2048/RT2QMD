
#include "buffer.cuh"

#include <stdio.h>


namespace mcutil {

	__device__   geo::DeviceTracerHandle* tracer_handle;
	__constant__ bool BUFFER_HAS_HID;


	__host__ void __host__deviceGetBufferPriority(size_t thread, RingBuffer* buffer, int* target) {
		__device__deviceGetBufferPriority <<< 1, (unsigned int)thread, MAX_DIMENSION_THREAD * 3 * sizeof(int) >>> (buffer, target);
	}


	__global__ void __device__deviceGetBufferPriority(RingBuffer* buffer, int* target) {
		// queue occupation list
		float* q_occupy = reinterpret_cast<float*>(&cache_univ[SHARED_OFFSET_BUFFER_REDUX]);
		// id of most crowded queue
		int*   q_target = reinterpret_cast<int*  >(&cache_univ[SHARED_OFFSET_BUFFER_REDUX + MAX_DIMENSION_THREAD]);

		q_target[threadIdx.x] = threadIdx.x;
		q_occupy[threadIdx.x] = buffer[threadIdx.x].getBufferSaturation();

		for (int stride = 1; stride < blockDim.x; stride *= 2) {
			if (threadIdx.x % (2 * stride) == 0) {
				if (q_occupy[threadIdx.x] < q_occupy[threadIdx.x + stride]) {
					q_occupy[threadIdx.x] = q_occupy[threadIdx.x + stride];
					q_target[threadIdx.x] = q_target[threadIdx.x + stride];
				}
			}
			__syncthreads();
		}

		assert(q_occupy[0] < 0.8f);
		*target = q_target[0];
	}


	__host__ cudaError_t setBufferTracerHandle(CUdeviceptr handle) {
		M_SOASymbolMapper(geo::DeviceTracerHandle*, handle, tracer_handle);
		return cudaSuccess;
	}


	__host__ cudaError_t setHIDFlag(bool has_hid) {
		M_SOAPtrMapper(bool, has_hid, BUFFER_HAS_HID);
		return cudaSuccess;
	}


	__host__ void initializeTracerHandle(int block, int thread) {
		__kernel__deviceInitializeTracerHandle <<< block, thread, mcutil::SIZE_SHARED_MEMORY_GLOBAL >>> ();
	}


	__global__ void __kernel__deviceInitializeTracerHandle() {
		unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
		tracer_handle->wee[idx] = -1.f;
	}


	__host__ void __host__devicePullBulk(int block, int thread, RingBuffer* buffer) {
		__kernel__devicePullBulk <<< block, thread, mcutil::SIZE_SHARED_MEMORY_GLOBAL >>> (buffer);
	}


	__global__ void __kernel__devicePullBulk(RingBuffer* buffer) {
		unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
		int target = buffer->pullBulk();
		tracer_handle->x[idx]          = buffer->x[target];
		tracer_handle->y[idx]          = buffer->y[target];
		tracer_handle->z[idx]          = buffer->z[target];
		tracer_handle->u[idx]          = buffer->u[target];
		tracer_handle->v[idx]          = buffer->v[target];
		tracer_handle->w[idx]          = buffer->w[target];
		tracer_handle->e[idx]          = buffer->e[target];
		tracer_handle->wee[idx]        = buffer->wee[target];
		if (BUFFER_HAS_HID)
			tracer_handle->hid[idx]    = buffer->hid[target];

		// flag & region parsing
		mcutil::UNION_FLAGS flags(buffer->flags[target]);

		ushort2      regs;
		regs.x = flags.base.region;
		regs.y = USHRT_MAX;
		tracer_handle->regions[idx] = regs;
		tracer_handle->flags[idx]   = flags.astype<unsigned int>();
	}


	__host__ void __host__devicePullAtomic(int block, int thread, RingBuffer* buffer) {
		__kernel__devicePullAtomic <<< block, thread, mcutil::SIZE_SHARED_MEMORY_GLOBAL >>> (buffer);
	}


	__global__ void __kernel__devicePullAtomic(RingBuffer* buffer) {
		unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
		int target;
		if (tracer_handle->wee[idx] <= 0.f) {
			target = buffer->pullAtomic();
			tracer_handle->x[idx]   = buffer->x[target];
			tracer_handle->y[idx]   = buffer->y[target];
			tracer_handle->z[idx]   = buffer->z[target];
			tracer_handle->u[idx]   = buffer->u[target];
			tracer_handle->v[idx]   = buffer->v[target];
			tracer_handle->w[idx]   = buffer->w[target];
			tracer_handle->e[idx]   = buffer->e[target];
			tracer_handle->wee[idx] = buffer->wee[target];
			if (BUFFER_HAS_HID)
				tracer_handle->hid[idx] = buffer->hid[target];

			// flag & region parsing
			mcutil::UNION_FLAGS flags(buffer->flags[target]);

			ushort2 regs;
			regs.x = flags.base.region;
			regs.y = USHRT_MAX;
			tracer_handle->regions[idx] = regs;
			tracer_handle->flags[idx]   = flags.astype<unsigned int>();
		}
	}


	__host__ void __host__devicePushBulk(int block, int thread, RingBuffer* buffer) {
		__kernel__devicePushBulk <<< block, thread, mcutil::SIZE_SHARED_MEMORY_GLOBAL >>> (buffer);
	}


	__global__ void __kernel__devicePushBulk(RingBuffer* buffer) {
		unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
		int target = buffer->pushBulk();
		buffer->x[target]   = tracer_handle->x[idx];
		buffer->y[target]   = tracer_handle->y[idx];
		buffer->z[target]   = tracer_handle->z[idx];
		buffer->u[target]   = tracer_handle->u[idx];
		buffer->v[target]   = tracer_handle->v[idx];
		buffer->w[target]   = tracer_handle->w[idx];
		buffer->e[target]   = tracer_handle->e[idx];
		buffer->wee[target] = tracer_handle->wee[idx];
		if (BUFFER_HAS_HID)
			buffer->hid[target] = tracer_handle->hid[idx];

		// flag & region packing
		ushort2 regs = tracer_handle->regions[idx];
		mcutil::UNION_FLAGS flags(tracer_handle->flags[idx]);
		flags.base.region = regs.x;

		buffer->flags[target] = flags.astype<unsigned int>();
	}


	__host__ void __host__devicePushAtomic(int block, int thread, RingBuffer* buffer) {
		__kernel__devicePushAtomic <<< block, thread, mcutil::SIZE_SHARED_MEMORY_GLOBAL >>> (buffer);
	}


	__global__ void __kernel__devicePushAtomic(RingBuffer* buffer) {
		unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
		int target;
		if (tracer_handle->wee[idx] > 0.f) {
			target = buffer->pushAtomic();
			buffer->x[target]   = tracer_handle->x[idx];
			buffer->y[target]   = tracer_handle->y[idx];
			buffer->z[target]   = tracer_handle->z[idx];
			buffer->u[target]   = tracer_handle->u[idx];
			buffer->v[target]   = tracer_handle->v[idx];
			buffer->w[target]   = tracer_handle->w[idx];
			buffer->e[target]   = tracer_handle->e[idx];
			buffer->wee[target] = tracer_handle->wee[idx];
			if (BUFFER_HAS_HID)
				buffer->hid[target] = tracer_handle->hid[idx];

			// flag & region packing
			ushort2 regs = tracer_handle->regions[idx];
			mcutil::UNION_FLAGS flags(tracer_handle->flags[idx]);

			flags.base.region = regs.x;
			buffer->flags[target] = flags.astype<unsigned int>();
		}
	}


	__host__ void __host__deviceGetPhaseSpace(
		int block,
		int thread,
		RingBuffer*            buffer,
		unsigned long long int from,
		geo::PhaseSpace*       ps_dev,
		bool                   has_hid,
		bool                   using_za
	) {
		__kernel__deviceGetPhaseSpace <<< block, thread, mcutil::SIZE_SHARED_MEMORY_GLOBAL >>> (buffer, from, ps_dev, has_hid, using_za);
	}


	__global__ void __kernel__deviceGetPhaseSpace(
		RingBuffer*            buffer,
		unsigned long long int from,
		geo::PhaseSpace*       ps_dev,
		bool                   has_hid,
		bool                   using_za
	) {
		int idx        = threadIdx.x + blockDim.x * blockIdx.x;
		int buffer_idx = (from + idx) % buffer->size;
		ps_dev[idx].position.x  = buffer->x[buffer_idx];
		ps_dev[idx].position.y  = buffer->y[buffer_idx];
		ps_dev[idx].position.z  = buffer->z[buffer_idx];
		ps_dev[idx].direction.x = buffer->u[buffer_idx];
		ps_dev[idx].direction.y = buffer->v[buffer_idx];
		ps_dev[idx].direction.z = buffer->w[buffer_idx];
		ps_dev[idx].energy      = buffer->e[buffer_idx];
		ps_dev[idx].weight      = buffer->wee[buffer_idx];
		if (using_za)
			ps_dev[idx].flags   = mcutil::UNION_FLAGS(buffer->za[buffer_idx]).astype<unsigned int>();
		else
			ps_dev[idx].flags   = buffer->flags[buffer_idx];
		ps_dev[idx].hid         = has_hid ? (int)buffer->hid[buffer_idx] : -1;
	}

}