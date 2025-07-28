#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>

#include "device/memory.cuh"
#include "device/tuning.cuh"

#include "transport.cuh"
#include "buffer_struct.hpp"


namespace mcutil {


	constexpr int SHARED_OFFSET_BUFFER_REDUX = 2;  // { leading thread index, size of buffer }


	typedef struct FLAGS_DEFAULT {
		unsigned short region : 16;
		unsigned short fmask  : 16;
	} FLAGS_DEFAULT;


	typedef struct FLAGS_NEUTRON_P {
		unsigned short region : 16;
		short          group  : 11;
		unsigned short fmask  : 5;
	} FLAGS_NEUTRON_P;


	// Neutron, extra
	typedef struct FLAGS_NEUTRON_S {
		unsigned short iso_idx : 16;
		unsigned char  rea_idx : 8;
		unsigned char  sec_pos : 8;
	} FLAGS_NEUTRON_S;


	// Generic ion
	typedef struct FLAGS_GENION {
		unsigned short region  : 16;
		unsigned char  fmask   : 8;
		unsigned char  ion_idx : 8;
	} FLAGS_GENION;


	// De-excitation
	typedef struct FLAGS_DEEX {
		unsigned short region : 16;
		unsigned char  z      : 8;
		unsigned char  a      : 8;
	} FLAGS_DEEX;


	// INCL
	typedef struct FLAGS_INCL {
		unsigned short region     : 16;
		unsigned char  target_idx : 8;
		unsigned char  proj_idx   : 8;  // GENION ion_idx -> projectile
	} FLAGS_INCL;


	// QMD
	typedef struct FLAGS_QMD {
		unsigned short region : 16;
		unsigned short phase  : 8;
		unsigned short fmask  : 8;
	} FLAGS_QMD;


	// union

	typedef struct UNION_FLAGS {
		union {
			unsigned int    origin;
			FLAGS_DEFAULT   base;
			FLAGS_NEUTRON_P neutron_p;
			FLAGS_NEUTRON_S neutron_s;
			FLAGS_GENION    genion;
			FLAGS_DEEX      deex;
			FLAGS_INCL      incl;
			FLAGS_QMD       qmd;
		};


		__host__ __device__ UNION_FLAGS(UNION_FLAGS& rhs)
			: origin(rhs.origin) {}


		__host__ __device__ UNION_FLAGS(const UNION_FLAGS& rhs)
			: origin(rhs.origin) {}


		template <typename T>
		__host__ __device__ explicit UNION_FLAGS(T origin) 
			: origin(reinterpret_cast<unsigned int&>(origin)) {}


		__host__ __device__ explicit UNION_FLAGS(void)
			: UNION_FLAGS(0x00000000u) {}


		template <typename T>
		__inline__ __host__ __device__ T astype() {
			return reinterpret_cast<T&>(this->origin);
		}


		__inline__ __host__ __device__ unsigned int astype() {
			return this->origin;
		}


	} UNION_FLAGS;


	template <>
	__inline__ __host__ __device__ unsigned int UNION_FLAGS::astype<unsigned int>() {
		return this->origin;
	}


	typedef struct RingBuffer {
		int   size;   //! @brief Buffer size (int)
		float sizef;  //! @brief Buffer size (float)
		unsigned long long int head;
		unsigned long long int tail;

		// SOA data
		float* x;    //! @brief Particle x position (cm)
		float* y;    //! @brief Particle y position (cm)
		float* z;    //! @brief Particle z position (cm)
		float* u;    //! @brief Particle x direction
		float* v;    //! @brief Particle y direction
		float* w;    //! @brief Particle z direction
		float* e;    //! @brief Particle kinetic energy (MeV)
		float* wee;  //! @brief Particle weight
		unsigned int* flags;  //! @brief Current region index
		unsigned int* hid;    //! @brief History id
		uchar4* za;  //! @brief Particle ZA number (used in ion-inelastic)

		__inline__ __device__ float getBufferSaturation();
		__inline__ __device__ int   pushBulk();
		__inline__ __device__ int   pushAtomic();
		__inline__ __device__ int   pushAtomicWarp(int size);
		__inline__ __device__ void  pushShared(int size, int shared_offset = 0);
		__inline__ __device__ int   pullBulk();
		__inline__ __device__ int   pullBulkWarp();
		__inline__ __device__ int   pullAtomic();
		__inline__ __device__ void  pullShared(int size, int shared_offset = 0);
	} RingBuffer;


	typedef struct LinearBuffer {
		int   size;   //! @brief Buffer size (int)
		float sizef;  //! @brief Buffer size (float)
		int   head;

		// SOA data
		int*   pid;  //! @brief Particle id
		float* x;    //! @brief Particle x position (cm)
		float* y;    //! @brief Particle y position (cm)
		float* z;    //!@brief Particle z position (cm)
		float* u;    //! @brief Particle x direction
		float* v;    //! @brief Particle y direction
		float* w;    //! @brief Particle z direction
		float* e;    //! @brief Particle kinetic energy (MeV)
		float* wee;  //! @brief Particle weight

		__inline__ __device__ float getBufferSaturation();
		__inline__ __device__ int   pushAtomic();

	} LinearBuffer;

	extern __device__   geo::DeviceTracerHandle* tracer_handle;
	extern __constant__ bool BUFFER_HAS_HID;


	__host__ void __host__deviceGetBufferPriority(size_t thread, RingBuffer* buffer, int* target);
	__global__ void __device__deviceGetBufferPriority(RingBuffer* buffer, int* target);

	__host__ cudaError_t setBufferTracerHandle(CUdeviceptr handle);
	__host__ cudaError_t setHIDFlag(bool has_hid);
	__host__ void initializeTracerHandle(int block, int thread);
	__global__ void __kernel__deviceInitializeTracerHandle();

	__host__ void __host__devicePullBulk(int block, int thread, RingBuffer* buffer);
	__global__ void __kernel__devicePullBulk(RingBuffer* buffer);
	__host__ void __host__devicePullAtomic(int block, int thread, RingBuffer* buffer);
	__global__ void __kernel__devicePullAtomic(RingBuffer* buffer);
	__host__ void __host__devicePushBulk(int block, int thread, RingBuffer* buffer);
	__global__ void __kernel__devicePushBulk(RingBuffer* buffer);
	__host__ void __host__devicePushAtomic(int block, int thread, RingBuffer* buffer);
	__global__ void __kernel__devicePushAtomic(RingBuffer* buffer);

	__host__ void __host__deviceGetPhaseSpace(
		int block,
		int thread,
		RingBuffer* buffer,
		unsigned long long int from,
		geo::PhaseSpace* ps_dev,
		bool has_hid,
		bool using_za
	);
	__global__ void __kernel__deviceGetPhaseSpace(
		RingBuffer* buffer,
		unsigned long long int from,
		geo::PhaseSpace* ps_dev,
		bool has_hid,
		bool using_za
	);


#ifdef __CUDACC__


	__device__ float RingBuffer::getBufferSaturation() {
		int usage = (int)(this->head - this->tail);
		return (float)usage / this->sizef;
	}


	__device__ int RingBuffer::pushBulk() {
		unsigned long long int* head_univ
			= reinterpret_cast<unsigned long long int*>(mcutil::cache_univ);
		if (!threadIdx.x)
			head_univ[0] = atomicAdd(&this->head, blockDim.x);
		__syncthreads();
		return (head_univ[0] + threadIdx.x) % this->size;
	}


	__device__ int RingBuffer::pushAtomic() {
		return atomicAdd(&this->head, 1) % this->size;
	}


	__device__ int RingBuffer::pushAtomicWarp(int size) {
		return atomicAdd(&this->head, size) % this->size;
	}


	__device__ void RingBuffer::pushShared(int size, int shared_offset) {
		unsigned long long int _head;
		int* cache_univ_i = reinterpret_cast<int*>(mcutil::cache_univ);
		if (!threadIdx.x) {
			_head  = atomicAdd(&this->head, size);
			cache_univ_i[shared_offset]     = (int)(_head % this->size);
			cache_univ_i[shared_offset + 1] = this->size;
		}
		__syncthreads();
	}


	__device__ int RingBuffer::pullBulk() {
		unsigned long long int* tail_univ
			= reinterpret_cast<unsigned long long int*>(mcutil::cache_univ);
		if (!threadIdx.x)
			tail_univ[0] = atomicAdd(&this->tail, blockDim.x);
		__syncthreads();
		return (tail_univ[0] + threadIdx.x) % this->size;
	}


	__device__ int RingBuffer::pullAtomic() {
		return atomicAdd(&this->tail, 1) % this->size;
	}


	__device__ void RingBuffer::pullShared(int size, int shared_offset) {
		unsigned long long int _tail;
		int* cache_univ_i = reinterpret_cast<int*>(mcutil::cache_univ);
		if (!threadIdx.x) {
			_tail = atomicAdd(&this->tail, size);
			cache_univ_i[shared_offset]     = (int)(_tail % this->size);
			cache_univ_i[shared_offset + 1] = this->size;
		}
		__syncthreads();
	}


	__device__ float LinearBuffer::getBufferSaturation() {
		return (float)this->head / this->sizef;
	}


	__device__ int LinearBuffer::pushAtomic() {
		return atomicAdd(&this->head, 1);
	}


#endif


}
