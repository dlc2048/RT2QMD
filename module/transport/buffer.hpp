#pragma once

#include <vector>
#include <cuda_runtime.h>

#include "device/tuning.hpp"
#include "device/memory_manager.hpp"
#include "device/memory.hpp"
#include "mclog/logger.hpp"
#include "particles/define.hpp"

#include "buffer.cuh"
#include "buffer_struct.hpp"


namespace mcutil {


    constexpr double IDEAL_BUFFER_RELEASE_RATIO = 0.03;
    constexpr double MAX_BUFFER_RELEASE_RATIO   = 0.1;
    constexpr double BUFFER_RELEASE_MARGIN      = 1.2;


    class DeviceBufferHandler {
    private:
        std::vector<int> _dev_malloc_flag;
        std::vector<RingBuffer> _buffer_host;
        RingBuffer* _buffer_dev;
        // for summarize
        size_t _mem_avail;
        double _mem_ratio;
        size_t _n_particle_per_buffer;  // buffer bank size
        // dimension
        size_t _block;
        size_t _thread;
        bool   _has_hid;  // History id activation flag
        // buffer priority
        int*   _priority_pinned_host;
        int*   _priority_pinned_dev;
    public:


        DeviceBufferHandler(const DeviceController& dev_prop, bool need_hid);


        ~DeviceBufferHandler();


        void summary() const;
        bool hasHid()  const { return this->_has_hid; }


        CUdeviceptr handle();
        BUFFER_TYPE getBufferPriority();

        RingBuffer* deviceptr() { return this->_buffer_dev; }

        void pullVector(BUFFER_TYPE btype);
        void pullAtomic(BUFFER_TYPE btype);
        void pushVector(BUFFER_TYPE btype);
        void pushAtomic(BUFFER_TYPE btype);


        std::vector<geo::PhaseSpace> getPhaseSpace(BUFFER_TYPE btype, bool has_hid);
        std::vector<geo::PhaseSpace> getAllPhaseSpace(bool has_hid);
        std::vector<geo::PhaseSpace> getTransportPhaseSpace(BUFFER_TYPE btype_source, bool has_hid);


        void clearTarget(BUFFER_TYPE btype);


        void clearAll();


        void clearTransportBuffer(BUFFER_TYPE btype_source);


        void setResidualStage(size_t block, double margin = BUFFER_RELEASE_MARGIN);


        void getBufferHistories();


        const RingBuffer& getHostBufferHandler(BUFFER_TYPE type);


        void result() const;


    };

}