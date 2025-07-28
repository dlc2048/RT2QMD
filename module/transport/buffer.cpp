
#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "buffer.hpp"


namespace mcutil {


    DeviceBufferHandler::DeviceBufferHandler(const DeviceController& dev_prop, bool need_hid) :
        _buffer_dev(nullptr), _has_hid(need_hid) {
        mclog::debug("Initialize global buffer system ...");
        // initialize buffer priority mirror (pinned)
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>
            (&this->_priority_pinned_host), sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>
            (&this->_priority_pinned_dev), sizeof(int)));
        // calculate activated buffer lists 
        // by geometry and physics options
        size_t buffer_size = (size_t)log2((int)BUFFER_TYPE::EOB);
        if (pow(2, buffer_size) == (int)BUFFER_TYPE::EOB)
            buffer_size = (size_t)pow(2, buffer_size);
        else
            buffer_size = (size_t)pow(2, buffer_size + 1);

        this->_dev_malloc_flag.resize(buffer_size, 0);
        // transport buffer always activated
        this->_dev_malloc_flag[BUFFER_TYPE::ELECTRON]   = 1;
        this->_dev_malloc_flag[BUFFER_TYPE::PHOTON]     = 1;
        this->_dev_malloc_flag[BUFFER_TYPE::POSITRON]   = 1;
        this->_dev_malloc_flag[BUFFER_TYPE::NEUTRON]    = 1;
        this->_dev_malloc_flag[BUFFER_TYPE::GNEUTRON]   = 1;
        this->_dev_malloc_flag[BUFFER_TYPE::GENION]     = 1;
        this->_dev_malloc_flag[BUFFER_TYPE::RELAXATION] = 1;

        // check physics

        if (Define::Electron::getInstance().activated()) {
            this->_dev_malloc_flag[BUFFER_TYPE::EBREM]  = 1;
            if (Define::Electron::getInstance().nBremSplit() > 1)  // Bremsstrahlung splitting on
                this->_dev_malloc_flag[BUFFER_TYPE::EBREM_SP] = 1;
            this->_dev_malloc_flag[BUFFER_TYPE::MOLLER] = 1;
        }

        if (Define::Photon::getInstance().activated()) {
            if (Define::Photon::getInstance().doRayleigh())
                this->_dev_malloc_flag[BUFFER_TYPE::RAYLEIGH] = 1;
            this->_dev_malloc_flag[BUFFER_TYPE::COMPTON] = 1;
            this->_dev_malloc_flag[BUFFER_TYPE::PHOTO]   = 1;
            this->_dev_malloc_flag[BUFFER_TYPE::PAIR]    = 1;
        }

        if (Define::Positron::getInstance().activated()) {
            this->_dev_malloc_flag[BUFFER_TYPE::PBREM]  = 1;
            if (Define::Electron::getInstance().nBremSplit() > 1)  // Bremsstrahlung splitting on
                this->_dev_malloc_flag[BUFFER_TYPE::PBREM_SP] = 1;
            this->_dev_malloc_flag[BUFFER_TYPE::BHABHA] = 1;
            this->_dev_malloc_flag[BUFFER_TYPE::ANNIHI] = 1;
        }

        if (Define::Neutron::getInstance().activated()) {
            this->_dev_malloc_flag[BUFFER_TYPE::NEU_SECONDARY] = 1;
        }

        if (Define::GenericIon::getInstance().activated()) {
            this->_dev_malloc_flag[BUFFER_TYPE::DELTA] = 1;
            switch (Define::IonInelastic::getInstance().modeHigh()) {
            case Define::ION_INELASTIC_METHOD_HIGH::ION_INELASTIC_HIGH_OFF:
                break;
            case Define::ION_INELASTIC_METHOD_HIGH::ION_INELASTIC_HIGH_QMD:
                this->_dev_malloc_flag[BUFFER_TYPE::ION_NUCLEAR]   = 1;
                this->_dev_malloc_flag[BUFFER_TYPE::DEEXCITATION]  = 1;
                this->_dev_malloc_flag[BUFFER_TYPE::PHOTON_EVAP]   = 1;
                if(Define::IonInelastic::getInstance().activateFission())
                    this->_dev_malloc_flag[BUFFER_TYPE::COMP_FISSION] = 1;
                if (Define::IonInelastic::getInstance().modeLow() ==
                    Define::ION_INELASTIC_METHOD_LOW::ION_INELASTIC_LOW_BME)
                    this->_dev_malloc_flag[BUFFER_TYPE::BME]           = 1;
                else
                    this->_dev_malloc_flag[BUFFER_TYPE::CN_FORMATION]  = 1;
                this->_dev_malloc_flag[BUFFER_TYPE::NUC_SECONDARY] = 1;
                break;
            case Define::ION_INELASTIC_METHOD_HIGH::ION_INELASTIC_HIGH_ABRASION:
                this->_dev_malloc_flag[BUFFER_TYPE::ION_NUCLEAR]   = 1;
                this->_dev_malloc_flag[BUFFER_TYPE::DEEXCITATION]  = 1;
                this->_dev_malloc_flag[BUFFER_TYPE::PHOTON_EVAP]   = 1;
                if (Define::IonInelastic::getInstance().activateFission())
                    this->_dev_malloc_flag[BUFFER_TYPE::COMP_FISSION] = 1;
                if (Define::IonInelastic::getInstance().modeLow() ==
                    Define::ION_INELASTIC_METHOD_LOW::ION_INELASTIC_LOW_BME)
                    this->_dev_malloc_flag[BUFFER_TYPE::BME]           = 1;
                else
                    this->_dev_malloc_flag[BUFFER_TYPE::CN_FORMATION]  = 1;
                this->_dev_malloc_flag[BUFFER_TYPE::ABRASION]      = 1;
                this->_dev_malloc_flag[BUFFER_TYPE::NUC_SECONDARY] = 1;
                break;
            default:
                assert(false);
            }
                
        }

        size_t n_device_buffer = 0;
        for (int activated : this->_dev_malloc_flag)
            if (activated) n_device_buffer++;

        // now calculate the avaliable GPU global memory
        this->_mem_avail = dev_prop.freeMemory();
        size_t mem_buffer = this->_mem_avail;
        this->_mem_ratio = dev_prop.bufferRatio();
        mem_buffer = size_t((double)mem_buffer * this->_mem_ratio);
        mem_buffer /= n_device_buffer;

        this->_block  = dev_prop.block();
        this->_thread = dev_prop.thread();

        size_t ps_size = sizeof(float) * 8 + sizeof(unsigned int);
        if (this->_has_hid)
            ps_size += sizeof(unsigned int);

        this->_n_particle_per_buffer  = mem_buffer / ps_size;
        this->_n_particle_per_buffer -= this->_n_particle_per_buffer % (_block * _thread);

        // optimize epoch
        size_t total_batch_size = dev_prop.block() * dev_prop.thread();
        if ((double)total_batch_size / (double)this->_n_particle_per_buffer > MAX_BUFFER_RELEASE_RATIO)
            mclog::fatal("Buffer release ratio over upper limit due to the GPU memory capacity");
        
        // allocate device memory
        this->_buffer_host.resize(_dev_malloc_flag.size());
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_buffer_dev),
            sizeof(RingBuffer) * _dev_malloc_flag.size()));
        for (size_t idx = 0; idx < this->_dev_malloc_flag.size(); ++idx) {
            int activated = this->_dev_malloc_flag[idx];
            RingBuffer& buffer_host = this->_buffer_host[idx];
            buffer_host.size  = (int)this->_n_particle_per_buffer;
            buffer_host.sizef = (float)buffer_host.size;
            buffer_host.head  = 
                idx ? 0 : (size_t)(total_batch_size * BUFFER_RELEASE_MARGIN);
            buffer_host.tail  = 0;

            if (activated) {  // allocate phase space memory
                float** dest[8] = {
                &buffer_host.x,   &buffer_host.y,   &buffer_host.z,
                &buffer_host.u,   &buffer_host.v,   &buffer_host.w,
                &buffer_host.e,   &buffer_host.wee
                };
                for (size_t di = 0; di < 8; ++di) {
                    float* soa;
                    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&soa),
                        sizeof(float) * this->_n_particle_per_buffer));
                    CUDA_CHECK(cudaMemcpy(dest[di], &soa,
                        sizeof(float*), cudaMemcpyHostToHost));
                }
                unsigned int* soai;
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&soai),
                    sizeof(unsigned int) * this->_n_particle_per_buffer));
                CUDA_CHECK(cudaMemcpy(&buffer_host.flags, &soai,
                    sizeof(unsigned int*), cudaMemcpyHostToHost));
                // HID
                if (this->_has_hid) {
                    unsigned int* hidi;
                    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hidi),
                        sizeof(unsigned int) * this->_n_particle_per_buffer));
                    CUDA_CHECK(cudaMemcpy(&buffer_host.hid, &hidi,
                        sizeof(unsigned int*), cudaMemcpyHostToHost));
                }
                // ZA
                switch (idx) {
                case BUFFER_TYPE::BME:
                case BUFFER_TYPE::ABRASION:
                    uchar4* za;
                    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&za),
                        sizeof(uchar4) * this->_n_particle_per_buffer));
                    CUDA_CHECK(cudaMemcpy(&buffer_host.za, &za,
                        sizeof(uchar4*), cudaMemcpyHostToHost));
                    break;
                default:
                    break;
                }
            }
            CUDA_CHECK(cudaMemcpy(&this->_buffer_dev[idx], &buffer_host,
                sizeof(RingBuffer), cudaMemcpyHostToDevice));
        }
        // set HID symbol
        CUDA_CHECK(setHIDFlag(this->_has_hid));
    }


    DeviceBufferHandler::~DeviceBufferHandler() {
        mclog::debug("Destroy device memory of global buffer system ...");
        CUDA_CHECK(cudaFreeHost(this->_priority_pinned_host));
        CUDA_CHECK(cudaFree(this->_priority_pinned_dev));
        for (size_t idx = 0; idx < this->_dev_malloc_flag.size(); ++idx) {
            int activated = this->_dev_malloc_flag[idx];
            if (!activated) continue;
            RingBuffer buffer_host;
            CUDA_CHECK(cudaMemcpy(&buffer_host, &this->_buffer_dev[idx],
                sizeof(RingBuffer), cudaMemcpyDeviceToHost));
            float** dest[8] = {
                &buffer_host.x, &buffer_host.y, &buffer_host.z,
                &buffer_host.u, &buffer_host.v, &buffer_host.w,
                &buffer_host.e, &buffer_host.wee
            };
            for (size_t di = 0; di < 8; ++di)
                CUDA_CHECK(cudaFree(*dest[di]));
            CUDA_CHECK(cudaFree(buffer_host.flags));
            if (this->_has_hid)
                CUDA_CHECK(cudaFree(buffer_host.hid));
        }
        CUDA_CHECK(cudaFree(this->_buffer_dev));
    }


    void DeviceBufferHandler::summary() const {
        mclog::info("*** Buffer Summaries ***");
        mclog::printVar("Available memories", (double)this->_mem_avail / (double)mcutil::MEMSIZE_MIB, "MiB");
        mclog::printVar("Memory usage ratio", this->_mem_ratio * 100.0, "%");
        size_t n_buffer = 0;
        for (int activated : this->_dev_malloc_flag)
            if (activated) n_buffer++;
        mclog::printVar("Number of buffer", n_buffer);
        mclog::printVar("Size per buffer", this->_n_particle_per_buffer, "Banks");
        mclog::print("");
    }


    CUdeviceptr DeviceBufferHandler::handle() {
        return reinterpret_cast<CUdeviceptr>(this->_buffer_dev);
    }


    BUFFER_TYPE DeviceBufferHandler::getBufferPriority() {
        __host__deviceGetBufferPriority(
            this->_dev_malloc_flag.size(), 
            this->_buffer_dev, 
            this->_priority_pinned_dev
        );
        cudaMemcpy(this->_priority_pinned_host, this->_priority_pinned_dev,
            sizeof(int), cudaMemcpyDeviceToHost);
        return static_cast<BUFFER_TYPE>(this->_priority_pinned_host[0]);
    }


    void DeviceBufferHandler::pullVector(BUFFER_TYPE btype) {
        __host__devicePullBulk((int)this->_block, (int)this->_thread, &this->_buffer_dev[btype]);
    }


    void DeviceBufferHandler::pullAtomic(BUFFER_TYPE btype) {
        __host__devicePullAtomic((int)this->_block, (int)this->_thread, &this->_buffer_dev[btype]);
    }


    void DeviceBufferHandler::pushVector(BUFFER_TYPE btype) {
        __host__devicePushBulk((int)this->_block, (int)this->_thread, &this->_buffer_dev[btype]);
    }


    void DeviceBufferHandler::pushAtomic(BUFFER_TYPE btype) {
        __host__devicePushAtomic((int)this->_block, (int)this->_thread, &this->_buffer_dev[btype]);
    }


    std::vector<geo::PhaseSpace> DeviceBufferHandler::getPhaseSpace(BUFFER_TYPE btype, bool has_hid) {
        std::vector<geo::PhaseSpace> ps_host;
        RingBuffer buffer_host;

        if (!this->_dev_malloc_flag[btype])
            return ps_host;

        CUDA_CHECK(cudaMemcpy(&buffer_host, &this->_buffer_dev[btype], 
            sizeof(RingBuffer), cudaMemcpyDeviceToHost));

        unsigned long long int from = buffer_host.tail;
        unsigned long long int to   = buffer_host.head;
        unsigned long long int size = this->_block * this->_thread;

        geo::PhaseSpace* ps_dev;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ps_dev),
            sizeof(geo::PhaseSpace) * this->_block * this->_thread));
        for (size_t i = from; i < to; i += size) {
            size_t seg_length = std::min(size, to - i);
            std::vector<geo::PhaseSpace> ps_host_seg(seg_length);
            bool using_za = btype == BUFFER_TYPE::ABRASION || btype == BUFFER_TYPE::BME;
            __host__deviceGetPhaseSpace((int)this->_block, (int)this->_thread, &this->_buffer_dev[btype], i, ps_dev, has_hid, using_za);
            CUDA_CHECK(cudaMemcpy(&ps_host_seg[0], ps_dev,
                sizeof(geo::PhaseSpace) * seg_length, cudaMemcpyDeviceToHost));
            ps_host.insert(ps_host.end(), ps_host_seg.begin(), ps_host_seg.end());
        }
        
        CUDA_CHECK(cudaFree(ps_dev));

        for (size_t i = 0; i < ps_host.size(); ++i)
            ps_host[i].type = static_cast<int>(btype);

        const std::vector<float>& ngroup = Define::Neutron::getInstance().ngroup();

        // convert neutron group -> energy
        if (btype == BUFFER_TYPE::NEUTRON) {
            for (auto& seg : ps_host) {
                mcutil::UNION_FLAGS flags(seg.energy);
                int group  = flags.neutron_p.group;
                seg.energy = (ngroup[group] + ngroup[group + 1]) * 0.5f;
            }
        }
        return ps_host;
    }


    std::vector<geo::PhaseSpace> DeviceBufferHandler::getAllPhaseSpace(bool has_hid) {
        std::vector<geo::PhaseSpace> ps_host;
        for (size_t i = 0; i < this->_dev_malloc_flag.size(); ++i) {
            std::vector<geo::PhaseSpace> ps_host_seg = this->getPhaseSpace(static_cast<mcutil::BUFFER_TYPE>(i), has_hid);
            ps_host.insert(ps_host.end(), ps_host_seg.begin(),  ps_host_seg.end());
        }
        return ps_host;
    }


    std::vector<geo::PhaseSpace> DeviceBufferHandler::getTransportPhaseSpace(BUFFER_TYPE btype_source, bool has_hid) {
        std::vector<geo::PhaseSpace> ps_host;
        for (size_t i = 0; i < BUFFER_TYPE::RELAXATION; ++i) {
            if (i == btype_source)
                continue;
            std::vector<geo::PhaseSpace> ps_host_seg = this->getPhaseSpace(static_cast<mcutil::BUFFER_TYPE>(i), has_hid);
            ps_host.insert(ps_host.end(), ps_host_seg.begin(), ps_host_seg.end());
        }
        return ps_host;
    }


    void DeviceBufferHandler::clearTarget(BUFFER_TYPE btype) {
        RingBuffer buffer_host;
        CUDA_CHECK(cudaMemcpy(&buffer_host, &this->_buffer_dev[btype],
            sizeof(RingBuffer), cudaMemcpyDeviceToHost));
        if (btype != BUFFER_TYPE::SOURCE)
            buffer_host.tail = buffer_host.head;
        CUDA_CHECK(cudaMemcpy(&this->_buffer_dev[btype], &buffer_host,
            sizeof(RingBuffer), cudaMemcpyHostToDevice));
    }


    void DeviceBufferHandler::clearAll() {
        for (size_t idx = 0; idx < this->_dev_malloc_flag.size(); ++idx) {
            RingBuffer& buffer_host = this->_buffer_host[idx];
            if (idx)
                buffer_host.head = 0;
            buffer_host.tail = 0;

            CUDA_CHECK(cudaMemcpy(&this->_buffer_dev[idx], &buffer_host,
                sizeof(RingBuffer), cudaMemcpyHostToDevice));
        }
    }


    void DeviceBufferHandler::clearTransportBuffer(BUFFER_TYPE btype_source) {
        for (size_t idx = 0; idx < BUFFER_TYPE::RELAXATION; ++idx) {
            if (idx == btype_source)
                continue;
            clearTarget(static_cast<BUFFER_TYPE>(idx));
        }
            
    }


    void DeviceBufferHandler::setResidualStage(size_t res_block, double margin) {
        RingBuffer& buffer_source = this->_buffer_host[0];
        this->_block = res_block;
        size_t total_batch_size = this->_block * this->_thread;
        buffer_source.head = (size_t)(total_batch_size * margin);
        
        CUDA_CHECK(cudaMemcpy(&this->_buffer_dev[0], &buffer_source,
            sizeof(RingBuffer), cudaMemcpyHostToDevice));
    }


    void DeviceBufferHandler::getBufferHistories() {
        // copy result from device
        for (size_t idx = 0; idx < this->_dev_malloc_flag.size(); ++idx) {
            RingBuffer& buffer_host = this->_buffer_host[idx];
            CUDA_CHECK(cudaMemcpy(&buffer_host, &this->_buffer_dev[idx],
                sizeof(RingBuffer), cudaMemcpyDeviceToHost));
        }
    }


    const RingBuffer& DeviceBufferHandler::getHostBufferHandler(BUFFER_TYPE type) {
        return this->_buffer_host[type];
    }


    void DeviceBufferHandler::result() const {
        mclog::info("*** Kernel Execution Histories ***");
        for (size_t idx = 0; idx < this->_dev_malloc_flag.size(); ++idx) {
            const RingBuffer& buffer_host = this->_buffer_host[idx];
            int activated = this->_dev_malloc_flag[idx];
            if (!activated) continue;
            BUFFER_TYPE bid = static_cast<BUFFER_TYPE>(idx);
            std::map<BUFFER_TYPE, std::string>::const_iterator iter
                = getBidName().find(bid);
            if (iter == getBidName().end()) continue;
            mclog::printVar(iter->second, (size_t)buffer_host.tail, "Banks");
        }
        mclog::info("*** Residual Banks ***");
        for (size_t idx = 0; idx < this->_dev_malloc_flag.size(); ++idx) {
            const RingBuffer& buffer_host = this->_buffer_host[idx];
            int activated = this->_dev_malloc_flag[idx];
            if (!activated) continue;
            BUFFER_TYPE bid = static_cast<BUFFER_TYPE>(idx);
            std::map<BUFFER_TYPE, std::string>::const_iterator iter
                = getBidName().find(bid);
            if (iter == getBidName().end()) continue;
            mclog::printVar(iter->second, (size_t)(buffer_host.head - buffer_host.tail), "Banks");
        }
    }

}