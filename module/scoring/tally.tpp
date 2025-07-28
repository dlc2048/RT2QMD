#pragma once


namespace tally {


    template <typename T>
    std::function<void(T*)> Deleter() {
        return [](T* ptr) { mcutil::DeviceVectorHelper(ptr).free(1); };
    }


    template <typename HS, typename DS>
    TallyHostDevicePair<HS, DS>::TallyHostDevicePair(const HS& host) {
        // host shared ptr
        this->_host = std::make_shared<HS>(host);
    }

    template <typename HS, typename DS>
    TallyHostDevicePair<HS, DS>::TallyHostDevicePair(std::shared_ptr<HS>& host) {
        this->_host = host;
    }


    template <typename HS, typename DS>
    size_t TallyHostDevicePair<HS, DS>::initDevice() {
        // deviceptr
        DS* dptr = nullptr;
        size_t memsize = this->_host->mallocDeviceStruct(&dptr);
        this->_device  = std::shared_ptr<DS>(dptr, Deleter<DS>());
        return memsize;
    }


    template <typename HS, typename DS>
    void TallyHostDevicePair<HS, DS>::resetDeviceData() {
        initDeviceStruct(this->_device.get());
    }


    template <typename HS, typename DS>
    void TallyHostDevicePair<HS, DS>::pullFromDevice(double total_weight) {
        this->_host->memcpyFromDeviceStruct(this->_device.get());
        this->_host->normalize(total_weight);
    }


    template <typename HS, typename DS>
    void TallyHostDevicePair<HS, DS>::_pullFromDeviceMesh(double total_weight) {
        if (this->_host->memcpyPolicy() == MESH_MEMCPY_POLICY::MEMCPY_HOST) {
            this->_host->memcpyFromDeviceStruct(this->_device.get());
            this->_host->normalize(total_weight);
        }
        else {
            this->_host->normalizeAndMecpyFromKernel(this->_device.get(), total_weight);
            switch (this->_host->memoryStructure()) {
            case MESH_MEMORY_STRUCTURE::MEMORY_DENSE:
                break;
            case MESH_MEMORY_STRUCTURE::MEMORY_SPARSE_COO:
                this->_host->generateSparseCOO();
                break;
            default:
                assert(false);
            }
        }
    }


}