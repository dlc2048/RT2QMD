#pragma once


namespace mcutil {

    template <class T, unsigned I, unsigned J>
    using Matrix2d = std::array<std::array<T, J>, I>;


    template <class T, unsigned I, unsigned J, unsigned K>
    using Matrix3d = std::array<std::array<std::array<T, K>, J>, I>;


    template <class T, unsigned I, unsigned J, unsigned K, unsigned L>
    using Matrix4d = std::array<std::array<std::array<std::array<T, L>, K>, J>, I>;


    template <typename T>
    size_t cudaMemcpyVectorToDevice(const std::vector<T>& host, T** device) {
        if (!host.empty()) {
            T* vector_dev;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&vector_dev),
                sizeof(T) * host.size()));
            CUDA_CHECK(cudaMemcpy(vector_dev, &host[0],
                sizeof(T) * host.size(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(device, &vector_dev,
                sizeof(T*), cudaMemcpyHostToHost));
        }
        return sizeof(T) * host.size();
    }


    template <typename T, typename T2>
    T2 InterpVectorInterface<T, T2>::_setCoeff(T xmin, T xmax, size_t n) {
        T2 c;
        c.y = (T)n / std::log(xmax / xmin);
        c.x = -c.y * std::log(xmin);
        return c;
    }


    template <typename T, typename T2>
    InterpVectorInterface<T, T2>::InterpVectorInterface(T2 c, size_t n) 
        : _c(c) {
        this->_xy.resize(n + 1);
    }


    template <typename T, typename T2>
    InterpVectorInterface<T, T2>::InterpVectorInterface(T xmin, T xmax, size_t n)
        : InterpVectorInterface(_setCoeff(xmin, xmax, n), n) {}


    template <typename T, typename T2>
    InterpVectorInterface<T, T2>::InterpVectorInterface(T2 c, const std::vector<T>& values) 
        : InterpVectorInterface(c, values.size() - 1) {
        T val_old = 0.0;
        for (size_t llx = 0; llx < values.size(); ++llx) {
            T lx  = (T)(llx - this->_c.x) / this->_c.y;
            T val = values[llx];
            if (llx) {
                this->_xy[llx - 1].y = (val - val_old) * this->_c.y;
                this->_xy[llx - 1].x = val - this->_xy[llx - 1].y * lx;
            }
            val_old = val;
        }
        // last element
        size_t n = this->size();
        this->_xy[n] = this->_xy[n - 1];
    }


    template <typename T, typename T2>
    InterpVectorInterface<T, T2>::InterpVectorInterface(T xmin, T xmax, const std::vector<T>& values)
        : InterpVectorInterface(_setCoeff(xmin, xmax, values.size()), values) {}


    template <typename T, typename T2>
    InterpVectorInterface<T, T2>::InterpVectorInterface(T2 c, const std::vector<T2>& llpoints)
        : _c(c), _xy(llpoints) {}

    /*
    template <typename T, typename T2>
    InterpVectorInterface<T, T2>::InterpVectorInterface(T xmin, T xmax, const std::vector<T2>& llpoints)
        : InterpVectorInterface(_setCoeff(xmin, xmax, values.size()), llpoints) {}
    */

    template <typename T, typename T2>
    size_t InterpVectorInterface<T, T2>::llx(T lx) const {
        size_t llx = (size_t)(this->_c.x + lx * this->_c.y);
        assert(llx < this->_xy.size());
        return llx;
    }


    template <typename T, typename T2>
    T InterpVectorInterface<T, T2>::get(T lx) const {
        size_t llx = this->llx(lx);
        return this->get(lx, llx);
    }


    template <typename T, typename T2>
    T InterpVectorInterface<T, T2>::get(T lx, size_t llx) const {
        assert(llx < this->_xy.size());
        return this->_xy[llx].x + this->_xy[llx].y * lx;
    }


    template <typename T>
    void DeviceVectorHelperBase<T>::_freeBase() {
        if (this->_deviceptr != nullptr)
            CUDA_CHECK(cudaFree(this->_deviceptr));
    }


    template <typename T, typename Enable>
    DeviceVectorHelper<T, Enable>::DeviceVectorHelper(const std::vector<T>& host_vector) {
        this->_length = host_vector.size();
        this->_memoryUsageAppend(mcutil::cudaMemcpyVectorToDevice<T>(host_vector, &this->_deviceptr));
    }


    template <typename T, typename Enable>
    DeviceVectorHelper<T, Enable>::DeviceVectorHelper(T* dev_ptr) {
        this->_deviceptr = dev_ptr;
    }


    template <typename T, typename Enable>
    void DeviceVectorHelper<T, Enable>::free(size_t size) {
        this->_freeBase();
    }


    template <typename T>
    DeviceVectorHelper<T, typename std::enable_if<has_nested_device_memory_type<T>::value>::type>::DeviceVectorHelper(const std::vector<T>& vector_host_member_dev) {
        this->_length = vector_host_member_dev.size();
        this->_memoryUsageAppend(mcutil::cudaMemcpyVectorToDevice<T>(vector_host_member_dev, &this->_deviceptr));
    }


    template <typename T>
    DeviceVectorHelper<T, typename std::enable_if<has_nested_device_memory_type<T>::value>::type>::DeviceVectorHelper(T* dev_ptr) {
        this->_deviceptr = dev_ptr;
    }


    template <typename T>
    void DeviceVectorHelper<T, typename std::enable_if<has_nested_device_memory_type<T>::value>::type>::free(size_t size) {
        if (!size)
            return;
        std::vector<T> vector_host_member_dev(size);
        CUDA_CHECK(cudaMemcpy(&vector_host_member_dev[0], this->_deviceptr,
            sizeof(T) * size, cudaMemcpyDeviceToHost));
        for (T& shmd : vector_host_member_dev)
            shmd.free();
        this->_freeBase();
    }


    template <typename T>
    DeviceVectorHelper<T, typename std::enable_if<std::is_pointer<T>::value>::type>::DeviceVectorHelper(const std::vector<T>& vector_host_member_dev) {
        this->_length = vector_host_member_dev.size();
        this->_memoryUsageAppend(mcutil::cudaMemcpyVectorToDevice<T>(vector_host_member_dev, &this->_deviceptr));
    }


    template <typename T>
    DeviceVectorHelper<T, typename std::enable_if<std::is_pointer<T>::value>::type>::DeviceVectorHelper(T* dev_ptr) {
        this->_deviceptr = dev_ptr;
    }


    template <typename T>
    void  DeviceVectorHelper<T, typename std::enable_if<std::is_pointer<T>::value>::type>::free(size_t size) {
        std::vector<T> nested_ptr(size);
        CUDA_CHECK(cudaMemcpy(&nested_ptr[0], this->_deviceptr,
            sizeof(T) * size, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < nested_ptr.size(); ++i)
            CUDA_CHECK(cudaFree(nested_ptr[i]));
        this->_freeBase();
    }


}
