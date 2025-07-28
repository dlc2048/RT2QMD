//
// Copyright (C) 2025 CM Lee, SJ Ye, Seoul Sational University
//
// Licensed to the Apache Software Foundation(ASF) under one
// or more contributor license agreements.See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// 	"License"); you may not use this file except in compliance
// 	with the License.You may obtain a copy of the License at
// 
// 	http ://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.See the License for the
// specific language governing permissionsand limitations
// under the License.

/**
 * @file    mcutil/device/memory.hpp
 * @brief   Vector memory managing
 * @author  CM Lee
 * @date    05/23/2023
 */

#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <exception>

#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "memory_manager.hpp"



#define M_MacroXSAoSMapper(type, cbegin, cend, member_alias) {                       \
    std::vector<type> member_list;                                                    \
    std::transform(cbegin, cend, std::back_inserter(member_list),                     \
        [&](const auto& data) { return data.member_alias; });                         \
    this->_deviceptr_lists.push_back(DeviceVectorHelper(member_list).deviceptr());    \
}



namespace mcutil {


    constexpr size_t MEMSIZE_KIB = 1024;
    constexpr size_t MEMSIZE_MIB = MEMSIZE_KIB * MEMSIZE_KIB;


    /**
    * @brief Static cast the std::vector<double> to std::vector<float>
    * @param arr Vector<double>
    * 
    * @return Vector<float>
    */
    std::vector<float>  cvtVectorDoubleToFloat(const std::vector<double>& arr);


    /**
    * @brief Static cast the std::vector<float> to std::vector<double>
    * @param arr Vector<float>
    *
    * @return Vector<double>
    */
    std::vector<double> cvtVectorFloatToDouble(const std::vector<float>& arr);


    /**
    * @brief Static cast the std::vector<double2> to std::vector<float2>
    * @param arr Vector<double2>
    *
    * @return Vector<float2>
    */
    std::vector<float2> cvtVectorDoubleToFloat(const std::vector<double2>& arr);


    /**
    * @brief Static cast the std::vector<double> to std::vector<double2>
    * @param arr Vector<double>
    * 
    * @return Vector<double2>
    */
    std::vector<double2> cvtVectorDoubleToDouble2(const std::vector<double>& arr);


    /**
    * @brief Copy all containing data from host vector to device memory
    * @param host   Host-side vector
    * @param device Pointer of device-side array. It must not be allocated before
    * 
    * @return Memory allocation size [bytes]
    */
    template <typename T>
    size_t cudaMemcpyVectorToDevice(const std::vector<T>& host, T** device);


    /**
    * @brief EGSnrc style log-log linear interpolation vector
    */
    template <typename T, typename T2>
    class InterpVectorInterface {
    private:
        T2              _c;  //! @brief {a, b} constants pair in equation llx = (int)(a + b * lx)
        std::vector<T2> _xy;  //! @brief Interpolation coeff pair


        T2 _setCoeff(T xmin, T xmax, size_t n);


    public:


        InterpVectorInterface() {};


        InterpVectorInterface(T2 c, size_t n);


        InterpVectorInterface(T xmin, T xmax, size_t n);


        InterpVectorInterface(T2 c, const std::vector<T>& values);


        InterpVectorInterface(T xmin, T xmax, const std::vector<T>& values);


        InterpVectorInterface(T2 c, const std::vector<T2>& llpoints);


        InterpVectorInterface(T xmin, T xmax, const std::vector<T2>& llpoints);


        T2 coeffs() const { return this->_c; }


        size_t llx(T lx) const;

        
        std::vector<T2>& llpoints() { return this->_xy; }


        const std::vector<T2>& llpoints() const { return this->_xy; }


        size_t size() const { return this->_xy.size() - 1; }


        T get(T lx) const;


        T get(T lx, size_t llx) const;


    };


    typedef InterpVectorInterface<double, double2> InterpVectorDouble;
    typedef InterpVectorInterface<float,  float2>  InterpVectorFloat;


    template <class ...Ts>
    struct void_t {
        using type = void;
    };


    template <class T, class = void>
    struct has_nested_device_memory_type : std::false_type {};


    template <class T>
    struct has_nested_device_memory_type<T, typename void_t<decltype(std::declval<T>().free())>::type> : std::true_type {};


    template <class T>
    class DeviceVectorHelperBase : public DeviceMemoryHandlerInterface {
    protected:
        size_t _length;
        T*     _deviceptr;


        DeviceVectorHelperBase() :
            _length(0x0u), _deviceptr(nullptr) {}


        void _freeBase();


    public:


        size_t      size()      const { return this->_length; }
        T*          address()   const { return this->_deviceptr;}
        CUdeviceptr deviceptr() const { return reinterpret_cast<CUdeviceptr>(this->_deviceptr); }

    };


    template <typename T, typename Enable = void>
    class DeviceVectorHelper : public DeviceVectorHelperBase<T> {
    public:


        DeviceVectorHelper() {}
            

        DeviceVectorHelper(const std::vector<T>& host_vector);


        DeviceVectorHelper(T* dev_ptr);


        void free(size_t size = 0x0u);


    };


    template <typename T>
    class DeviceVectorHelper<T, typename std::enable_if<has_nested_device_memory_type<T>::value>::type> : public DeviceVectorHelperBase<T> {
    public:


        DeviceVectorHelper() {}


        DeviceVectorHelper(const std::vector<T>& vector_host_member_dev);


        DeviceVectorHelper(T* dev_ptr);


        void free(size_t size);


    };


    template <typename T>
    class DeviceVectorHelper<T, typename std::enable_if<std::is_pointer<T>::value>::type> : public DeviceVectorHelperBase<T> {
    public:


        DeviceVectorHelper() {}


        DeviceVectorHelper(const std::vector<T>& vector_host_member_dev);


        DeviceVectorHelper(T* dev_ptr);


        void free(size_t size);


    };


    template <typename T>
    class HostSoABase : public DeviceMemoryHandlerInterface {
    protected:
        std::vector<CUdeviceptr> _deviceptr_lists;


    public:


        HostSoABase(const std::vector<T>& vector_host_member_dev);


        std::vector<CUdeviceptr>& deviceptrLists() { return this->_deviceptr_lists; }


    };


}


#include "memory.tpp"
