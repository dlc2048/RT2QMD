//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

/**
 * @file    mcutil/device/algorithm.hpp
 * @brief   Part of the exception handler in OptiX 7 library
 * @author  NVIDIA Corporation
 * @date    
 */

#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <sstream>
#include <stdexcept>

namespace mcutil {


    class Exception : public std::runtime_error {
    public:
        Exception(const char* msg)
            : std::runtime_error(msg)
        {
        }
    };


    inline void cudaCheck(cudaError_t error, const char* call, const char* file, unsigned int line)
    {
        if (error != cudaSuccess)
        {
            std::stringstream ss;
            ss << "CUDA call (" << call << " ) failed with error: '"
                << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";
            throw Exception(ss.str().c_str());
        }
    }


}


#define CUDA_CHECK( call ) mcutil::cudaCheck( call, #call, __FILE__, __LINE__ )