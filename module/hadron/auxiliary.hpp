/**
 * @file    module/hadron/auxiliary.hpp
 * @brief   ENSDF data table
 * @author  CM Lee
 * @date    06/14/2024
 */

#pragma once

#include <filesystem>
#include <map>

#include <cuda_runtime.h>

#include "mclog/logger.hpp"
#include "singleton/singleton.hpp"
#include "device/memory_manager.hpp"
#include "device/memory.hpp"
#include "prompt/env.hpp"

#include "auxiliary.cuh"


namespace Hadron {


    inline const std::filesystem::path HOME = std::filesystem::path("resource/hadron");


    class NNScatteringTableHandler :
        public Singleton<NNScatteringTableHandler>,
        public mcutil::DeviceMemoryHandlerInterface {
        friend class Singleton<NNScatteringTableHandler>;
    private:
        static const std::filesystem::path _np_file;
        static const std::filesystem::path _pp_file;

        float _nenergy[2];  // ISO 0,1
        float _nangle[2];
        std::vector<float> _elab[2];
        std::vector<float> _sig[2];


        NNScatteringTable* _table_dev;


        NNScatteringTableHandler();


        ~NNScatteringTableHandler();


    public:


        CUdeviceptr deviceptr() {
            return reinterpret_cast<CUdeviceptr>(this->_table_dev);
        }


    };


}