/**
 * @file    module/deexcitation/handler.hpp
 * @brief   De-excitation global variable, data, memory and device address handler
 * @author  CM Lee
 * @date    07/12/2024
 */

#pragma once

#include "device/memory.hpp"

#include "particles/define.hpp"
#include "hadron/nucleus.hpp"

#include "auxiliary.hpp"
#include "auxiliary.cuh"
#include "channel_fission.cuh"


namespace deexcitation {


    /**
    * @brief De-excitation device memory manager
    */
    class DeviceMemoryHandler : public mcutil::DeviceMemoryHandlerInterface {
    private:
        float* _dev_m;     // emitted mass [MeV/c^2]
        float* _dev_m2;    // m^2 [MeV^2/c^4]
        float* _dev_crho;  // coulomb barrier rho [fm]
    public:


        DeviceMemoryHandler();


        ~DeviceMemoryHandler();


        void summary() const;


    };

}