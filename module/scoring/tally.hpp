#pragma once

#include <set>
#include <algorithm>
#include <regex>
#include <functional>

#include <cuda_runtime.h>

#ifndef RT2QMD_STANDALONE
#include "mesh_track.hpp"
#include "mesh_density.hpp"
#include "cross.hpp"
#include "track.hpp"
#include "density.hpp"
#include "phase_space.hpp"
#include "detector.hpp"
#include "activation.hpp"
#include "letd.hpp"
#include "mesh_letd.hpp"
#endif

#include "tally.cuh"


namespace tally {


    /**
    * @brief shared_ptr deleter
    */
    template <typename T>
    std::function<void(T*)> Deleter();


    template <typename HS, typename DS>
    class TallyHostDevicePair {
    protected:
        std::shared_ptr<HS> _host;
        std::shared_ptr<DS> _device;


        void _pullFromDeviceMesh(double total_weight);


    public:


        TallyHostDevicePair(const HS& host);


        TallyHostDevicePair(std::shared_ptr<HS>& host);


        ~TallyHostDevicePair() = default;


        size_t initDevice();


        void resetDeviceData();


        void pullFromDevice(double total_weight);


        size_t structSize() { return sizeof(DS); }


        std::shared_ptr<HS> host() { return this->_host; }


        const std::shared_ptr<HS> host() const { return this->_host; }


        DS* device() { return this->_device.get(); }


    };


#ifndef RT2QMD_STANDALONE


    template <>
    void TallyHostDevicePair<MeshDensity, DeviceMeshDensity>::pullFromDevice(double total_weight);


    template <>
    void TallyHostDevicePair<MeshTrack, DeviceMeshTrack>::pullFromDevice(double total_weight);


#endif


}


#include "tally.tpp"