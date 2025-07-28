
#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "tally.hpp"


namespace tally {


    __host__ void DeviceHandle::Deleter(DeviceHandle* ptr) {
        DeviceHandle host;
        CUDA_CHECK(cudaMemcpy(&host, ptr, sizeof(DeviceHandle), cudaMemcpyDeviceToHost));
        if (host.n_mesh_track)   CUDA_CHECK(cudaFree(host.mesh_track));
        if (host.n_mesh_density) CUDA_CHECK(cudaFree(host.mesh_density));
        if (host.n_cross)        CUDA_CHECK(cudaFree(host.cross));
        if (host.n_track)        CUDA_CHECK(cudaFree(host.track));
        if (host.n_density)      CUDA_CHECK(cudaFree(host.density));
        if (host.n_detector)     CUDA_CHECK(cudaFree(host.detector));
        if (host.n_phase_space)  CUDA_CHECK(cudaFree(host.phase_space));
        if (host.n_activation)   CUDA_CHECK(cudaFree(host.activation));
        if (host.n_letd)         CUDA_CHECK(cudaFree(host.letd));
        if (host.n_mesh_letd)    CUDA_CHECK(cudaFree(host.mesh_letd));
        CUDA_CHECK(cudaFree(ptr));
    }


#ifndef RT2QMD_STANDALONE


    template <>
    void TallyHostDevicePair<MeshDensity, DeviceMeshDensity>::pullFromDevice(double total_weight) {
        this->_pullFromDeviceMesh(total_weight);
    }


    template <>
    void TallyHostDevicePair<MeshTrack, DeviceMeshTrack>::pullFromDevice(double total_weight) {
        this->_pullFromDeviceMesh(total_weight);
    }


#endif


}