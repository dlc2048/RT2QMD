
#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "auxiliary.hpp"
#include "fortran/fortran.hpp"


namespace Hadron {


    const std::filesystem::path NNScatteringTableHandler::_np_file 
        = std::filesystem::path("npdata.bin");
    const std::filesystem::path NNScatteringTableHandler::_pp_file 
        = std::filesystem::path("ppdata.bin");


    NNScatteringTableHandler::NNScatteringTableHandler() {
        namespace fp = std::filesystem;
        std::string home = mcutil::getMCRT2HomePath();
        if (home.empty())
            mclog::fatal("Environment variable 'MCRT2_HOME' is missing");
        fp::path files[2];
        files[0] = fp::path(home) / HOME / this->_np_file;
        files[1] = fp::path(home) / HOME / this->_pp_file;

        for (size_t i = 0; i < 2; ++i) {
            mcutil::FortranIfstream data(files[i].string());
            std::vector<int> data_structure = data.read<int>();
            this->_nenergy[i] = data_structure[0];
            this->_nangle[i]  = data_structure[1];
            this->_sig[i]     = data.read<float>();
            data.read<float>();
            this->_elab[i]    = data.read<float>();
            data.read<float>();
            data.read<float>();
        }

        // initialize device data
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_table_dev),
            sizeof(NNScatteringTable)));
        this->_memoryUsageAppend(sizeof(NNScatteringTable));

        // host structure
        NNScatteringTable table_host;
        for (size_t i = 0; i < 2; ++i) {
            table_host.nenergy[i] = this->_nenergy[0];
            table_host.nangle[i]  = this->_nangle[0];

            mcutil::DeviceVectorHelper sig(this->_sig[i]);
            mcutil::DeviceVectorHelper elab(this->_elab[i]);
            this->_memoryUsageAppend(sig.memoryUsage());
            this->_memoryUsageAppend(elab.memoryUsage());
            
            table_host.sig[i] = sig.address();
            table_host.elab[i] = elab.address();
        }

        CUDA_CHECK(cudaMemcpy(this->_table_dev, &table_host, 
            sizeof(NNScatteringTable), cudaMemcpyHostToDevice));
    }


    NNScatteringTableHandler::~NNScatteringTableHandler() {
        /* intended leakage */
    }


}