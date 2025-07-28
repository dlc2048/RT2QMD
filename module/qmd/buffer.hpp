
#pragma once

#include <vector>
#include <numeric>

#include "mclog/logger.hpp"
#include "fortran/fortran.hpp"
#include "genericion/auxiliary.hpp"

#include "buffer.cuh"


namespace RT2QMD {
    namespace Buffer {


        constexpr int PROBLEM_BUFFER_SIZE_MULTIPLIER = 2;


        class ModelDumpHost {
        private:
            std::string        _file;
            std::string        _function;
            int                _line;
            QMDModel           _model;
            // 2D field
            std::vector<float>              _rr2;
            std::vector<float>              _pp2;
            std::vector<float>              _rbij;
            std::vector<float>              _rha;
            std::vector<float>              _rhe;
            std::vector<float>              _rhc;
            // 1D field
            std::vector<float>              _ffrx;
            std::vector<float>              _ffry;
            std::vector<float>              _ffrz;
            std::vector<float>              _ffpx;
            std::vector<float>              _ffpy;
            std::vector<float>              _ffpz;
            std::vector<float>              _f0rx;
            std::vector<float>              _f0ry;
            std::vector<float>              _f0rz;
            std::vector<float>              _f0px;
            std::vector<float>              _f0py;
            std::vector<float>              _f0pz;
            std::vector<float>              _rh3d;
            // Particle dump
            std::vector<ParticleDumpBuffer> _particles;
        public:


            ModelDumpHost(ModelDumpBuffer* dev);


            void write(mcutil::FortranOfstream& stream) const;


        };


        class DeviceMemoryHandler {
        private:
            size_t           _block;                // kernel block size
            size_t           _thread;               // kernel thread size
            size_t           _max_field_dimension;  // maximum field dimension (nxn)
            size_t           _memory_share;         // total memory share [bytes]
            QMDModel*        _dev_qmd_model_ptr;    // device QMD model pointer
            size_t           _n_dump;               // dump size
            ModelDumpBuffer* _dev_dump_ptr;         // device dump pointer
            void* _soa_list_p1[Participant::SOA_MEMBERS];   // one-dimensional participant element, device pointer
            void* _soa_list_f2[MeanField::SOA_MEMBERS_2D];  // two-dimensional field element, device pointer
            void* _soa_list_f1[MeanField::SOA_MEMBERS_1D];  // one-dimensional field element, device pointer


            void _initBufferSystem();


            void _initDumpSystem();


        public:


            DeviceMemoryHandler(size_t block, size_t thread, size_t max_field_dimension);


            ~DeviceMemoryHandler();


            std::vector<ModelDumpHost> getDumpData() const;


            void writeDumpData(const std::string& file_name) const;


        };


    }
}