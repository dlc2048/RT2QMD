#pragma once

#include "parser/input.hpp"
#include "device/memory_manager.hpp"
#include "device/algorithm.hpp"

#include "scoring/tally.cuh"

#include "auxiliary.cuh"


namespace auxiliary {


    /**
    * @brief Settings for module (interactions) test
    */
    class ModuleTestSettings {
    protected:
        bool   _full_mode;
        bool   _write_ps;
        int    _n_iter_kernel;
        size_t _nps;
        bool   _hid;
        int    _bid;
        int    _hid_cumul;
        

    public:


        ModuleTestSettings() {}


        ModuleTestSettings(mcutil::ArgInput& args);


        bool    fullMode()    const { return this->_full_mode; }
        bool    writePS()     const { return this->_write_ps; }
        int     nIterKernel() const { return this->_n_iter_kernel; }
        size_t  nps()         const { return this->_nps; }
        bool    hid()         const { return this->_hid; }
        int     hid_now()     const { return this->_hid_cumul; }
        int     bid()         const { return this->_bid; }


        void    increment(size_t hist)   { this->_hid_cumul += hist; }


        void    init() { this->_hid_cumul = 0; }


    };


    class EventSegment {
    protected:
        float3 _position;
        float3 _direction;
        float  _energy;
        float  _weight;
        int    _aux1;
        int    _aux2;


    public:


        EventSegment(mcutil::ArgInput& args);


        float3  position()    { return this->_position; }
        float3  direction()   { return this->_direction; }
        float   energy()      { return this->_energy; }
        float   weight()      { return this->_weight; }
        int     aux1()        { return this->_aux1; }
        int     aux2()        { return this->_aux2; }


    };


    class EventGenerator : public mcutil::DeviceMemoryHandlerInterface {
    private:
        ModuleTestSettings _settings;

        DeviceEvent* _event_list_dev;
        mcutil::DeviceAliasData* _event_prob_dev;

        size_t _qmd_max_dim;


    public:


        EventGenerator(mcutil::Input& input);


        ~EventGenerator();


        ModuleTestSettings& settings() { return this->_settings; }


        void sample(int block, int thread);


        size_t maxDimQMD() const { return this->_qmd_max_dim; }


    };


    class DummyDeviceMemoryHandler : public mcutil::DeviceMemoryHandlerInterface {
    private:
        int* _dev_region_mat_table_dummy;
        tally::DeviceHandle _tally_handle_dummy;
    public:


        DummyDeviceMemoryHandler();


        ~DummyDeviceMemoryHandler();


        CUdeviceptr ptrDummyRegionMaterialTable() const { 
            return reinterpret_cast<CUdeviceptr>(this->_dev_region_mat_table_dummy);
        }


        const tally::DeviceHandle& getDummyTallyHandle() const {
            return this->_tally_handle_dummy;
        }


    };





}
