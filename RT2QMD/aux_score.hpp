#pragma once

#include "parser/input.hpp"
#include "scoring/tally_interface.hpp"
#include "scoring/tally.hpp"

#include "aux_score.cuh"


namespace tally {


    template <typename HostData, typename Deviceptr>
    using TallyDeque = std::deque<TallyHostDevicePair<HostData, Deviceptr>>;


    class SecYield :
        public TallyContext,
        public FilterContext,
        public FluenceContext {
    private:
        double2       _murange;    // solid angle range
        std::set<int> _za_filter;  // ion filter
    public:


        SecYield(mcutil::ArgInput& args);


        size_t mallocDeviceStruct(DeviceSecYield** device);


        void memcpyFromDeviceStruct(DeviceSecYield* device);


        void write(const std::string& file_name) const;


        void summary() const;


        const std::set<int>& filter() const { return this->_za_filter; }


    };


    void initDeviceStruct(DeviceSecYield* device);


    template <typename HS, typename DS>
    size_t allocDevice(TallyDeque<HS, DS>& deque) {
        size_t memsize = 0x0u;
        for (auto& pair : deque) {
            memsize += pair.initDevice();
        }
        return memsize;
    }


    /**
    * @brief tally manager for the event generator and the module tester (MT)
    */
    class DeviceMemoryHandlerMT : public mcutil::DeviceMemoryHandlerInterface {
    private:

        // project name
        const std::string          _project;
        // projectile lists
        std::vector<int>           _proj_list;

        // dimension
        size_t _block;
        size_t _thread;

        // host side
        TallyDeque<SecYield, DeviceSecYield> _yield;

        // tally kernel handle {bid, Device ptr}
        std::map<int, DeviceYieldHandle> _yield_handle;

        template <typename HS, typename DS>
        TallyDeque<HS, DS> _read(mcutil::Input& input, std::set<std::string>& namelist) {
            TallyDeque<HS, DS> tally_pair_list;
            std::deque<HS> tally_list = mcutil::InputCardFactory<HS>::readAll(input);
            for (HS& tally : tally_list) {
                if (namelist.find(tally.name()) != namelist.end())
                    mclog::fatalNameAlreadyExist(tally.name());
                else
                    namelist.insert(tally.name());
                TallyHostDevicePair<HS, DS> deque_item(tally);
                tally_pair_list.push_back(deque_item);
            }
            return tally_pair_list;
        }


        void _setIonFilterForYield();


    public:


        DeviceMemoryHandlerMT(
            mcutil::Input& input,
            const std::string& project_name,
            const mcutil::DeviceController& dev_prop,
            const std::vector<int>& projectile_list
        );


        void summary() const;


        void appendYield(mcutil::RingBuffer* buffer_ptr, mcutil::BUFFER_TYPE bid);


        void prepareYieldHandle();


        const DeviceYieldHandle& yieldHandle(mcutil::BUFFER_TYPE bid) const { return this->_yield_handle.find(bid)->second; }


        void pull(double total_weight);


        void initTallies();


        void write(const std::filesystem::path& path, int iter) const;


    };


}