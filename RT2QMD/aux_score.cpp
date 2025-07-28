
#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "aux_score.hpp"


namespace mcutil {


    template <>
    ArgumentCard InputCardFactory<tally::SecYield>::_setCard() {
        ArgumentCard arg_card("YIELD");
        tally::TallyContext  ::_initializeStaticArgs(arg_card);
        tally::FilterContext ::_initializeStaticArgs(arg_card);
        tally::FluenceContext::_initializeStaticArgs(arg_card);
        arg_card.insert<int>("za");
        arg_card.insert<double>("dtheta", { 0.0, 180.0 }, { 0.0, 0.0 }, { 180.0, 180.0 });  // 4pi default
        return arg_card;
    }


}


namespace tally {


    __host__ void DeviceSecYield::free() {
        mcutil::DeviceVectorHelper(this->data).free();
    }


    SecYield::SecYield(mcutil::ArgInput& args) :
        TallyContext(args),
        FilterContext(args),
        FluenceContext(args) {
        this->_unit       = "#/sr/hist";

        std::vector<double> dtheta = args["dtheta"].cast<double>();
        if (dtheta[0] >= dtheta[1])
            mclog::fatal("'dtheta[1]' must be larger than 'dtheta[0]'");
        this->_murange    = { 
            std::cos(dtheta[1] * M_PI / 180.0), 
            std::cos(dtheta[0] * M_PI / 180.0)
        };
        this->_normalizer = 2.0 * M_PI * (this->_murange.y - this->_murange.x);   // 2pi dmu -> dOmega

        std::vector<int> za_list = args["za"].cast<int>();

        for (int za_this : za_list) {
            if (this->_za_filter.find(za_this) != this->_za_filter.end()) {
                std::stringstream ss;
                mclog::fatalListElementAlreadyExist(za_this);
            }
            this->_za_filter.insert(za_this);
        }

        // host memory
        this->_data.clear();
        this->_data.resize(this->_nbin, 0.0);
        this->_unc.clear();
        this->_unc.resize(this->_nbin, 0.0);
    }


    size_t SecYield::mallocDeviceStruct(DeviceSecYield** device) {
        size_t memsize = 0x0u;
        DeviceSecYield struct_host;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(device), sizeof(DeviceSecYield)));
        memsize += sizeof(DeviceSecYield);

        // za filter
        for (int i = 0; i < Hadron::Projectile::ZA_SCORING_MASK_SIZE; ++i)
            struct_host.za_mask[i] = this->_za_mask[i];

        struct_host.type    = static_cast<int>(this->_etype);
        double2 eihd = this->getInterpCoeff();
        struct_host.eih     = { (float)eihd.x, (float)eihd.y };
        struct_host.nebin   = this->_nbin;
        struct_host.murange = { (float)this->_murange.x, (float)this->_murange.y };

        // allocate T & T2 memory
        size_t size = this->_nbin;
        tallyMemoryInitialize(&struct_host.data, size);
        CUDA_CHECK(cudaMemcpy(*device, &struct_host,
            sizeof(DeviceSecYield), cudaMemcpyHostToDevice));
        return memsize + sizeof(double2) * size;
    }


    void SecYield::memcpyFromDeviceStruct(DeviceSecYield* device) {
        DeviceSecYield struct_host;
        CUDA_CHECK(cudaMemcpy(&struct_host, device,
            sizeof(DeviceSecYield), cudaMemcpyDeviceToHost));
        // calculate memory size
        size_t size_dev = struct_host.nebin;
        size_t size     = this->_nbin;
        if (size_dev != size)
            throw std::runtime_error(
                "YIELD data dimension mismatched between host and device"
            );
        this->_data.clear();
        this->_data.resize(size);
        this->_unc.clear();
        this->_unc.resize(size);
        std::vector<double2> data_raw = this->_getDataFromDevice(struct_host.data, size);
        for (size_t i = 0; i < size; ++i) {
            this->_data[i] = data_raw[i].x;
            this->_unc[i]  = data_raw[i].y;
        }
    }


    void SecYield::write(const std::string& file_name) const {
        if (this->_ascii) {
            std::ofstream text_file(file_name);
            mclog::FormattedTable fmt({ 16, 16 });
            fmt << "Format" << "YIELD";
            text_file << fmt.str() << std::endl;
            this->_writeTallyHeader(text_file);
            this->_writeTallyFilter(text_file);
            this->_write1DAlignedData(text_file, this->_data, this->_unc);
        }
        else {
            mcutil::FortranOfstream binary(file_name);
            this->_writeTallyHeader(binary);
            this->_writeTallyFilter(binary);
            this->_writeTallyEnergyStructure(binary);
            this->_writeTallyData(binary);
        }
    }


    void SecYield::summary() const {
        this->_summaryTallyContext();
        this->_summaryFilterContext();
        this->_summaryFluenceContext();
    }


    void initDeviceStruct(DeviceSecYield* device) {
        DeviceSecYield struct_host;
        CUDA_CHECK(cudaMemcpy(&struct_host, device,
            sizeof(DeviceSecYield), cudaMemcpyDeviceToHost));
        // calculate memory size
        size_t size = struct_host.nebin;
        tallyMemoryReset(&struct_host.data, size);
    }


    void DeviceMemoryHandlerMT::_setIonFilterForYield() {
        for (auto& yield : this->_yield) {
            const std::set<int>& filter = yield.host()->filter();

            if (filter.size() == 1 && *filter.begin() == 0)
                continue;

            if (filter.empty())
                continue;

            yield.host()->clearZAFilter();

            if (!(yield.host()->isActivated(Define::PID::PID_GENION) ||
                yield.host()->isActivated(Define::PID::PID_NEUTRON))) {  // GENION is not activated to this tally
                std::stringstream ss;
                ss << "Attempting to apply an ion filter to '" << yield.host()->name() << "', that does not score the GENION or NEUTRON part";
                mclog::warning(ss);
            }
            for (int za : filter) {
                int2 za_split = physics::splitZA(za);
                if (za_split.y == 0) {  // ZZZ000
                    std::vector<uint32_t> index_list;
                    for (uint32_t i = 0; i < this->_proj_list.size(); ++i) {
                        int2 za_p_split = physics::splitZA(this->_proj_list[i]);
                        if (za_p_split.x == za_split.x)
                            index_list.push_back(i);
                    }
                    if (index_list.empty()) {
                        std::stringstream ss;
                        ss << "Undefined projectile '" << za << "'";
                        mclog::fatal(ss);
                    }
                    for (uint32_t pos : index_list) {
                        if (!yield.host()->switchTargetZA(this->_proj_list[pos], pos)) {
                            std::stringstream ss;
                            ss << "Ion filter for projectile '" << this->_proj_list[pos] << "' is already activated";
                            mclog::fatal(ss);
                        }
                    }
                }
                else {  // ZZZAAA
                    ptrdiff_t pos = std::distance(this->_proj_list.begin(), std::find(this->_proj_list.begin(), this->_proj_list.end(), za));
                    if (pos == (ptrdiff_t)this->_proj_list.size()) {
                        std::stringstream ss;
                        ss << "Undefined projectile '" << za << "'";
                        mclog::fatal(ss);
                    }
                    if (!yield.host()->switchTargetZA(za, (uint32_t)pos)) {
                        std::stringstream ss;
                        ss << "Ion filter for projectile '" << za << "' is already activated";
                        mclog::fatal(ss);
                    }
                }
            }
        }
    }


    DeviceMemoryHandlerMT::DeviceMemoryHandlerMT(
        mcutil::Input& input,
        const std::string& project_name,
        const mcutil::DeviceController& dev_prop,
        const std::vector<int>& projectile_list
    ) : _block(dev_prop.block()), _thread(dev_prop.thread()), _project(project_name) {

        std::set<std::string> namelist;
        this->_yield     = this->_read<SecYield, DeviceSecYield>(input, namelist);
        this->_proj_list = projectile_list;

        this->_setIonFilterForYield();

        try {
            this->_memoryUsageAppend(allocDevice(this->_yield));
        }
        catch (std::runtime_error& e) {
            mclog::warning(e.what());
            mclog::fatal("CUDA failed: out of memory");
        }

        return;
    }


    void DeviceMemoryHandlerMT::summary() const {
        // do something
    }


    void DeviceMemoryHandlerMT::appendYield(mcutil::RingBuffer* buffer_ptr) {
        for (size_t i = 0; i < mcutil::BUFFER_TYPE::RELAXATION; ++i) {
            auto iter = this->_yield_handle.find(i);
            if (iter != this->_yield_handle.end())
                appendYieldFromBuffer(
                    this->_block, this->_thread, 
                    buffer_ptr,
                    static_cast<mcutil::BUFFER_TYPE>(i), 
                    &iter->second
                );
        }
    }


    void DeviceMemoryHandlerMT::prepareYieldHandle() {
        
        for (const auto& p : mcutil::getPidHash()) {
            int pid = p.first;
            int bid = p.second;

            DeviceYieldHandle  handle_host;

            // YIELD
            {
                std::vector<DeviceSecYield*> yield_dev;
                for (auto& pair : this->_yield) {
                    if (pair.host()->isActivated(pid))
                        yield_dev.push_back(pair.device());
                }
                handle_host.n_yield = (int)yield_dev.size();
                if (handle_host.n_yield) {
                    mcutil::DeviceVectorHelper vector(yield_dev);
                    handle_host.yield = vector.address();
                }
            }

            // prepare device shared ptr
            this->_yield_handle.insert({ bid, handle_host });
        }

        return;
    }


    void DeviceMemoryHandlerMT::pull(double total_weight) {
        for (auto& pair : this->_yield)
            pair.pullFromDevice(total_weight);
    }


    void DeviceMemoryHandlerMT::initTallies() {
        for (auto& pair : this->_yield)
            pair.resetDeviceData();
    }


    void DeviceMemoryHandlerMT::write(const std::filesystem::path& path, int iter) const {
        for (const auto& tally : this->_yield) {
            std::stringstream ss;
            ss << this->_project << "_" << tally.host()->name() << "_" << iter;
            ss << (tally.host()->ascii() ? ".txt" : ".yld");
            tally.host()->summary();
            tally.host()->write((path / std::filesystem::path(ss.str())).string());
        }
    }


}