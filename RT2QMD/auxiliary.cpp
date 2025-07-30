
#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "physics/physics.hpp"

#include "auxiliary.hpp"


namespace mcutil {


    template <>
    ArgumentCard InputCardFactory<auxiliary::ModuleTestSettings>::_setCard() {
        ArgumentCard arg_card("MODULE_TEST");
        arg_card.insert<bool>("full_mode", std::vector<bool>{ 0 });
        arg_card.insert<bool>("write_ps",  std::vector<bool>{ 1 });
        arg_card.insert<int>("iter", { 1 },   { 1 }, { 100000 });
        arg_card.insert<double>("nps", std::vector<double>{ 1000000 });
        arg_card.insert<std::string>("event", { "QMD" });
        arg_card.insert<bool>("hid", std::vector<bool>{ 0 });
        return arg_card;
    }


    template <>
    ArgumentCard InputCardFactory<auxiliary::EventSegment>::_setCard() {
        ArgumentCard arg_card("EVENT");
        arg_card.insert<double>("position", { 0.0,  0.0,  0.0 });
        arg_card.insert<double>("direction", { 0.0,  0.0,  1.0 });
        arg_card.insert<double>("energy", { 0 }, { 1e+10 });
        arg_card.insert<double>("weight", { 1.0 }, { 1e-5 }, { 1e+5 });
        arg_card.insert<int>("zap", std::vector<int>{ 0 });
        arg_card.insert<int>("zat", std::vector<int>{ 0 });
        return arg_card;
    }


}


namespace auxiliary {


    ModuleTestSettings::ModuleTestSettings(mcutil::ArgInput& args) 
        : _hid_cumul(0) {
        this->_full_mode     = args["full_mode"].cast<bool>()[0];
        this->_write_ps      = args["write_ps"].cast<bool>()[0];
        this->_n_iter_kernel = args["iter"].cast<int>()[0];
        this->_nps           = (size_t)args["nps"].cast<double>()[0];
        this->_hid           = args["hid"].cast<bool>()[0];

        // bid
        std::string event_name = args["event"].cast<std::string>()[0];
        if (event_name == "QMD")
            this->_bid = mcutil::BUFFER_TYPE::QMD;
        else if (event_name == "deex")
            this->_bid = mcutil::BUFFER_TYPE::DEEXCITATION;
        else {
            mclog::fatal("'event' must be 'QMD' or 'deex'");
        }

        if (this->_write_ps && this->_nps > 5e7) {
            mclog::fatal("nps number must be smaller than 5e7 in 'write_ps' mode");
        }
    }


    EventSegment::EventSegment(mcutil::ArgInput& args) {
        std::vector<double> pos = args["position"].cast<double>();
        this->_position.x = (float)pos[0];
        this->_position.y = (float)pos[1];
        this->_position.z = (float)pos[2];

        std::vector<double> dir = args["direction"].cast<double>();
        this->_direction.x = (float)dir[0];
        this->_direction.y = (float)dir[1];
        this->_direction.z = (float)dir[2];

        this->_energy = args["energy"].cast<double>()[0];
        this->_weight = args["weight"].cast<double>()[0];
        this->_aux1 = args["zap"].cast<int>()[0];
        this->_aux2 = args["zat"].cast<int>()[0];
    }


    EventGenerator::EventGenerator(mcutil::Input& input) : _qmd_max_dim(0) {
        // read event generator settings
        std::deque<ModuleTestSettings> module_test_setting_lists
            = mcutil::InputCardFactory<ModuleTestSettings>::readAll(input);  // use the last card
        if (module_test_setting_lists.empty())
            mclog::fatalNecessary(mcutil::InputCardFactory<ModuleTestSettings>::getCardDefinition().key());
        this->_settings = module_test_setting_lists.back();

        // read events
        std::deque<EventSegment> event_list
            = mcutil::InputCardFactory<EventSegment>::readAll(input);
        if (event_list.empty())
            mclog::fatalNecessary(mcutil::InputCardFactory<EventSegment>::getCardDefinition().key());

        // build weight alias table
        std::vector<double>      probs;
        std::vector<DeviceEvent> evec;
        for (auto& event_seg : event_list) {
            DeviceEvent edev_seg;
            edev_seg.pos    = event_seg.position();
            edev_seg.dir    = event_seg.direction();
            edev_seg.energy = event_seg.energy();
            edev_seg.aux1   = event_seg.aux1();
            edev_seg.aux2   = event_seg.aux2();

            int2 za_aux1 = physics::splitZA(event_seg.aux1());
            int2 za_aux2 = physics::splitZA(event_seg.aux2());

            if (za_aux1.y <= 0)
                mclog::fatal("'zap' should be positive");
            else if (za_aux1.y > 255)
                mclog::fatal("'zap' should be smaller than 256");
            else if (za_aux1.x > za_aux1.y)
                mclog::fatal("'zap' Z number should be smaller than A number");

            if (this->_settings.bid() == mcutil::BUFFER_TYPE::QMD) {
                if (za_aux2.y <= 0)
                    mclog::fatal("'zat' should be positive");
                else if (za_aux2.y > 255)
                    mclog::fatal("'zat' should be smaller than 256");
                else if (za_aux2.x >  za_aux2.y)
                    mclog::fatal("'zat' Z number should be smaller than A number");

                if (za_aux1.y + za_aux2.y > (int)UCHAR_MAX)
                    mclog::fatal("max dimension of QMD event generator (256) is exceeded");

                this->_qmd_max_dim = za_aux1.y + za_aux2.y;
            }

            evec.push_back(edev_seg);
            probs.push_back(event_seg.weight());
        }
            
        mcutil::AliasTable table(probs.size(), &probs[0]);

        // set device table
        mcutil::DeviceVectorHelper event_list_vec(evec);
        this->_memoryUsageAppend(event_list_vec.memoryUsage());
        this->_event_list_dev = event_list_vec.address();

        mcutil::DeviceAliasData table_struct_dev_address_host;
        this->_memoryUsageAppend(table.memcpyToHostStruct(&table_struct_dev_address_host));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_event_prob_dev), 
            sizeof(mcutil::DeviceAliasData)));
        CUDA_CHECK(cudaMemcpy(this->_event_prob_dev, &table_struct_dev_address_host, 
            sizeof(mcutil::DeviceAliasData), cudaMemcpyHostToDevice));

        CUDA_CHECK(setEventListPtr(this->_event_list_dev, this->_event_prob_dev));
        return;
    }


    EventGenerator::~EventGenerator() {
        mcutil::DeviceVectorHelper(this->_event_list_dev).free();
        mcutil::DeviceVectorHelper(this->_event_prob_dev).free(1);
    }


    void EventGenerator::sample(int block, int thread) {
        initPhaseSpace(block, thread, static_cast<mcutil::BUFFER_TYPE>(this->_settings.bid()), this->_settings.hid_now());
        this->_settings.increment(block * thread);
    }


    DummyDeviceMemoryHandler::DummyDeviceMemoryHandler() {

        std::vector<int> mat_table(1, 0);
        mcutil::DeviceVectorHelper mat_table_vector(mat_table);
        this->_memoryUsageAppend(mat_table_vector.memoryUsage());
        this->_dev_region_mat_table_dummy = mat_table_vector.address();

        this->_tally_handle_dummy.n_mesh_track   = 0;
        this->_tally_handle_dummy.n_mesh_density = 0;
        this->_tally_handle_dummy.n_cross        = 0;
        this->_tally_handle_dummy.n_track        = 0;
        this->_tally_handle_dummy.n_density      = 0;
        this->_tally_handle_dummy.n_detector     = 0;
        this->_tally_handle_dummy.n_phase_space  = 0;
        this->_tally_handle_dummy.n_activation   = 0;
        this->_tally_handle_dummy.n_letd         = 0;
        this->_tally_handle_dummy.n_mesh_letd    = 0;
        this->_tally_handle_dummy.mesh_track     = nullptr;
        this->_tally_handle_dummy.mesh_density   = nullptr;
        this->_tally_handle_dummy.cross          = nullptr;
        this->_tally_handle_dummy.track          = nullptr;
        this->_tally_handle_dummy.density        = nullptr;
        this->_tally_handle_dummy.detector       = nullptr;
        this->_tally_handle_dummy.phase_space    = nullptr;
        this->_tally_handle_dummy.activation     = nullptr;
        this->_tally_handle_dummy.letd           = nullptr;
        this->_tally_handle_dummy.mesh_letd      = nullptr;
    }


    DummyDeviceMemoryHandler::~DummyDeviceMemoryHandler() {
        mcutil::DeviceVectorHelper(this->_dev_region_mat_table_dummy).free();
    }


}