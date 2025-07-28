
#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "auxiliary.hpp"


namespace deexcitation {


    const std::filesystem::path CameronCorrection::_sp_file 
        = std::filesystem::path("cameron_sp_correction.bin");

    const std::filesystem::path CameronCorrection::_spin_file
        = std::filesystem::path("cameron_spin_correction.bin");

    const std::filesystem::path CameronCorrection::_pair_file
        = std::filesystem::path("cameron_pairing_correction.bin");


    CameronCorrection::CameronCorrection() {
        namespace fp = std::filesystem;
        std::string home = mcutil::getMCRT2HomePath();
        if (home.empty())
            mclog::fatal("Environment variable 'MCRT2_HOME' is missing");

        fp::path sp_file(home);
        sp_file = sp_file / HOME / this->_sp_file;
        mcutil::FortranIfstream data_sp(sp_file.string());
        this->_sp_correction_p = data_sp.read<float>();
        this->_sp_correction_n = data_sp.read<float>();

        fp::path spin_file(home);
        spin_file = spin_file / HOME / this->_spin_file;
        mcutil::FortranIfstream data_spin(spin_file.string());
        this->_spin_correction_p = data_spin.read<float>();
        this->_spin_correction_n = data_spin.read<float>();

        fp::path pair_file(home);
        pair_file = pair_file / HOME / this->_pair_file;
        mcutil::FortranIfstream data_pair(pair_file.string());
        this->_pair_correction_p = data_pair.read<float>();
        this->_pair_correction_n = data_pair.read<float>();


        mcutil::DeviceVectorHelper spin_pairing_p(this->_sp_correction_p);
        mcutil::DeviceVectorHelper spin_pairing_n(this->_sp_correction_n);
        mcutil::DeviceVectorHelper pairing_p(this->_pair_correction_p);
        mcutil::DeviceVectorHelper pairing_n(this->_pair_correction_n);
        mcutil::DeviceVectorHelper spin_p(this->_spin_correction_p);
        mcutil::DeviceVectorHelper spin_n(this->_spin_correction_n);

        this->_memoryUsageAppend(spin_pairing_p.memoryUsage());
        this->_memoryUsageAppend(spin_pairing_n.memoryUsage());
        this->_memoryUsageAppend(pairing_p.memoryUsage());
        this->_memoryUsageAppend(pairing_n.memoryUsage());
        this->_memoryUsageAppend(spin_p.memoryUsage());
        this->_memoryUsageAppend(spin_n.memoryUsage());

        this->_sp_correction_p_dev   = spin_pairing_p.address();
        this->_sp_correction_n_dev   = spin_pairing_n.address();
        this->_pair_correction_p_dev = pairing_p.address();
        this->_pair_correction_n_dev = pairing_n.address();
        this->_spin_correction_p_dev = spin_p.address();
        this->_spin_correction_n_dev = spin_n.address();
    }


    CameronCorrection::~CameronCorrection() {
        // Singleton DLL intended memory leak
        /*
        mcutil::DeviceVectorHelper(this->_sp_correction_p_dev).free();
        mcutil::DeviceVectorHelper(this->_sp_correction_n_dev).free();
        mcutil::DeviceVectorHelper(this->_pair_correction_p_dev).free();
        mcutil::DeviceVectorHelper(this->_pair_correction_n_dev).free();
        mcutil::DeviceVectorHelper(this->_spin_correction_p_dev).free();
        mcutil::DeviceVectorHelper(this->_spin_correction_n_dev).free();
        */
    }


    const std::filesystem::path CoulombBarrier::_cr_file
        = std::filesystem::path("coulomb_radius.bin");


    CoulombBarrier::CoulombBarrier() {
        namespace fp = std::filesystem;
        std::string home = mcutil::getMCRT2HomePath();
        if (home.empty())
            mclog::fatal("Environment variable 'MCRT2_HOME' is missing");

        fp::path cr_file(home);
        cr_file = cr_file / HOME / this->_cr_file;
        mcutil::FortranIfstream data_cr(cr_file.string());
        this->_coulomb_radius = data_cr.read<float>();

        mcutil::DeviceVectorHelper coulomb_radius_vector(this->_coulomb_radius);
        this->_memoryUsageAppend(coulomb_radius_vector.memoryUsage());
        this->_coulomb_radius_dev = coulomb_radius_vector.address();
    }


    CoulombBarrier::~CoulombBarrier() {
        // for DLL
        /*
        mcutil::DeviceVectorHelper(this->_coulomb_radius_dev).free();
        */
    }


    double CoulombBarrier::coulombBarrierRadius(int z, int a) const {
        double r = Nucleus::explicitNuclearRadius({ (unsigned char)z, (unsigned char)a });
        if (r == 0.0) {
            z = std::min(z, 92);
            r = this->_coulomb_radius[z] * std::pow((double)a, 1. / 3.);
        }
        return r;
    }


    UnstableBreakUp::UnstableBreakUp() {
        // mass table
        Nucleus::MassTableHandler& host_mass_table
            = Nucleus::MassTableHandler::getInstance();

        this->_z = { 0, 0, 0, 1, 1, 1, 2, 2, 0, 2 };
        this->_a = { 0, 0, 1, 1, 2, 3, 3, 4, 2, 2 };
        this->_m = { 
            0.0, 
            0.0,
            constants::MASS_NEUTRON, 
            constants::MASS_PROTON,
            host_mass_table.getMass(1002), 
            host_mass_table.getMass(1003), 
            host_mass_table.getMass(2003),
            host_mass_table.getMass(2004),
            constants::MASS_NEUTRON * 2.0,
            constants::MASS_PROTON  * 2.0
        };

        for (double m : this->_m)
            this->_m2.push_back(m * m);
    }


    bool UnstableBreakUp::isApplicable(int z, int a) {
        if (a >= MAX_A_BREAKUP)
            return false;
        // mass table
        Nucleus::MassTableHandler& host_mass_table
            = Nucleus::MassTableHandler::getInstance();

        double m0   = host_mass_table.getMass(z * 1000 + a);
        bool   found_channel = false;
        for (int i = CHANNEL::CHANNEL_NEUTRON; i < CHANNEL::CHANNEL_UNKNWON; i++) {
            int zres = z - this->_z[i];
            int ares = a - this->_a[i];
            if (zres >= 0 && ares >= zres && ares >= this->_a[i]) {  // physically allowed channel
                if (ares <= 4) {  // simple channel
                    for (int j = CHANNEL::CHANNEL_NEUTRON; j < CHANNEL::CHANNEL_UNKNWON; ++j) {
                        if (zres == PROJ_Z[j] && ares == PROJ_A[j]) {
                            double delm = m0 - this->_m[i] - this->_m[j];
                            if (delm > 0.f) {
                                found_channel = true;
                                break;
                            }
                        }
                    }
                }
                if (found_channel)
                    break;
                // no simple channel
                double mres = host_mass_table.getMass(zres * 1000 + ares);
                double e    = m0 - mres - this->_m[i];
                if (e > 0.f) {
                    found_channel = true;
                    break;
                }
            }
        }
        return found_channel;
    }


    namespace photon {


        Transition::Transition() :
            daughter_level     (-1),
            transition_energy  (-1.0),
            relative_intensity (-1.0),
            multipolarity_id   (-1),
            multipolarity_ratio(-1.0),
            ic_alpha           (-1.0),
            ic_ratio           {+0.0}
        {}


        NuclearLevel::NuclearLevel() :
            _level_energy   (0.0),
            _level_half_life(0.0),
            _spin           (0)
        {}


        NuclearLevel::NuclearLevel(double exc_energy, double half_life, int spin) :
            _level_energy   (exc_energy),
            _level_half_life(half_life),
            _spin           (spin)
        {}


        void NuclearLevel::appendTransitionMode(const Transition& mode) {
            this->_transition.push_back(mode);
        }


        const std::filesystem::path PhotonEvaporationTable::_library 
            = std::filesystem::path("PhotonEvaporation5.7");


        const std::vector<NuclearLevel>& PhotonEvaporationTable::_loadLevelData(int za) {
            namespace fp = std::filesystem;

            std::stringstream fnsstream;
            int z = physics::getZnumberFromZA(za);
            int a = physics::getAnumberFromZA(za);
            fnsstream << "z" << z << ".a" << a;

            std::string home = mcutil::getMCRT2HomePath();
            if (home.empty())
                mclog::fatal("Environment variable 'MCRT2_HOME' is missing");
            fp::path file_name(home);
            file_name = file_name / HOME / this->_library / fnsstream.str();

            std::ifstream fstream(file_name.string());
            if (!fstream.good())
                mclog::fatalFileNotExist(file_name.string());


            auto  pos        = this->_table.insert({ za, std::vector<NuclearLevel>() });
            auto& level_data = pos.first->second;

            std::string line;
            int n_remaining_transition = 0;
            while (!std::getline(fstream, line).eof()) {
                std::stringstream ss(line);
                if (n_remaining_transition) {
                    n_remaining_transition--;
                    Transition transition;
                    ss >> transition;
                    level_data.back().appendTransitionMode(transition);
                }
                else {
                    NuclearLevel level;
                    ss >> level;
                    ss >> n_remaining_transition;
                    level_data.push_back(level);
                }
            }

            if (n_remaining_transition) {
                std::stringstream ss;
                ss << "Encounter unexpected EOF in level data " << file_name.string();
                mclog::fatal(ss);
            }

            return level_data;
        }


        const NuclearLevel& PhotonEvaporationTable::get(int za, int level) {
            std::map<int, std::vector<NuclearLevel>>::const_iterator iter
                = this->_table.find(za);

            const std::vector<NuclearLevel>& data_list = iter == this->_table.end()
                ? this->_loadLevelData(za)
                : iter->second;

            if (data_list.size() <= level) {
                std::stringstream ss;
                ss << "The nuclear level " << level << " data is not found in nucleus " << za;
                mclog::fatal(ss);
            }
            return data_list[level];
        }


        void PhotonEvaporationTable::readAll() {
            namespace fp = std::filesystem;

            std::string home = mcutil::getMCRT2HomePath();
            if (home.empty())
                mclog::fatal("Environment variable 'MCRT2_HOME' is missing");
            fp::path path(home);
            path = path / HOME / this->_library;

            std::regex  zafile_regex("z(\\d{1,3})\\.a(\\d{1,3})");
            std::smatch match;

            for (const fp::directory_entry& entry : fp::directory_iterator(path)) {
                std::string file_name = entry.path().filename().string();
                if (std::regex_match(file_name, match, zafile_regex)) {
                    int z, a, za;
                    std::stringstream zstr(match[1]);
                    std::stringstream astr(match[2]);
                    zstr >> z;
                    astr >> a;
                    za = z * 1000 + a;
                    this->get(za, 0);
                }
            }
        }


        void PhotonEvaporationTable::clear() {
            this->_table.clear();
        }


        std::stringstream& operator>>(std::stringstream& sstream, Transition& q) {
            sstream 
                >> q.daughter_level 
                >> q.transition_energy
                >> q.relative_intensity 
                >> q.multipolarity_id 
                >> q.multipolarity_ratio
                >> q.ic_alpha;

            if (q.ic_alpha > 0.0) {
                for (int i = 0; i < 10; ++i)
                    sstream >> q.ic_ratio[i];
            }

            return sstream;
        }


        std::stringstream& operator>>(std::stringstream& sstream, NuclearLevel& q) {
            std::string floating_level;
            double      exc_energy, half_life, spin;
            int         level_id;
            sstream >> level_id >> floating_level >> exc_energy >> half_life >> spin;
            q = NuclearLevel(exc_energy, half_life, (int)spin);
            return sstream;
        } 


    }


}