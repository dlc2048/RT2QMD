
#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include <algorithm>
#include <istream>

#include "nucleus.hpp"
#include "physics/constants.hpp"


namespace Nucleus {


    MassRecord::MassRecord() :
        _a(0), _z(0), _excess(0.0) {}


    MassRecord::MassRecord(const int a, const int z, const double e) :
        _a(a), _z(z), _excess(e) {}


    std::istream& operator>>(std::istream& in, MassRecord& record) {
        return (in >> record._a >> record._z >> record._excess);
    }


    bool operator<(const MassRecord& lhs, const MassRecord& rhs) {
        int zl, zr;
        int al, ar;
        zl = lhs.z();
        zr = rhs.z();
        al = lhs.a();
        ar = rhs.a();
        if (zl < zr)
            return true;
        else if (zl == zr) {
            if (al < ar)
                return true;
            else
                return false;
        }
        else
            return false;
    }


    bool operator==(const MassRecord& lhs, const MassRecord& rhs) {
        int zl, zr;
        int al, ar;
        zl = lhs.z();
        zr = rhs.z();
        al = lhs.a();
        ar = rhs.a();
        return (zl == zr && al == ar);
    }


    const std::filesystem::path MassTableHandler::_mass_file = std::filesystem::path("walletlifetime.dat");


    MassTableHandler::MassTableHandler() {
        namespace fp = std::filesystem;
        std::string home = mcutil::getMCRT2HomePath();
        if (home.empty())
            mclog::fatal("Environment variable 'MCRT2_HOME' is missing");
        fp::path file(home);
        file = file / HOME / this->_mass_file;

        // Open the file stream
        std::ifstream mass_table_in(file.c_str());
        if (!mass_table_in.good()) {
            std::stringstream ss;
            ss << "Cannot open file '" << file.string() << "'";
            mclog::fatal(ss);
        }

        // Read
        std::set<MassRecord> records;

        while (mass_table_in.good()) {
            MassRecord rec;
            mass_table_in >> rec;
            if (mass_table_in.good())
                records.insert(rec);
        }
        mass_table_in.close();
        {
            std::stringstream ss;
            ss << "Read nuclear mass table '" << file.string() << "'";
            mclog::debug(ss);
        }

        int zmax;
        zmax = std::max_element(records.begin(), records.end(), compareZ)->z();
        this->_zmax = (size_t)zmax;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_table_dev),
            sizeof(MassTable) * (this->_zmax + 1)));
        this->_memoryUsageAppend(sizeof(MassTable) * (this->_zmax + 1));
        for (size_t z = 0; z <= this->_zmax; ++z) {
            // Fill mass array
            std::vector<float> mass_arr;
            int amin = 99999, amax = -1, anow = -1;
            bool fhit = false;

            for (const MassRecord& rec : records) {
                if (rec.z() != z)
                    continue;
                int a = rec.a();
                if (!fhit) {
                    amin = a;
                    anow = a;
                    fhit = true;
                }
                for (; anow < a; ++anow)
                    mass_arr.push_back(-1.0);
                double mass = a * constants::ATOMIC_MASS_UNIT + rec.e() - electronMass((int)z);
                anow++;
                mass_arr.push_back((float)mass);
                int za = (int)z * 1000 + a;
                this->_table_host.insert({ za, mass });
            }
            amax = anow;

            // Set host side table
            MassTable table_host;
            table_host.z = (int)z;
            table_host.amin = amin;
            table_host.amax = amax;
            if (amax > 0)
                this->_memoryUsageAppend(mcutil::cudaMemcpyVectorToDevice(mass_arr, &table_host.mass));
            CUDA_CHECK(cudaMemcpy(&this->_table_dev[z], &table_host,
                sizeof(MassTable), cudaMemcpyHostToDevice));
        }
    }


    MassTableHandler::~MassTableHandler() {
        /*
        for (size_t z = 0; z <= this->_zmax; ++z) {
            MassTable table_host;
            CUDA_CHECK(cudaMemcpy(&table_host, &this->_table_dev[z],
                sizeof(MassTable), cudaMemcpyDeviceToHost));
            if (table_host.amax > 0)
                CUDA_CHECK(cudaFree(table_host.mass));
        }
        CUDA_CHECK(cudaFree(this->_table_dev));
        */
    };


    CUdeviceptr MassTableHandler::deviceptr() {
        return reinterpret_cast<CUdeviceptr>(this->_table_dev);
    }


    double MassTableHandler::getMass(int za) const {
        int z = physics::getZnumberFromZA(za);
        int a = physics::getAnumberFromZA(za);
        int za_rec = z * 1000 + a;

        std::map<int, double>::const_iterator iter = this->_table_host.find(za_rec);
        if (iter == this->_table_host.end())
            return (double)getWeizsaeckerMass(a, z);
        else
            return iter->second;
    }


    ENSDFRecord::ENSDFRecord() :
        _excitation(0.0), _life_time(-1.0), _spin(0), _dmoment(0.0) {}



    std::istream& operator>>(std::istream& in, ENSDFRecord& record) {
        std::string float_level;
        std::istream& out = in >> record._excitation >> float_level >> record._life_time >> record._spin >> record._dmoment;
        record._excitation *= 1e-3;
        record._life_time  *= 1e-9;
        return out;
    }


    bool operator<(const ENSDFRecord& lhs, const ENSDFRecord& rhs) {
        return lhs.excitation() < rhs.excitation();
    }


    bool operator==(const ENSDFRecord& lhs, const ENSDFRecord& rhs) {
        return
            lhs.excitation() == rhs.excitation() &&
            lhs.lifeTime()   == rhs.lifeTime()   &&
            lhs.spin()       == rhs.spin()       &&
            lhs.dmoment()    == rhs.dmoment();
    }


    const std::filesystem::path ENSDFTable::_ensdf_file = std::filesystem::path("ENSDFSTATE.dat");


    ENSDFTable::ENSDFTable() {
        namespace fp = std::filesystem;
        std::string home = mcutil::getMCRT2HomePath();
        if (home.empty())
            mclog::fatal("Environment variable 'MCRT2_HOME' is missing");
        fp::path file(home);
        file = file / HOME / this->_ensdf_file;

        // Open the file stream
        std::ifstream ensdf_in(file.c_str());
        if (!ensdf_in.good()) {
            std::stringstream ss;
            ss << "Cannot open file '" << file.string() << "'";
            mclog::fatal(ss);
        }

        // read ENSDF data
        while (ensdf_in.good()) {
            int z, a, za;
            ENSDFRecord rec;
            ensdf_in >> z >> a >> rec;
            za = z * 1000 + a;
            if (ensdf_in.good()) {
                auto pos = this->_table_host.insert({ za, std::vector<ENSDFRecord>() });
                pos.first->second.push_back(rec);
            }
        }
        ensdf_in.close();
        {
            std::stringstream ss;
            ss << "Read ENSDF table '" << file.string() << "'";
            mclog::debug(ss);
        }

        // sort ENSDF data
        for (auto& items : this->_table_host)
            std::sort(items.second.begin(), items.second.end());

        // build device side long-lived particle lists
        double life_cut = Define::IonInelastic::getInstance().evaporationCutoff();

        std::vector<int> stable_offset;
        std::vector<int> stable_list;

        int current_z_entry = 0;

        stable_offset.push_back(0);  // Z = 0
        stable_list.push_back(1);

        for (auto& items : this->_table_host) {
            int za = items.first;
            int z  = physics::getZnumberFromZA(za);
            int a  = physics::getAnumberFromZA(za);

            if (z != current_z_entry) {
                // end of list
                current_z_entry = z;
                stable_list.push_back(9999);
                stable_offset.push_back((int)stable_list.size());
            }

            double lifetime = items.second[0].lifeTime();
            if (lifetime < 0 || lifetime > life_cut)
                stable_list.push_back(a);
        }

        // memory
        mcutil::DeviceVectorHelper stable_offset_vec(stable_offset);
        mcutil::DeviceVectorHelper stable_list_vec(stable_list);

        this->_memoryUsageAppend(stable_offset_vec.memoryUsage());
        this->_memoryUsageAppend(stable_list_vec.memoryUsage());

        this->_table_dev.offset      = stable_offset_vec.address();
        this->_table_dev.mass_number = stable_list_vec.address();
    }


    ENSDFTable::~ENSDFTable() {
        /*
        mcutil::DeviceVectorHelper(this->_table_dev.offset).free();
        mcutil::DeviceVectorHelper(this->_table_dev.mass_number).free();
        */
    }


    size_t ENSDFTable::numberOfSate(int za) const {
        auto pos = this->_table_host.find(za);
        if (pos == this->_table_host.end())
            return 0x0u;
        else
            return pos->second.size();
    }


    const ENSDFRecord& ENSDFTable::get(int za, int level) const {
        return this->_table_host.find(za)->second[level];
    }


    std::vector<int> ENSDFTable::isotopes(int z) const {
        std::vector<int> za_list;
        for (auto& iter : this->_table_host) {
            int z_this = physics::getZnumberFromZA(iter.first);
            if (z_this == z)
                za_list.push_back(iter.first);
        }
        std::sort(za_list.begin(), za_list.end());
        return za_list;
    }


    bool compareA(const MassRecord& lhs, const MassRecord& rhs) {
        return lhs.a() < rhs.a();
    }


    bool compareZ(const MassRecord& lhs, const MassRecord& rhs) {
        return lhs.z() < rhs.z();
    }


    double electronMass(int z) {
        double zf      = (double)z;
        double emass   = zf * (double)constants::MASS_ELECTRON;
        double binding = 14.4381 * std::pow(zf, 2.39) + 1.55468 * 1e-6 * std::pow(zf, 5.35);
        return emass - binding * 1e-6;
    }


}
