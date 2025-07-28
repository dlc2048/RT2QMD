
#include "atomic.hpp"


namespace Atomic {


    const std::filesystem::path MassTable::_mass_file = std::filesystem::path("mass_table.bin");


    MassTable::MassTable() {
        namespace fp = std::filesystem;
        std::string home = mcutil::getMCRT2HomePath();
        if (home.empty())
            mclog::fatal("Environment variable 'MCRT2_HOME' is missing");
        fp::path file_name(home);
        fp::path lib(Define::Neutron::getInstance().library());
        file_name = file_name / HOME / lib / this->_mass_file;

        mcutil::FortranIfstream data(file_name.string());

        std::vector<int>   za   = data.read<int>();
        std::vector<float> mass = data.read<float>();

        if (za.size() != mass.size()) {
            std::stringstream ss;
            ss << "Mass table '" << file_name.string() << "' is corrupted";
            mclog::fatal(ss);
        }

        for (size_t i = 0; i < za.size(); ++i)
            this->_table.insert({ za[i], (double)mass[i] });
    }


    double MassTable::getMass(int za) {
        auto table_iter = this->_table.find(za);
        if (table_iter == this->_table.end()) {
            std::stringstream ss;
            ss << "No such isotope '" << za << "' in library '" << Define::Neutron::getInstance().library() << "'";
            mclog::warning(ss);
            double mass = Nucleus::MassTableHandler::getInstance().getMass(za) / constants::ATOMIC_MASS_UNIT;
            this->_table.insert({ za, mass });
            return mass;
        }
        else
            return table_iter->second;
    }


}