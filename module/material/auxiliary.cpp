
#include "auxiliary.hpp"


namespace mat {


    const std::filesystem::path AtomNamelist::_name_file = std::filesystem::path("elements.dat");


    AtomNamelist::AtomNamelist() {
        namespace fp = std::filesystem;
        std::string home = mcutil::getMCRT2HomePath();
        if (home.empty())
            mclog::fatal("Environment variable 'MCRT2_HOME' is missing");
        fp::path file(home);
        file = file / HOME / this->_name_file;
        std::ifstream data(file.string());

        if (data.bad())
            mclog::fatalFileNotExist(file.string());

        int z;
        std::string symbol;
        std::string name;
        while (!data.eof()) {
            data >> z;
            data >> symbol;
            data >> name;
            this->_namelist.insert({ z, {symbol, name} });

            // lowercase
            std::transform(symbol.begin(), symbol.end(), symbol.begin(), 
                [](unsigned char c) { return std::tolower(c); });
            std::transform(name.begin(),   name.end(),   name.begin(),
                [](unsigned char c) { return std::tolower(c); });

            this->_symbol_to_z.insert({ symbol, z });
            this->_name_to_z.insert({ name, z });
        }
    }


    const std::string& AtomNamelist::symbol(int z) const {
        return this->_namelist.find(z)->second.first;
    }


    const std::string& AtomNamelist::name(int z) const {
        return this->_namelist.find(z)->second.second;
    }


    int AtomNamelist::findSymbol(const std::string& symbol) const {
        auto iter = this->_symbol_to_z.find(symbol);
        if (iter == this->_symbol_to_z.end())
            return -1;
        else
            return iter->second;
    }


    int AtomNamelist::findName(const std::string& names) const {
        auto iter = this->_name_to_z.find(names);
        if (iter == this->_name_to_z.end())
            return -1;
        else
            return iter->second;
    }


    const std::filesystem::path NaturalAbundance::_nist_file = std::filesystem::path("NIST_abundance.txt");


    NaturalAbundance::NaturalAbundance() {
        namespace fp = std::filesystem;

        this->_table.insert({ 0, std::map<int, double>() });  // default empty data

        std::string home = mcutil::getMCRT2HomePath();
        if (home.empty())
            mclog::fatal("Environment variable 'MCRT2_HOME' is missing");
        fp::path file(home);
        file = file / HOME / this->_nist_file;

        std::ifstream data(file.string());

        if (data.bad())
            mclog::fatalFileNotExist(file.string());

        int    current_z = 0;
        double f_cumul   = 0.0;
        std::map<int, double> composition;
        std::string line;
        while (std::getline(data, line)) {
            if (line.empty())
                continue;

            std::string f_str = line.substr(40);  // strip
            f_str.erase(std::remove_if(f_str.begin(), f_str.end(), ::isspace), f_str.end());
            std::deque<std::string> flist = mcutil::split(f_str, '(', 1);

            int z;
            int a;
            double fraction = 0.0;

            std::stringstream z_converter(line.substr(0, 3));
            std::stringstream a_converter(line.substr(8, 3));
            if (!flist.empty()) {
                std::stringstream f_converter(flist[0]);
                f_converter >> fraction;
            }
            z_converter >> z;
            a_converter >> a;

            if (!z_converter.fail()) {  // new Z
                // normalize
                for (auto& seg : composition)
                    seg.second /= f_cumul;

                this->_table.insert({ current_z, composition });
                composition.clear();

                current_z = z;
                f_cumul   = 0.0;
            }

            if (fraction > 0.0) {
                composition.insert({ 1000 * z + a, fraction });
                f_cumul += fraction;
            }
        }

        // last
        if (!composition.empty()) {
            for (auto& seg : composition)
                seg.second /= f_cumul;
            this->_table.insert({ current_z, composition });
        }
    }


    const std::map<int, double>& NaturalAbundance::composition(int z) const {
        auto iter = this->_table.find(z);
        if (iter == this->_table.end())
            iter = this->_table.find(0);
        return iter->second;
    }


    void ENDFIsotopeTable::insert(const Isotope& new_isotope) {
        this->_isotope_list.insert(new_isotope);
    }


    bool ENDFIsotopeTable::valid(const Isotope& isotope) {
        Isotope isotope_zero = isotope;
        isotope_zero.temp = 0.f;
        auto iter = this->_isotope_list.lower_bound(isotope_zero);
        return iter->za == isotope.za && iter->isom == isotope.isom;
    }


    std::vector<Isotope> ENDFIsotopeTable::sabTarget(const std::string& sab) {
        std::vector<Isotope> target;
        for (const auto& isotope : this->_isotope_list) {
            if (isotope.sab == sab)
                target.push_back(isotope);
        }
        return target;
    }


    const std::filesystem::path IonizationEnergy::_ie_file = std::filesystem::path("mean_ie.bin");


    IonizationEnergy::IonizationEnergy() {
        namespace fp = std::filesystem;
        std::string home = mcutil::getMCRT2HomePath();
        if (home.empty())
            mclog::fatal("Environment variable 'MCRT2_HOME' is missing");
        fp::path file(home);
        file = file / HOME / this->_ie_file;

        mcutil::FortranIfstream data(file.string());
        this->_mie_arr = data.read<double>();
    }


    double IonizationEnergy::get(int z, bool is_gas) const {
        assert(z >= 1);
        double iev = this->_mie_arr[z - 1];
        if (Define::Electron::getInstance().ieFudge()) {
            // SLAC-265 Berger & Seltzer's fudge (table 6 in ref 59)
            switch (z) {
            case 1:
                iev = 19.2;
                break;
            case 6:
                iev = is_gas ? 70.0 : 81.0;
                break;
            case 7:
                iev = 82.0;
                break;
            case 8:
                iev = is_gas ? 97.0 : 106.0;
                break;
            case 9:
                iev = 112.0;
                break;
            case 17:
                iev = 180.0;
                break;
            default:
                iev *= 1.13;
                break;
            }
        }
        return iev;
    }


    const std::filesystem::path FermiVelocity::_fv_file = std::filesystem::path("fermi_velocity.bin");


    FermiVelocity::FermiVelocity() {
        namespace fp = std::filesystem;
        std::string home = mcutil::getMCRT2HomePath();
        if (home.empty())
            mclog::fatal("Environment variable 'MCRT2_HOME' is missing");
        fp::path file(home);
        file = file / HOME / this->_fv_file;

        mcutil::FortranIfstream data(file.string());
        this->_fv_arr = data.read<double>();
    }


    double FermiVelocity::get(int z) const {
        assert(z >= 1);
        int iz = z - 1;
        if (iz >= this->_fv_arr.size())
            iz = (int)this->_fv_arr.size() - 1;
        return this->_fv_arr[iz];
    }


    int findIsotope(const std::string& name) {
        static const std::regex r_num       (R"(^[0-9]+$)");
        static const std::regex r_str       (R"(^[A-Za-z]+$)");
        static const std::regex r_strnum    (R"(^([A-Za-z]+)([0-9]+)$)");
        static const std::regex r_strdashnum(R"(^([A-Za-z]+)-([0-9]+)$)");

        AtomNamelist& atom_list = AtomNamelist::getInstance();

        int za = -1;  // default
        std::smatch match;

        if (std::regex_match(name, match, r_num)) {  // ZZZAAA
            za = std::stoi(match.str(0));
        }
        else if (std::regex_match(name, match, r_str)) {  // name
            int z;
            z = atom_list.findSymbol(match.str(0));
            if (z >= 0)
                za = z * 1000;
            else {
                z = atom_list.findName(match.str(0));
                if (z >= 0)
                    za = z * 1000;
            }
        }
        else if (std::regex_match(name, match, r_strnum)) {  // nameAAA
            int z;
            z = atom_list.findSymbol(match.str(1));
            if (z < 0)
                z = atom_list.findName(match.str(1));

            if (z >= 0) {
                int a = std::stoi(match.str(2));
                if (a > 0 && a < 1000)
                    za = z * 1000 + a;
            }
        }
        else if (std::regex_match(name, match, r_strdashnum)) {  // name-AAA
            int z;
            z = atom_list.findSymbol(match.str(1));
            if (z < 0)
                z = atom_list.findName(match.str(1));

            if (z >= 0) {
                int a = std::stoi(match.str(2));
                if (a > 0 && a < 1000)
                    za = z * 1000 + a;
            }
        }

        return za;
    }


}