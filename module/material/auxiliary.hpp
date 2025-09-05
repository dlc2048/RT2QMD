#pragma once

#include <filesystem>
#include <map>
#include <set>
#include <regex>
#include <assert.h>

#include "singleton/singleton.hpp"
#include "fortran/fortran.hpp"
#include "prompt/env.hpp"
#include "particles/define.hpp"


namespace mat {


    constexpr double ROOM_TEMPERATURE = 293.6;


    static const std::filesystem::path HOME 
        = std::filesystem::path("resource") / std::filesystem::path("material");


    typedef struct Isotope {
        int         za;       // ZA number
        int         isom;     // Isomeric number
        float       temp;     // Temperature
        std::string sab;      // S(a,b) name
        bool        fissile;  // Fissile flag


        bool operator==(const Isotope& rhs) const {
            return (
                this->za      == rhs.za && 
                this->isom    == rhs.isom && 
                this->temp    == rhs.temp && 
                this->sab     == rhs.sab &&
                this->fissile == rhs.fissile
            );
        }


        bool operator<(const Isotope& rhs) const {
            if (this->za == rhs.za) {
                if (this->isom == rhs.isom) {
                    if (this->sab == rhs.sab) {
                        return this->temp < rhs.temp;
                    }
                    else
                        return this->sab < rhs.sab;
                }
                else
                    return this->isom < rhs.isom;
            }
            else
                return this->za < rhs.za;
        }


        Isotope() :
            za(0), isom(0), temp((float)ROOM_TEMPERATURE), sab("") {}


    } Isotope;


    class AtomNamelist : public Singleton<AtomNamelist> {
        friend class Singleton<AtomNamelist>;
    private:
        static const std::filesystem::path _name_file;

        std::map<int, std::pair<std::string, std::string>> _namelist;
        std::map<std::string, int> _symbol_to_z;
        std::map<std::string, int> _name_to_z;


        AtomNamelist();


    public:


        const std::string& symbol(int z) const;


        const std::string& name(int z)   const;


        int findSymbol(const std::string& symbol) const;


        int findName(const std::string& name) const;


    };


    class NaturalAbundance : public Singleton<NaturalAbundance> {
        friend class Singleton<NaturalAbundance>;
    private:
        static const std::filesystem::path _nist_file;


        std::map<int, std::map<int, double>> _table;   // Z - { za, fracton }


        NaturalAbundance();


    public:


        const std::map<int, double>& composition(int z) const;


    };


    class ENDFIsotopeTable : public Singleton<ENDFIsotopeTable> {
        friend class Singleton<ENDFIsotopeTable>;
    private:


        std::set<Isotope> _isotope_list;


        ENDFIsotopeTable() {}


    public:


        void insert(const Isotope& new_isotope);


        bool valid(const Isotope& isotope);


        std::vector<Isotope> sabTarget(const std::string& sab);


    };


    class IonizationEnergy : public Singleton<IonizationEnergy> {
        friend class Singleton<IonizationEnergy>;
    private:
        static const std::filesystem::path _ie_file;

        std::vector<double> _mie_arr;


        IonizationEnergy();


    public:


        double get(int z, bool is_gas) const;


    };


    class FermiVelocity : public Singleton<FermiVelocity> {
        friend class Singleton<FermiVelocity>;
    private:
        static const std::filesystem::path _fv_file;

        std::vector<double> _fv_arr;


        FermiVelocity();


    public:


        double get(int z) const;


    };


    int findIsotope(const std::string& name);


}