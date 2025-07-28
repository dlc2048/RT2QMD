/**
 * @file    module/hadron/nucleus.hpp
 * @brief   Functions that encapsulate a mass table (G4INCLNuclearMassTable.hh)
 *          and ENSDF data table (ENSDFSTATE.dat)
 * @author  CM Lee
 * @date    02/14/2024
 */

#pragma once

#include <set>
#include <map>
#include <string>
#include <filesystem>

#include <cuda_runtime.h>

#include "nucleus.cuh"

#include "singleton/singleton.hpp"
#include "prompt/env.hpp"
#include "device/memory.hpp"
#include "physics/physics.hpp"
#include "particles/define.hpp"


namespace Nucleus {


    inline const std::filesystem::path HOME = std::filesystem::path("resource/hadron");


    class MassRecord {
        friend std::istream& operator>>(std::istream& in, MassRecord& record);
        friend bool operator<(const MassRecord& lhs, const MassRecord& rhs);
        friend bool operator==(const MassRecord& lhs, const MassRecord& rhs);
    private:
        int    _a;
        int    _z;
        double _excess;
    public:
        MassRecord();
        MassRecord(const int a, const int z, const double e);
        int    a() const { return this->_a; }
        int    z() const { return this->_z; }
        double e() const { return this->_excess; }
    };


    /**
    * @brief Host and device memory for nuclear mass table.
    *        Note that this table presents mass of nucleus without orbital electron
    */
    class MassTableHandler : 
        public Singleton<MassTableHandler>,
        public mcutil::DeviceMemoryHandlerInterface {
        friend class Singleton<MassTableHandler>;
    private:
        static const std::filesystem::path _mass_file;  //! @brief Filename of INCL real nuclear masses data
        std::map<int, double> _table_host;  //! @brief Nuclear mass table, host side
        size_t                _zmax;        //! @brief Maximum z number of table
        MassTable* _table_dev;   //! @brief Nuclear mass table, device side pointer


        MassTableHandler();


        ~MassTableHandler();


    public:


        /**
        * @brief Device memory handle
        * @return CUdeviceptr handle
        */
        CUdeviceptr deviceptr();


        /**
        * @brief Get mass of nucleus, host side
        * @param za Nucleus ZA number
        * 
        * @return Mass of nucleus [MeV/c^2]
        */
        double getMass(int za) const;


    };


    class ENSDFRecord {
        friend std::istream& operator>>(std::istream& in, ENSDFRecord& record);
        friend bool operator<(const ENSDFRecord& lhs, const ENSDFRecord& rhs);
        friend bool operator==(const ENSDFRecord& lhs, const ENSDFRecord& rhs);
    private:
        double _excitation;  // Excitation energy [MeV]
        double _life_time;   // Mean life time [seconds]
        int    _spin;        // Spin [h_bar/2]
        double _dmoment;     // Dipole magnetic moment [joule/tesla]
    public:
        ENSDFRecord();
        double excitation() const { return this->_excitation; }
        double lifeTime()   const { return this->_life_time; }
        int    spin()       const { return this->_spin; }
        double dmoment()    const { return this->_dmoment; }
    };




    class ENSDFTable : 
        public Singleton<ENSDFTable>,
        public mcutil::DeviceMemoryHandlerInterface {
        friend class Singleton<ENSDFTable>;
    private:
        static const std::filesystem::path _ensdf_file;  //! @brief Filename of ENSDF data
        std::map<int, std::vector<ENSDFRecord>> _table_host;  //! @brief ENSDF table, host side

        LongLivedNucleiTable _table_dev;  //! @brief long-live particles


        ENSDFTable();


    public:


        ~ENSDFTable();


        size_t numberOfSate(int za) const;


        const ENSDFRecord& get(int za, int level = 0) const;


        std::vector<int> isotopes(int z) const;


        const LongLivedNucleiTable& ptrLongLivedNucleiTable() const { return this->_table_dev; }


    };


    bool compareA(const MassRecord& lhs, const MassRecord& rhs);


    bool compareZ(const MassRecord& lhs, const MassRecord& rhs);


    /**
    * @brief Empirical total electron mass (binding + mass) 
    *        from Atomic Mass Unit 2012
    * @param z Atomic number of nuclei
    * 
    * @return Net electron mass [MeV]
    */
    double electronMass(int z);


}