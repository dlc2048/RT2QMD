/**
 * @file    module/genericion/auxiliary.hpp
 * @brief   Auxiliary tables for generic ion
 * @author  CM Lee
 * @date    06/21/2024
 */

#pragma once

#include <vector>

#include "singleton/singleton.hpp"

#include "hadron/nucleus.hpp"
#include "particles/define.hpp"
#include "deexcitation/auxiliary.hpp"

#include "nuc_secondary/secondary.cuh"


namespace genion {

#ifdef RT2QMD_STANDALONE
    constexpr int GENION_MAX_Z = Hadron::Projectile::TABLE_MAX_Z;
    constexpr int GENION_MAX_A = Hadron::Projectile::TABLE_MAX_A;
#endif

    struct IsoProjectile {
        int    z;     // Atomic number
        int    a;     // Mass number
        double mu;    // Mass per nucleon
        double mr;    // Mass per nucleon ratio against reference projectile
        int    spin;  // Nucleus spin [hbar/2]


        double mass() const { return this->mu * (double)this->a; }
        int    za()   const { return this->z * 1000 + this->a; }


    };


    class IsoProjectileTable :
        public Singleton<IsoProjectileTable>,
        public mcutil::DeviceMemoryHandlerInterface {
        friend class Singleton<IsoProjectileTable>;
    private:
        // host side
        std::vector<IsoProjectile> _iso_proj[GENION_MAX_Z];
        int     _offset_host[GENION_MAX_Z];
        int     _table_size;
        // device side
        int*    _iso_offset_dev;
        uchar2* _list_za_dev;
        float*  _list_m_dev;
        float*  _list_mu_dev;
        float*  _list_mr_dev;
        int*    _list_spin_dev;


        IsoProjectileTable();


        void _initDeviceMemory();


    public:


        ~IsoProjectileTable();


        bool exist(int za) const;


        std::vector<int> listProjectileZA() const;


        std::vector<double> listProjectileMass() const;


        const IsoProjectile& heaviestProjectile(int z) const { return this->_iso_proj[z - 1].back(); }


        const IsoProjectile& refProjectile(int z) const;


        const IsoProjectile& projectile(int z, int a) const;


        int offset(int z) const { return this->_offset_host[z - 1]; }


        int index(int za) const;


        int  tableSize() const { return this->_table_size; }
        int*    ptrProjectileOffset() const { return this->_iso_offset_dev; }
        uchar2* ptrProjectileZA()     const { return this->_list_za_dev;    }
        float*  ptrProjectileMass()   const { return this->_list_m_dev;     }
        float*  ptrProjectileMU()     const { return this->_list_mu_dev;    }
        float*  ptrProjectileMR()     const { return this->_list_mr_dev;    }
        int*    ptrProjectileSpin()   const { return this->_list_spin_dev;  }


    };


}