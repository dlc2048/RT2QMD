/**
 * @file    module/hadron/projectile.hpp
 * @brief   Aligned mass table for heavy ion projectile
 * @author  CM Lee
 * @date    04/02/2024
 */

#pragma once

#include "singleton/singleton.hpp"
#include "device/memory_manager.hpp"

#include "projectile.cuh"
#include "nucleus.hpp"


namespace Hadron {
    namespace Projectile {


        /**
        * @brief Mass ratio table for range multiplier
        */
        class MassRatioTable : 
            public Singleton<MassRatioTable>, 
            public mcutil::DeviceMemoryHandlerInterface {
            friend class Singleton<MassRatioTable>;
        private:
            std::vector<double> _table_host;  //! @brief Projectile mass table, host side
            float*              _table_dev;   //! @brief Projectile mass table, device side pointer


            MassRatioTable();


            ~MassRatioTable();


        public:


            /**
            * @brief Device memory handle
            * @return CUdeviceptr handle
            */
            CUdeviceptr deviceptr();


            /**
            * @brief Get the mass ratio of projectile against to reference projectile, host side
            * @param za Projectile ZA number
            *
            * @return Mass ratio of projectile (this projectile / reference projectile)
            */
            double getRatio(int za) const;


            /**
            * @brief Get the ZA number of the reference projectile
            * @param za Projectile ZA number
            *
            * @return ZA number of the reference projectile
            */
            int referenceZA(int za) const;


            /**
            * @brief Get the spin of the reference projectile
            * @param za Projectile ZA number
            *
            * @return Spin of the reference projectile, multiplied by 2
            */
            int referenceSpin(int za) const;


        };


        /**
        * @brief Reference projectile for dedx and range calculation
        */
        class ProjectileRef {
        private:
            double _mass;
            int    _spin;
            int    _charge;
        public:
            ProjectileRef(int za);
            double mass()   const { return this->_mass; }
            int    spin()   const { return this->_spin; }
            int    charge() const { return this->_charge; }
        };


    }
}