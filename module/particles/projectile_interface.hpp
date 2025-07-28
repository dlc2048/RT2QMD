/**
 * @file    module/particles/projectile_interface.hpp
 * @brief   Projectile interface for phyisical quantity definition
 * @author  CM Lee
 * @date    04/03/2024
 */

#pragma once


namespace Define {


    class ProjectileInterface {
    protected:
        double _mass;     //! @brief Mass of projectile [MeV/c^2]
        int    _spin;     //! @brief Spin of projectile, multiplied by 2
        int    _charge;   //! @brief Elementary charge of projectile
    public:
        double mass()   const { return this->_mass; }
        int    spin()   const { return this->_spin; }
        int    charge() const { return this->_charge; }
    };


}