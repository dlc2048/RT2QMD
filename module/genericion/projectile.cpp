
#include "projectile.hpp"


namespace genion {


    Projectile::Projectile(int za) {
        const Nucleus::ENSDFTable&  table  = Nucleus::ENSDFTable::getInstance();
        const Nucleus::ENSDFRecord& record = table.get(za);
        // get mass
        double mass   = Nucleus::MassTableHandler::getInstance().getMass(za);
        this->_charge = physics::getZnumberFromZA(za);
        this->_spin   = record.spin();
        this->_mass   = mass;
    }


}