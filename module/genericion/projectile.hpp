/**
 * @file    module/genericion/projectile.hpp
 * @brief   Generic ion projectile definitions
 * @author  CM Lee
 * @date    06/19/2024
 */

#pragma once

#include <memory>

#include "particles/projectile_interface.hpp"
#include "hadron/nucleus.hpp"


namespace genion {


    class Projectile : public Define::ProjectileInterface {
    public:


        Projectile(int za);


    };


}