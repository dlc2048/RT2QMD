/**
 * @file    module/hadron/atomic.hpp
 * @brief   ENDF atomic mass table
 * @author  CM Lee
 * @date    01/31/2025
 */

#pragma once

#include <filesystem>
#include <map>

#include "singleton/singleton.hpp"
#include "fortran/fortran.hpp"
#include "prompt/env.hpp"

#include "particles/define.hpp"

#include "nucleus.hpp"


namespace Atomic {


    inline const std::filesystem::path HOME = std::filesystem::path("resource/neutron");  // Using ENDF mass table


    class MassTable : public Singleton<MassTable> {
        friend class Singleton<MassTable>;
    private:
        static const std::filesystem::path _mass_file;  //! @brief Filename of ENDF atomic mass table
        std::map<int, double> _table;  //! @brief Atomic mass table


        MassTable();


    public:


        double getMass(int za);


    };


}