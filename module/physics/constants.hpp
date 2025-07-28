#pragma once

#include <map>
#include "singleton/singleton.hpp"
#include "constants.cuh"


namespace constants {

    constexpr double AVOGADRO           = 0.602214076;     //! @brief Avogadro number [#/mol/barn]
    constexpr double GAS_DENSITY_THRES  = 0.1;             //! @brief Gas density threshold [g/cm^3]
    constexpr double XS_MB_CM           = 1660.5655;       //! @brief XS conversion from [mb/atom] to [cm^2/g]

    constexpr double DALTON_MEV         = 931.49410242;    //! @brief Dalton unit in MeV [MeV]
    constexpr double BOHR_RADIUS        
        = HBARC / FSC / (double)MASS_ELECTRON;             //! @brief Bohr radius [cm]

    // Tsai's radiation logarithm
    constexpr double RADIATION_LOGARITHM_PRIM[4] = { 6.144, 5.621, 5.805, 5.924 };
    constexpr double RADIATION_LOGARITHM[4]      = { 5.310, 4.790, 4.740, 4.710 };

    // Radiation length
    constexpr double FSC_RCL = FSC * ELECTRON_RADIUS * ELECTRON_RADIUS * BARN;


}