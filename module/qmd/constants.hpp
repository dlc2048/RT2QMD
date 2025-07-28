/**
 * @file    module/qmd/constants.hpp
 * @brief   QMD constants handler (G4QMD)
 * @author  CM Lee
 * @date    02/14/2024
 */

#pragma once

#include "constants.cuh"


namespace RT2QMD {
    namespace constants {

        constexpr double MASS_ELECTRON     = 5.109988e-4;    //! @brief Electron mass [GeV/c^2]
        constexpr double RHO_SATURATION    = 0.168;          //! @brief saturation density                     << G4QMDParameters::rho0


        /**
        * @brief QMD parameter initializer for device memory
        */
        class ParameterInitializer {
        private:
            double _rmass;  //! @brief mass of nucleus [GeV/c^2]
            double _ebin;   //! @brief bounding energy [GeV]
            double _esymm;  //! @brief symmetric energy [GeV]
            double _rpot;
        public:
            ParameterInitializer();
        };


    }
}