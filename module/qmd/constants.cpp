
#include "constants.hpp"

#define _USE_MATH_DEFINES

#include <cmath>

#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif


namespace RT2QMD {
    namespace constants {


        ParameterInitializer::ParameterInitializer() :
            _rmass(0.938),
            _ebin (-16.0 * 1e-3),
            _esymm(25.0 * 1e-3),
            _rpot (1.0 / 3.0) {

            // Skyrme

            double pfer = ::constants::HBARC * 1e10 * std::pow(3.0 / 2.0 * M_PI * M_PI * RHO_SATURATION, this->_rpot);
            double efer = pfer * pfer / 2.0 / this->_rmass;
            double t3   = 8.0 / 3.0 / this->_rpot 
                / std::pow(RHO_SATURATION, this->_rpot + 1.0) 
                * (efer / 5.0 - this->_ebin);
            double t0   = -16.0 / 15.0 * efer / RHO_SATURATION 
                - (this->_rpot + 1.0) * t3 * std::pow(RHO_SATURATION, this->_rpot);
            double aaa  = 3.0 / 4.0 * t0 * RHO_SATURATION;
            double bbb  = 3.0 / 8.0 * t3 * (2.0 + this->_rpot) * std::pow(RHO_SATURATION, this->_rpot + 1.0);

            // Hamiltonian coeffs

            double h_skyrme_c0 = aaa / (RHO_SATURATION * std::pow(4.0 * M_PI * WAVE_PACKET_WIDTH, 1.5) * 2.0);
            double h_skyrme_c3 = bbb / (
                std::pow(RHO_SATURATION, this->_rpot + 1.0) *
                std::pow(4.0 * M_PI * WAVE_PACKET_WIDTH, 1.5 * (this->_rpot + 1.0)) * 
                (this->_rpot + 2.0)
                );
            double h_symmetry  = this->_esymm / (RHO_SATURATION 
                * std::pow(4.0 * M_PI * WAVE_PACKET_WIDTH, 1.5) * 2.0);
            double h_coulomb   = CCOUL / 2.0;

            // Hamiltonian distance

            double d_skyrme_c0   = 1.0 / 4.0 / WAVE_PACKET_WIDTH;
            double d_skyrme_c0_s = std::sqrt(d_skyrme_c0);
            double d_coulomb     = 2.0 / sqrt(4.0 * M_PI * WAVE_PACKET_WIDTH);

            // Clustering
            double cpf2 = pow(1.5 * M_PI * M_PI * pow(4.0 * M_PI * WAVE_PACKET_WIDTH, -1.5), 2.0 / 3.0);
            cpf2 *= ::constants::HBARC * ::constants::HBARC * 1e20;

            // Hamiltonian gradient

            double g_skyrme_c0 = -h_skyrme_c0 / (2.0 * WAVE_PACKET_WIDTH);
            double g_skyrme_c3 = -h_skyrme_c3 / (4.0 * WAVE_PACKET_WIDTH) * (this->_rpot + 1.0);
            double g_symmetry  = -h_symmetry  / (2.0 * WAVE_PACKET_WIDTH);

            // Ground state nuclei

            double gs_cd = 1.0 / std::pow(4.0 * M_PI * WAVE_PACKET_WIDTH, 1.5);
            double gs_c0 = h_skyrme_c0 * 2.0;
            double gs_c3 = h_skyrme_c3 * (this->_rpot + 2.0);
            double gs_cs = h_symmetry  * 2.0;
            double gs_cl = h_coulomb   * 2.0;

            // copy to device ptr

            CUDA_CHECK(setSymbolHCoeffs(
                (float)h_skyrme_c0, 
                (float)h_skyrme_c3, 
                (float)h_symmetry, 
                (float)h_coulomb
            ));
            CUDA_CHECK(setSymbolHDistance(
                (float)d_skyrme_c0,
                (float)d_skyrme_c0_s,
                (float)d_coulomb
            ));
            CUDA_CHECK(setSymbolClusterCoeffs(
                (float)cpf2
            ));
            CUDA_CHECK(setSymbolHGradient(
                (float)g_skyrme_c0,
                (float)g_skyrme_c3,
                (float)g_symmetry
            ));
            CUDA_CHECK(setSymbolGroundNucleusCoeffs(
                (float)gs_cd,
                (float)gs_c0,
                (float)gs_c3,
                (float)gs_cs,
                (float)gs_cl
            ));
        }


    }
}