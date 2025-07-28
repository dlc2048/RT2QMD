
#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "projectile.hpp"


namespace Hadron {
    namespace Projectile {


        MassRatioTable::MassRatioTable() {
            Nucleus::MassTableHandler& nuc_mass = Nucleus::MassTableHandler::getInstance();
            // host side
            this->_table_host = std::vector<double>(TABLE_MAX_Z * TABLE_MAX_A);
            for (int i = 0; i < TABLE_MAX_Z; ++i) {
                int    z        = i + 1;
                int    za_ref   = REFERENCE_ZA[i];
                double mass_ref = z == 1 ? (double)constants::MASS_PROTON : nuc_mass.getMass(za_ref);
                for (int j = 0; j < TABLE_MAX_A; ++j) {
                    int    a    = j + 1;
                    int    za   = 1000 * z + a;
                    double mass = z == 1 && a == 1 ? (double)constants::MASS_PROTON : nuc_mass.getMass(za);
                    this->_table_host[i * TABLE_MAX_A + j] = mass / mass_ref;
                }
            }
            // device side
            std::vector<float> ftable = mcutil::cvtVectorDoubleToFloat(this->_table_host);
            this->_memoryUsageAppend(mcutil::cudaMemcpyVectorToDevice(ftable, &this->_table_dev));
        }


        MassRatioTable::~MassRatioTable() {
            /*
            CUDA_CHECK(cudaFree(this->_table_dev));
            */
        }


        CUdeviceptr MassRatioTable::deviceptr() {
            return reinterpret_cast<CUdeviceptr>(this->_table_dev);
        }


        double MassRatioTable::getRatio(int za) const {
            int z = physics::getZnumberFromZA(za);
            int a = physics::getAnumberFromZA(za);
            assert(z <= TABLE_MAX_Z && z >= 1);
            assert(a <= TABLE_MAX_A && a >= 1);
            int idx = (z - 1) * TABLE_MAX_A + (a - 1);
            return this->_table_host[idx];
        }


        int MassRatioTable::referenceZA(int za) const {
            int z = physics::getZnumberFromZA(za);
            assert(z <= TABLE_MAX_Z && z >= 1);
            return REFERENCE_ZA[z - 1];
        }


        int MassRatioTable::referenceSpin(int za) const {
            int z = physics::getZnumberFromZA(za);
            assert(z <= TABLE_MAX_Z && z >= 1);
            return REFERENCE_SPIN[z - 1];
        }


        ProjectileRef::ProjectileRef(int za) {
            int z = physics::getZnumberFromZA(za);
            assert(z <= TABLE_MAX_Z && z >= 1);
            this->_charge = z;
            this->_spin   = MassRatioTable::getInstance().referenceSpin(za);
            int za_ref    = MassRatioTable::getInstance().referenceZA(za);
            int a         = physics::getAnumberFromZA(za_ref);
            this->_mass   = z == 1 && a == 1 
                ? (double)constants::MASS_PROTON 
                : Nucleus::MassTableHandler::getInstance().getMass(za_ref);
        }


    }
}