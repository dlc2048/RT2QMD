
#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "handler.hpp"


namespace deexcitation {


    DeviceMemoryHandler::DeviceMemoryHandler() {
        mclog::debug("Initialize nuclear de-excitation data ...");
        // Cameron corrections
        CameronCorrection& correction = CameronCorrection::getInstance();

        CUDA_CHECK(fission::setCameronSpinPairingCorrections(
            correction.ptrSpinPairingProton(),
            correction.ptrSpinPairingNeutron()
        ));
        CUDA_CHECK(fission::setCameronPairingCorrections(
            correction.ptrPairingProton(),
            correction.ptrPairingNeutron()
        ));
        CUDA_CHECK(fission::setCameronSpinCorrections(
            correction.ptrSpinProton(),
            correction.ptrSpinNeutron()
        ));

        // Coulomb radius
        CoulombBarrier& coulomb_barrier = CoulombBarrier::getInstance();

        CUDA_CHECK(setCoulombBarrierRadius(coulomb_barrier.ptrCoulombRadius()));

        // Emitted particle info
        // mass 
        Nucleus::MassTableHandler& host_mass_table 
            = Nucleus::MassTableHandler::getInstance();

        std::vector<double> m;
        std::vector<double> m2;
        m.push_back(0.0);
        m.push_back(0.0);
        m.push_back(constants::MASS_NEUTRON);
        m.push_back(constants::MASS_PROTON);
        m.push_back(host_mass_table.getMass(1002));  // deuteron
        m.push_back(host_mass_table.getMass(1003));  // triton
        m.push_back(host_mass_table.getMass(2003));  // He3
        m.push_back(host_mass_table.getMass(2004));  // He4
        m.push_back(constants::MASS_NEUTRON * 2.0);  // 2n
        m.push_back(constants::MASS_PROTON  * 2.0);  // 2p

        for (double mass : m)
            m2.push_back(mass * mass);

        // coulomb rho
        std::vector<double> crho;

        crho.push_back(0.0);
        crho.push_back(0.0);
        crho.push_back(coulomb_barrier.coulombBarrierRadius(0, 1));  // neutron
        crho.push_back(coulomb_barrier.coulombBarrierRadius(1, 1));  // proton
        crho.push_back(coulomb_barrier.coulombBarrierRadius(1, 2));  // deuteron
        crho.push_back(coulomb_barrier.coulombBarrierRadius(1, 3));  // triton
        crho.push_back(coulomb_barrier.coulombBarrierRadius(2, 3));  // He3
        crho.push_back(coulomb_barrier.coulombBarrierRadius(2, 4));  // He4
        crho.push_back(0.0);
        crho.push_back(0.0);

        for (double& cr : crho)
            cr *= 0.4;

        // memory
        std::vector<float> m_float    = mcutil::cvtVectorDoubleToFloat(m);
        std::vector<float> m2_float   = mcutil::cvtVectorDoubleToFloat(m2);
        std::vector<float> crho_flaot = mcutil::cvtVectorDoubleToFloat(crho);

        mcutil::DeviceVectorHelper m_vec(m_float);
        mcutil::DeviceVectorHelper m2_vec(m2_float);
        mcutil::DeviceVectorHelper crho_vec(crho_flaot);

        this->_memoryUsageAppend(m_vec.memoryUsage());
        this->_memoryUsageAppend(m2_vec.memoryUsage());
        this->_memoryUsageAppend(crho_vec.memoryUsage());

        this->_dev_m    = m_vec.address();
        this->_dev_m2   = m2_vec.address();
        this->_dev_crho = crho_vec.address();

        CUDA_CHECK(setEmittedParticleMass(this->_dev_m, this->_dev_m2));
        CUDA_CHECK(setEmittedParticleCBRho(this->_dev_crho));

        CUDA_CHECK(setFissionFlag(Define::IonInelastic::getInstance().activateFission()));

        // nucleus symbol
        mclog::debug("Link de-excitation device symbol ...");
        CUDA_CHECK(deexcitation::setMassTableHandle(Nucleus::MassTableHandler::getInstance().deviceptr()));
        CUDA_CHECK(deexcitation::setStableTable(Nucleus::ENSDFTable::getInstance().ptrLongLivedNucleiTable()));
    }


    DeviceMemoryHandler::~DeviceMemoryHandler() {
        mclog::debug("Destroy device memory of nuclear de-excitation data ...");
        mcutil::DeviceVectorHelper(this->_dev_m).free();
        mcutil::DeviceVectorHelper(this->_dev_m2).free();
        mcutil::DeviceVectorHelper(this->_dev_crho).free();
    }


    void DeviceMemoryHandler::summary() const {

        Define::IonInelastic& ie_def = Define::IonInelastic::getInstance();

        double bytes_to_mib = 1.0 / (double)mcutil::MEMSIZE_MIB;

        mclog::info("*** Nuclear De-Excitation Summaries ***");
        mclog::printVar("Evaporation cutoff    ", ie_def.evaporationCutoff(), "s");
        mclog::printVar("Fission branch        ", ie_def.activateFission() ? "On" : "Off");
        mclog::printVar("Memory usage          ", this->memoryUsage() * bytes_to_mib, "MiB");
        mclog::print("");
    }


}