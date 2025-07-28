#pragma once


namespace tally {


    typedef enum DENSITY_TYPE {
        DENSITY_DEPO       = 0,   // Energy deposition [MeV/cm3]
        DENSITY_DOSE       = 1,   // Physical dose [MeV/g]
        DENSITY_RBEDOSE    = 2,   // RBE weighted dose [MeV/g]
        DENSITY_LETD       = 3,   // Dose-averaged LET [MeV/cm]
        DENSITY_ACTIVATION = 4,   // Low neutron activation [#/cm3]
        DENSITY_SPALLATION = 5,   // Fast neutron & generic ion spallation [#/cm3]
    } DENSITY_TYPE;


    typedef enum CROSS_TYPE {
        CROSS_FLUENCE = 0,     // Particle fluence [mu weighted #/cm2]
        CROSS_CURRENT = 1      // Particle current [#/cm2]
    } CROSS_TYPE;

    
    typedef enum MULTIPLIER_TYPE {
        MULTIPLIER_NONE = 0,   // None [Dimensionless]
        MULTIPLIER_LET  = 1,   // Linear Energy Transfer [MeV/cm]
        MULTIPLIER_RBE  = 2,   // Relative Biological Effectiveness [Dimensionless]
    } MULTIPLIER_TYPE;


    typedef enum ENERGY_TYPE {
        ENERGY_LINEAR = 0,  // linear uniform
        ENERGY_LOG    = 1,  // log uniform
    } ENERGY_TYPE;


    /* data structure (host control) */ 


    constexpr int MESH_MEMCPY_AUTO_KERNEL_THRESHOLD = 64 * 64 * 64;


    enum class MESH_MEMCPY_POLICY {
        MEMCPY_AUTO   = 0,  // automatic decision
        MEMCPY_HOST   = 1,  // using host function (CPU)
        MEMCPY_KERNEL = 2   // using device kernel (GPU)
    };


    enum class MESH_MEMORY_STRUCTURE {
        MEMORY_DENSE      = 0,   // dense format
        MEMORY_SPARSE_COO = 1    // sparse format (COO)
    };


}