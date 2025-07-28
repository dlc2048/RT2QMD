
#include "nuclear_level.cuh"


namespace deexcitation {


    __constant__ bool USE_SIMPLE_LEVEL_DENSITY = true;


    __host__ cudaError_t setSimpleLevelDensity(bool use_it) {
        return cudaSuccess;
    }


}