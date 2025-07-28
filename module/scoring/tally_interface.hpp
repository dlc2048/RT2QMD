#pragma once

#include <set>
#include <algorithm>
#include <regex>
#include <map>
#include <list>
#include <string>
#include <functional>

#include <cuda_runtime.h>

#include "device/algorithm.hpp"
#include "device/memory.hpp"
#include "mclog/logger.hpp"
#include "parser/parser.hpp"
#include "fortran/fortran.hpp"

#ifndef RT2QMD_STANDALONE
#include "device/sparse.hpp"
#include "world/world.hpp"
#endif

#include "particles/define.hpp"
#include "transport/buffer.hpp"
#include "physics/physics.hpp"
#include "hadron/projectile.hpp"

#include "tally_struct.h"


#ifdef RT2QMD_STANDALONE


namespace geo {


    // dummy for RT2QMD standalone
    class RegionContext {
    public:
        RegionContext(const std::string& name) {}
        std::string name() const { return ""; }
    };


}


#endif


namespace tally {


    /**
    * @brief Literal unit of densit type context
    */
    inline std::map<DENSITY_TYPE, std::string> DENSITY_UNIT_LITERAL = {
        {DENSITY_TYPE::DENSITY_DEPO,       "MeV/cm3/hist"   },
        {DENSITY_TYPE::DENSITY_DOSE,       "MeV/g/hist"     },
        {DENSITY_TYPE::DENSITY_RBEDOSE,    "MeV/g/hist"     },
        {DENSITY_TYPE::DENSITY_LETD,       "MeV/cm/hist"    },
        {DENSITY_TYPE::DENSITY_ACTIVATION, "#/cm3/hist"     },
        {DENSITY_TYPE::DENSITY_SPALLATION, "#/cm3/hist"     },
        
    };


    /**
    * @brief tally part operator
    */
    enum class OPERATOR_TYPE {
        NONE    = -2,
        NOT     = -3,
        OPERAND = -4
    };

    
    struct EquationElem {
        OPERATOR_TYPE type;
        int           pid;


        EquationElem(OPERATOR_TYPE op_type, const std::string& pid_literal);


        EquationElem(OPERATOR_TYPE op_type, int pid);


        EquationElem(OPERATOR_TYPE op_type);


        EquationElem(const std::string& pid_literal);


    };


    static std::map<std::string, OPERATOR_TYPE> OPERATOR_LITERAL = {
        {"+", OPERATOR_TYPE::NONE },
        {"-", OPERATOR_TYPE::NOT  }
    };


    /**
    * @brief Tally part naming aliasing (std::string to pid)
    * @param part_literal Literal part name
    *
    * @return particle PID (int)
    */
    int tallyPartAlias(const std::string& part_literal);


    bool toLogicalEquation(std::list<EquationElem>& equation);



    class TallyContext {
    protected:
        std::string         _name;  // tally name
        std::string         _unit;  // unit literal string
        std::vector<double> _data;  // tally value
        std::vector<double> _unc;   // tally statistical uncertainty
        double _normalizer;         // tally normalization factor
        bool   _ascii;              // ascii flag (output write)

        mcutil::SAVE_POLICY _save_policy;


        TallyContext(mcutil::ArgInput& args);


        std::vector<double2> _getDataFromDevice(
            double2* device_data,
            size_t   length
        );


        void _writeTallyHeader(mcutil::FortranOfstream& stream) const;


        void _writeTallyHeader(std::ofstream& stream) const;


        void _writeTallyData(mcutil::FortranOfstream& stream, size_t max_length = 0) const;


        void _summaryTallyContext() const;


    public:


        static void _initializeStaticArgs(mcutil::ArgumentCard& card);


        const std::string& name() const { return this->_name; }


        const std::string& unit() const { return this->_unit; }


        void normalize(double total_weight);


        double normalizer() const { return this->_normalizer; }


        void setNormalizer(double normalizer);


        void errPropagateLETD();


        bool ascii() const { return this->_ascii; }


        std::ios::openmode writeMode() const;


    };


    class FilterContext {
    protected:
        std::set<int>       _pid;   // target id lists (particle id)
        std::set<uint32_t>  _bid;   // target id lists (implicit buffer id, not pid)
        std::set<int>       _za;    // target genion ZA lists (empty -> all)
        std::string         _part;  // part literal string

        uint32_t _za_mask[Hadron::Projectile::ZA_SCORING_MASK_SIZE];  // ZA mask


        void _switchTargetPID(const EquationElem& part, bool state);


        void _summaryFilterContext() const;


        void _writeTallyFilter(mcutil::FortranOfstream& stream) const;


        void _writeTallyFilter(std::ofstream& stream) const;


        FilterContext(mcutil::ArgInput& args);


    public:


        static void _initializeStaticArgs(mcutil::ArgumentCard& card);


        void clearZAFilter();


        bool switchTargetZA(int za, uint32_t pos);


        const std::set<int>& pid() const { return this->_pid; }
        const std::set<uint32_t>& bid() const { return this->_bid; }


        bool isActivated(int pid) const;


        bool ionAll() const;


        bool ionIsActivated(uint32_t ion_id) const;


        std::vector<int> ionFilterList() const;


    };


    class MeshContext : public mcutil::Affine {
    protected:
        uint3 _shape;  // tally shape

        MESH_MEMCPY_POLICY    _memcpy_policy;     // memcpy method (host or device)
        MESH_MEMORY_STRUCTURE _memory_structure;

        // dense
        
        size_t _aligned_length;  // length of aligned 1D vector
        float* _data_aligned;    // for kernel memcpy
        float* _unc_aligned;     // for kernel memcpy

        // sparse

        size_t _block;
        size_t _thread;

#ifndef RT2QMD_STANDALONE
        std::shared_ptr<mcutil::DenseToCOOSparseHandler> _coo_sparse_handler;
#endif

        size_t _nnz;           // number of element
        double _tolerance;     // zero tolerance (using when sparse matrix conversion)
        int*   _sparse_index;  // sparse index vector 
        float* _sparse_data;   // sparse data vector
        float* _sparse_unc;    // sparse unc vector


        MeshContext(mcutil::ArgInput& args);


        void _writeMeshGeometryInfo(mcutil::FortranOfstream& stream) const;


        void _writeMeshGeometryInfo(std::ofstream& stream) const;


        void _write3DAlignedData(
            std::ofstream& stream,
            const std::vector<double>& data,  // 4D data
            size_t offset = 0u,
            size_t stride = 1u
        ) const;


        void _writeDenseDataFromDeviceMemory(mcutil::FortranOfstream& stream, size_t max_length) const;


        void _writeCOOSparseDataFromDeviceMemory(mcutil::FortranOfstream& stream, size_t /*max_length*/) const;


        void _writeTallyDataFromDeviceMemory(mcutil::FortranOfstream& stream, size_t max_length = 0) const;


        void _summaryMeshContext() const;


        size_t _prepareAlignedMemory();


        size_t _prepareSparseMemory();


    public:


        ~MeshContext();


        static void _initializeStaticArgs(mcutil::ArgumentCard& card);


        void setKernelDimension(size_t block, size_t thread);


        double3 origin() const;
        double3 size()   const;
        uint3   shape()  const;


        double meshValueTolerance() const { return this->_tolerance; }


        MESH_MEMCPY_POLICY    memcpyPolicy()    const { return this->_memcpy_policy;    }
        MESH_MEMORY_STRUCTURE memoryStructure() const { return this->_memory_structure; }


        float* alignedDataPtr() const { return this->_data_aligned; }
        float* alignedUncPtr()  const { return this->_unc_aligned;  }


        /**
        * @brief Generate sparse matrix (data) from aligned data, coo type
        */
        void generateSparseCOO();


    };


    class FluenceContext {
    protected:
        ENERGY_TYPE        _etype;   // tally energy bin type
        double2            _erange;  // energy range
        uint32_t           _nbin;    // number of energy bin
        std::vector<float> _ebin;    // numeric energy bin (case of neutron)


        FluenceContext(mcutil::ArgInput& args);


        void _writeTallyEnergyStructure(mcutil::FortranOfstream& stream) const;


        void _writeTallyEnergyStructure(std::ofstream& stream) const;


        void _write1DAlignedData(
            std::ofstream& stream,
            const std::vector<double>& data,
            const std::vector<double>& unc
        ) const;


        void _summaryFluenceContext() const;


    public:


        static void _initializeStaticArgs(mcutil::ArgumentCard& card);


        ENERGY_TYPE interpType() const;


        double emin() const;


        double emax() const;


        uint32_t nbin() const;


        double2 getInterpCoeff() const;


    };


    class BoundaryCrossingContext {
    protected:
        geo::RegionContext _from;  // boundary before
        geo::RegionContext _to;    // boundary after
        double _area;  // boundary area


        BoundaryCrossingContext(mcutil::ArgInput& args);


        virtual void _summaryBoundaryCrossingContext() const;


    public:


        static void _initializeStaticArgs(mcutil::ArgumentCard& card);


        geo::RegionContext& from() { return this->_from; }
        geo::RegionContext& to()   { return this->_to; }


        double area() const;


        void setArea(double area);



    };


    class VolumetricContext {
    protected:
        geo::RegionContext _where;   // tally target volume
        double _volume;  // tally volume [cm3]


        VolumetricContext(mcutil::ArgInput& args);


        virtual void _summaryVolumetricContext() const;


    public:


        static void _initializeStaticArgs(mcutil::ArgumentCard& card);


        geo::RegionContext& region();


        double volume() const;


        void setVolume(double volume);


    };


    class DensityContext {
    protected:
        DENSITY_TYPE  _dtype;   // tally scoring unit
        

        DensityContext(mcutil::ArgInput& args);


        void _summaryDensityContext() const;


    public:


        static void _initializeStaticArgs(mcutil::ArgumentCard& card);


        const DENSITY_TYPE& dtype() const { return this->_dtype; }


    };


    size_t tallyMemoryInitialize(double2** data_dev, size_t length);


    void tallyMemoryReset(double2** data_dev, size_t length);


    double stdev(double ex, double ex2);  // E[X] & E[X^2]


}