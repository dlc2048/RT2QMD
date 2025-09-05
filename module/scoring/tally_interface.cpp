//
// Copyright (C) 2025 CM Lee, SJ Ye, Seoul Sational University
//
// Licensed to the Apache Software Foundation(ASF) under one
// or more contributor license agreements.See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// 	"License"); you may not use this file except in compliance
// 	with the License.You may obtain a copy of the License at
// 
// 	http ://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.See the License for the
// specific language governing permissionsand limitations
// under the License.

/**
 * @file    module/scoring/tally_interface.cpp
 * @brief   Tally class interface
 * @author  CM Lee
 * @date    02/18/2025
 */


#ifdef RT2QMD_STANDALONE
#include "device/exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "tally_interface.hpp"
#include "tally_interface.cuh"

#include "material/auxiliary.hpp"


namespace tally {


    EquationElem::EquationElem() :
        type(OPERATOR_TYPE::NONE), pid(Define::PID::PID_UNKNOWN), za(-1) {}


    EquationElem::EquationElem(OPERATOR_TYPE op_type, const std::string& pname_literal)
        : type(op_type), pid(Define::PID::PID_UNKNOWN), za(-1) {
        int pid, za;

        // check part alias
        pid = tallyPartAlias(pname_literal);
        if (pid != Define::PID::PID_UNKNOWN) {
            this->pid = pid;
            return;
        }

        // check ZA alias
        za = mat::findIsotope(pname_literal);
        if (za >= 0) {
            this->za = za;
            return;
        }

        std::stringstream ss;
        ss << "Encountered invalid part '" << pname_literal << "'";
        mclog::fatal(ss);
    }


    EquationElem::EquationElem(OPERATOR_TYPE op_type, int pid)
        : type(op_type), za(-1) {
        // check pid
        if (op_type == OPERATOR_TYPE::OPERAND && mcutil::getPidHash().find(pid) == mcutil::getPidHash().end()) {
            std::stringstream ss;
            ss << "Encountered invalid particle id '" << pid << "'";
            mclog::fatal(ss);
        }
        this->pid = pid;
    }


    EquationElem::EquationElem(OPERATOR_TYPE op_type)
        : EquationElem(op_type, Define::PID::PID_UNKNOWN) {}


    EquationElem::EquationElem(const std::string& pname_literal) :
        EquationElem(OPERATOR_TYPE::OPERAND, pname_literal) {}


    int tallyPartAlias(const std::string& part_literal) {

        // lowercase
        std::string part = part_literal;
        std::transform(part.begin(), part.end(), part.begin(), ::tolower);

        // basic particle alias
        static const std::map<std::string, int> PART_ALIAS_BASIC = {
            { "all"            , Define::PID::PID_ALL      },  // all part
            { "em"             , Define::PID::PID_EM       },  // electromagnetic
            { "electromagnetic", Define::PID::PID_EM       },  // electromagnetic
            { "hadron"         , Define::PID::PID_HADRON   },  // hadron
            { "electron"       , Define::PID::PID_ELECTRON },
            { "e"              , Define::PID::PID_ELECTRON },
            { "photon"         , Define::PID::PID_PHOTON   },
            { "gamma"          , Define::PID::PID_PHOTON   },
            { "g"              , Define::PID::PID_PHOTON   },
            { "positron"       , Define::PID::PID_POSITRON },
            { "neutron"        , Define::PID::PID_NEUTRON  },
            { "n"              , Define::PID::PID_NEUTRON  },
            { "ion"            , Define::PID::PID_GENION   },
            { "genion"         , Define::PID::PID_GENION   },
            { "genericion"     , Define::PID::PID_GENION   },
        };

        // test basic alias
        auto iter = PART_ALIAS_BASIC.find(part);
        if (iter != PART_ALIAS_BASIC.end())
            return iter->second;

        return Define::PID::PID_UNKNOWN;
    }


    bool toLogicalEquation(std::list<EquationElem>& equation) {
        std::deque<EquationElem> temp_stack;
        std::list<EquationElem>  eq_post;
        for (const auto& e : equation) {
            if (e.type == OPERATOR_TYPE::NONE)
                continue;
            else if (e.type == OPERATOR_TYPE::OPERAND) {
                while (!temp_stack.empty()) {
                    eq_post.push_back(temp_stack.back());
                    temp_stack.pop_back();
                }
                eq_post.push_back(e);
            }
            else if (temp_stack.empty())
                temp_stack.push_back(e);
            else {
                return false;
            }
        }
        if (!temp_stack.empty())
            return false;
        equation.swap(eq_post);
        return true;
    }

    
    TallyContext::TallyContext(mcutil::ArgInput& args) : _normalizer(1.0) {
        this->_name  = args["name"].cast<std::string>()[0];
        this->_ascii = args["ascii"].cast<bool>()[0];
        this->_save_policy = args["overwrite"].cast<bool>()[0]
            ? mcutil::SAVE_POLICY::SAVE_NEW
            : mcutil::SAVE_POLICY::SAVE_APPEND;
    }


    void TallyContext::_initializeStaticArgs(mcutil::ArgumentCard& card) {
        card.insert<std::string>("name", 1);
        card.insert<bool>("ascii",     std::vector<bool>{ false });
        card.insert<bool>("overwrite", std::vector<bool>{ true  });
    }


    std::vector<double2> TallyContext::_getDataFromDevice(
        double2* device_data,
        size_t   length
    ) {
        // copy device data & r2 memory to host
        size_t n = 10000;
        std::vector<double2> host_data;
        host_data.resize(length);
        for (size_t i = 0; i < length; i += n) {
            size_t stride = std::min(i + n, length) - i;
            CUDA_CHECK(cudaMemcpy(&host_data[i], device_data + i,
                sizeof(double2) * stride, cudaMemcpyDeviceToHost));
        }
        return host_data;
    }


    void TallyContext::_writeTallyHeader(mcutil::FortranOfstream& stream) const {
        int ssize;
        ssize = (int)this->_name.size();
        stream.write(reinterpret_cast<const unsigned char*>(
            this->_name.c_str()), ssize);
        ssize = (int)this->_unit.size();
        stream.write(reinterpret_cast<const unsigned char*>(
            this->_unit.c_str()), ssize);
    }


    void TallyContext::_writeTallyHeader(std::ofstream& stream) const {
        mclog::FormattedTable fmt({ 16, 16 });
        fmt << "Name" << this->_name;
        stream << fmt.str() << std::endl;
        fmt.clear();
        fmt << "Unit" << this->_unit;
        stream << fmt.str() << std::endl;
    }


    void TallyContext::_writeTallyData(mcutil::FortranOfstream& stream, size_t max_length) const {
        std::vector<float> data_float
            = mcutil::cvtVectorDoubleToFloat(this->_data);
        std::vector<float> unc_float
            = mcutil::cvtVectorDoubleToFloat(this->_unc);
        size_t size_data = max_length ? std::min(max_length, data_float.size()) : data_float.size();
        size_t size_unc  = max_length ? std::min(max_length, unc_float.size())  : unc_float.size();
        stream.write(reinterpret_cast<const unsigned char*>
            (&data_float[0]), sizeof(float) * size_data);
        stream.write(reinterpret_cast<const unsigned char*>
            (&unc_float[0]),  sizeof(float) * size_unc);
    }


    void TallyContext::_summaryTallyContext() const {
        mclog::printName(this->name());
        mclog::printVar("Unit", this->_unit);
    }


    void TallyContext::normalize(double total_weight) {
        for (size_t i = 0; i < this->_data.size(); ++i) {
            double& d  = this->_data[i];
            double& d2 = this->_unc[i];
            // per weight
            d  /= total_weight;
            d2 /= total_weight;
            // now d2 is the standard deviation of variable
            d2  = stdev(d, d2);
            // d2 is the standard deviation of estimated mean
            d2 /= sqrt(total_weight);
            // per normalizer
            d  /= this->_normalizer;
        }
    }


    void TallyContext::setNormalizer(double normalizer) {
        this->_normalizer = normalizer;
    }


    void TallyContext::errPropagateLETD() {
        size_t size_data = this->_data.size();
        size_t size_unc  = this->_unc.size();

        assert(size_data == size_unc);
        assert(size_data % 2 == 0);

        size_data /= 2;

        for (size_t i = 0; i < size_data; ++i) {
            this->_data[i] = this->_data[i + size_data] / this->_data[i];
            this->_unc[i]  = std::sqrt(this->_unc[i] * this->_unc[i] + this->_unc[i + size_data] * this->_unc[i + size_data]);
        }
    }


    std::ios::openmode TallyContext::writeMode() const {
        return this->_save_policy == mcutil::SAVE_POLICY::SAVE_NEW
            ? std::ios::out
            : std::ios::app;
    }


    std::vector<int> FilterContext::_PROJECTILE_LIST;


    void FilterContext::_switchTargetPID(const EquationElem& part, bool state) {
        for (int pid = Define::PID::PID_ELECTRON; pid < Define::PID::PID_VACANCY; pid = pid << 1) {
            if (part.pid & pid) {  // included
                if (state) {  // turn on
                    if (this->_pid & pid) {
                        std::stringstream ss;
                        ss << "Cannot include part '" << mcutil::getPidName().find(pid)->second << "'. It already exist";
                        mclog::fatal(ss);
                    }
                    else
                        this->_pid |= pid;
                }
                else {
                    if (this->_pid & pid)
                        this->_pid &= ~pid;
                    else {
                        std::stringstream ss;
                        ss << "Cannot exclude part '" << mcutil::getPidName().find(pid)->second << "'. It is already excluded";
                        mclog::fatal(ss);
                    }
                }
            }
        }
    }


    void FilterContext::_switchTargetZA(const EquationElem& part, bool state) {
        int2 za_split = physics::splitZA(part.za);
        if (za_split.y == 0) {  // ZZZ000
            size_t counter = 0;
            for (size_t i = 0; i < FilterContext::_PROJECTILE_LIST.size(); ++i) {
                int2 za_p_split = physics::splitZA(FilterContext::_PROJECTILE_LIST[i]);
                if (za_p_split.x == za_split.x) {
                    this->_switchTargetZA(i, FilterContext::_PROJECTILE_LIST[i], state);
                    counter += 1;
                }
            }
            if (!counter) {
                std::stringstream ss;
                ss << "Undefined projectile '" << part.za << "'";
                mclog::fatal(ss);
            }
        }
        else {  // ZZZAAA
            ptrdiff_t pos = std::distance(
                FilterContext::_PROJECTILE_LIST.begin(),
                std::find(
                    FilterContext::_PROJECTILE_LIST.begin(),
                    FilterContext::_PROJECTILE_LIST.end(),
                    part.za
                )
            );
            this->_switchTargetZA(pos, part.za, state);
        }
    }


    void FilterContext::_switchTargetZA(long long pos, int za, bool state) {

        if (pos == FilterContext::_PROJECTILE_LIST.size()) {  // turn activation
            if (this->_za_activation) {
                mclog::fatal("Attemting to set multiple ion filter for activation");
            }
            this->_za_activation = za;
            return;
        }

        uint32_t index  = pos / Hadron::Projectile::ZA_SCORING_MASK_STRIDE;
        uint32_t offset = pos % Hadron::Projectile::ZA_SCORING_MASK_STRIDE;

        assert(pos < Hadron::Projectile::ZA_SCORING_MASK_DIM);

        if (this->_za_mask[index] & (0x1u << offset)) {
            if (state) {  // 
                std::stringstream ss;
                ss << "Ion filter for projectile '" << za << "' already exist";
                mclog::fatal(ss);
            }
            else
                this->_za_mask[index] &= ~(0x1u << offset);
        }
        else {
            if (state) {
                this->_za_mask[index] |= 0x1u << offset;
            }
            else {
                std::stringstream ss;
                ss << "Ion filter for projectile '" << za << "' is already excluded";
                mclog::fatal(ss);
            }
        }
    }


    void FilterContext::_summaryFilterContext() const {
        mclog::printVar("Part", this->_part);
        if (this->_pid & Define::PID::PID_HADRON) {  // HADRON activated
            std::string filter;
            if (this->_za_activation) {
                filter = "On";
                mclog::printVar("Ion Filter", filter);
                std::stringstream ss;
                ss << "    ";
                ss << this->_za_activation;
                mclog::print(ss);
            }
            else {
                filter = this->ionAll() ? "Off" : "On";
                mclog::printVar("Ion Filter", filter);
                if (!this->ionAll()) {
                    std::stringstream ss;
                    ss << "    ";
                    for (auto za : this->ionFilterList())
                        ss << "  " << za;
                    mclog::print(ss);
                }
            }
        }
    }


    void FilterContext::_writeTallyFilter(mcutil::FortranOfstream& stream) const {
        int ssize;
        ssize = (int)this->_part.size();
        stream.write(reinterpret_cast<const unsigned char*>(
            this->_part.c_str()), ssize);
    }


    void FilterContext::_writeTallyFilter(std::ofstream& stream) const {
        mclog::FormattedTable fmt({ 16, 16 });
        fmt << "Part" << this->_part;
        stream << fmt.str() << std::endl;
        stream << std::endl;
    }


    void FilterContext::_syncZAMaskAndActivationZA() {
        size_t n_mask = 0u;
        int    last_c = 0;
        for (int i = 0; i < Hadron::Projectile::ZA_SCORING_MASK_SIZE; ++i) {
            for (int j = 0; j < Hadron::Projectile::ZA_SCORING_MASK_STRIDE; ++j) {
                if (this->_za_mask[i] & (0x1u << j)) {
                    n_mask++;
                    last_c = i * Hadron::Projectile::ZA_SCORING_MASK_STRIDE + j;
                }
            }
        }
        // case 1, n_mask == 0
        if (n_mask == 0) {
            if (this->_za_activation == 0)
                mclog::fatal("Ion filter is not set for activation tally");
        }
        else if (n_mask == 1) {
            if (this->_za_activation > 0)
                mclog::fatal("Attemting to set multiple ion filter for activation tally");
            this->_za_activation = FilterContext::_PROJECTILE_LIST[last_c];
        }
        else {
            mclog::fatal("Attemting to set multiple ion filter for activation");
        }
    }


    FilterContext::FilterContext(mcutil::ArgInput& args) : 
        _pid(Define::PID::PID_UNKNOWN), _za_mask{ 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u }, _za_activation(0) {
        
        std::vector<std::string> equation_literal = args["part"].cast<std::string>();
        if (equation_literal.size() < 1)
            mclog::fatalValueSize("part", 1, equation_literal.size());

        // join and split
        std::deque<std::string> equation_splited;
        for (const auto& equation_seg : equation_literal) {
            std::deque<std::string> equation_seg_splited = mcutil::split(equation_seg, "+-");
            equation_splited.insert(
                equation_splited.end(), 
                equation_seg_splited.begin(), 
                equation_seg_splited.end()
            );
        }

        // read infix equation
        std::list<EquationElem> equation;
        for (size_t i = 0; i < equation_splited.size(); ++i) {
            const std::string& part = equation_splited[i];
            auto iter = OPERATOR_LITERAL.find(part);
            if (iter == OPERATOR_LITERAL.end()) { // element is operand
                equation.push_back(EquationElem(part));
            }
            else  // element is operator
                equation.push_back(EquationElem(iter->second));
        }

        // make logical equation
        if (!toLogicalEquation(equation))
            mclog::fatal("Part equation has problem");

        // analyse equation
        bool mode = true;
        for (const auto& seg : equation) {
            if (seg.type == OPERATOR_TYPE::NOT) {
                mode = false;
                continue;
            }
            if (seg.pid != Define::PID::PID_UNKNOWN)
                this->_switchTargetPID(seg, mode);
            else
                this->_switchTargetZA(seg, mode);

            mode = true;
        }
    }


    void FilterContext::_initializeStaticArgs(mcutil::ArgumentCard& card) {
        card.insert<std::string>("part");
    }


    void FilterContext::clearZAFilter() {
        for (int i = 0; i < Hadron::Projectile::ZA_SCORING_MASK_SIZE; ++i)
            this->_za_mask[i] = 0x00000000u;
    }


    void FilterContext::setZAFilterAll() {
        for (int i = 0; i < Hadron::Projectile::ZA_SCORING_MASK_SIZE; ++i)
            this->_za_mask[i] = 0xffffffffu;
    }

    
    bool FilterContext::isActivated(int pid) const {
        return this->_pid & pid;
    }


    bool FilterContext::ionAll() const {
        for (auto mask : this->_za_mask) {
            if (mask != 0xffffffffu)
                return false;
        }
        return true;
    }


    bool FilterContext::ionNone() const {
        for (auto mask : this->_za_mask) {
            if (mask != 0x00000000u)
                return false;
        }
        return true;
    }


    bool FilterContext::ionIsActivated(uint32_t ion_id) const {
        uint32_t index  = ion_id / Hadron::Projectile::ZA_SCORING_MASK_STRIDE;
        uint32_t offset = ion_id % Hadron::Projectile::ZA_SCORING_MASK_STRIDE;
        return this->_za_mask[index] & (0x1u << offset);
    }


    std::vector<int> FilterContext::ionFilterList() const {
        std::vector<int> flist;
        for (uint32_t i = 0; i < FilterContext::_PROJECTILE_LIST.size(); ++i) {
            if (this->ionIsActivated(i))
                flist.push_back(FilterContext::_PROJECTILE_LIST[i]);
        }
        return flist;
    }


    void FilterContext::syncZAFilterAndParticleFilter(TALLY_FILTER_TYPE type) {

        if (type == TALLY_FILTER_TYPE::ACTIVATION) {
            this->_syncZAMaskAndActivationZA();
            this->_pid |= Define::PID::PID_HADRON;
        }
        else if (type == TALLY_FILTER_TYPE::FLUENCE) {  // fluence
            if (this->isActivated(Define::PID::PID_GENION) && this->ionNone())
                this->setZAFilterAll();
            else if (!this->ionNone()) {
                this->_pid |= Define::PID::PID_GENION;
            }
            this->_za_activation = 0;
        }
        else if (type == TALLY_FILTER_TYPE::ENERGY) {  // energy

            // ion
            if ((this->isActivated(Define::PID::PID_GENION) || this->isActivated(Define::PID::PID_NEUTRON)) && this->ionNone()) {
                this->setZAFilterAll();
                this->_pid |= Define::PID::PID_HADRON;
            }
            else if (!this->ionNone())
                this->_pid |= Define::PID::PID_HADRON;

            // EM
            if (this->isActivated(Define::PID::PID_PHOTON)   ||
                this->isActivated(Define::PID::PID_ELECTRON) ||
                this->isActivated(Define::PID::PID_POSITRON))
                this->_pid |= Define::PID::PID_EM;

            this->_za_activation = 0;
        }

        // set the literal name
        switch (this->_pid) {
        case Define::PID::PID_EM:
            this->_part = "EM";
            break;
        case Define::PID::PID_NEUTRON:
            this->_part = "neutron";
            break;
        case Define::PID::PID_GENION:
            this->_part = "genericion";
            break;
        case Define::PID::PID_HADRON:
            this->_part = "hadron";
            break;
        case Define::PID::PID_ALL:
            this->_part = "all";
            break;
        default:
            this->_part = "mixed";
        }
    }


    MeshContext::MeshContext(mcutil::ArgInput& args) :
        mcutil::Affine(), 
        _aligned_length(0x0u),
        _data_aligned(nullptr), 
        _unc_aligned(nullptr),
        _nnz(0x0u),
        _sparse_index(nullptr),
        _sparse_data(nullptr),
        _sparse_unc(nullptr) {

        std::vector<double> xrange = args["xrange"].cast<double>();
        std::vector<double> yrange = args["yrange"].cast<double>();
        std::vector<double> zrange = args["zrange"].cast<double>();
        if (xrange[0] >= xrange[1])
            mclog::fatal("'xmax' must be larger than 'xmin'");
        if (yrange[0] >= yrange[1])
            mclog::fatal("'ymax' must be larger than 'ymin'");
        if (zrange[0] >= zrange[1])
            mclog::fatal("'ymax' must be larger than 'ymin'");
        int nx = args["nx"].cast<int>()[0];
        int ny = args["ny"].cast<int>()[0];
        int nz = args["nz"].cast<int>()[0];
        if (nx <= 0)
            mclog::fatal("'nx' must be positive value");
        if (ny <= 0)
            mclog::fatal("'nx' must be positive value");
        if (nz <= 0)
            mclog::fatal("'nz' must be positive value");
        
        double size[3] = {
            (xrange[1] - xrange[0]) / (double)nx,
            (yrange[1] - yrange[0]) / (double)ny,
            (zrange[1] - zrange[0]) / (double)nz
        };

        this->translate(+0.5, +0.5, +0.5);
        this->transform(mcutil::Affine(
            size[0], 0.0, 0.0, xrange[0],
            0.0, size[1], 0.0, yrange[0],
            0.0, 0.0, size[2], zrange[0]
        ));
        this->_shape  = { (uint32_t)nx, (uint32_t)ny, (uint32_t)nz };

        // memcpy policy
        std::string mcmcpy_policy_literal = args["memcpy"].cast<std::string>()[0];
        if (mcmcpy_policy_literal == "auto")
            this->_memcpy_policy = MESH_MEMCPY_POLICY::MEMCPY_AUTO;
        else if (mcmcpy_policy_literal == "host")
            this->_memcpy_policy = MESH_MEMCPY_POLICY::MEMCPY_HOST;
        else if (mcmcpy_policy_literal == "kernel")
            this->_memcpy_policy = MESH_MEMCPY_POLICY::MEMCPY_KERNEL;
        else
            mclog::fatal("'memcpy' must be 'auto', 'host', or 'kernel'");

        if (this->_memcpy_policy == MESH_MEMCPY_POLICY::MEMCPY_AUTO) {
            size_t dim = this->_shape.x * this->_shape.y * this->_shape.z;
            this->_memcpy_policy = dim >= MESH_MEMCPY_AUTO_KERNEL_THRESHOLD 
                ? MESH_MEMCPY_POLICY::MEMCPY_KERNEL
                : MESH_MEMCPY_POLICY::MEMCPY_HOST;
        }

        // sparse
        std::string sparse_mode_literal = args["sparse"].cast<std::string>()[0];
        if (sparse_mode_literal == "dense")
            this->_memory_structure = MESH_MEMORY_STRUCTURE::MEMORY_DENSE;
        else if (sparse_mode_literal == "coo")
            this->_memory_structure = MESH_MEMORY_STRUCTURE::MEMORY_SPARSE_COO;
        else
            mclog::fatal("'sparse' must be 'dense' or 'coo'");

        // force device memcpy policy when sparse data structure is used
        if (this->_memory_structure != MESH_MEMORY_STRUCTURE::MEMORY_DENSE &&
            this->_memcpy_policy    != MESH_MEMCPY_POLICY::MEMCPY_KERNEL) {
            mclog::warning("Kernel memcpy mode is being forced for CUSPARSE API usage");
            this->_memcpy_policy = MESH_MEMCPY_POLICY::MEMCPY_KERNEL;
        }

        // tolerance
        this->_tolerance = args["sparse_tolerance"].cast<double>()[0];
        if (this->_memory_structure == MESH_MEMORY_STRUCTURE::MEMORY_DENSE &&
            this->_memcpy_policy    == MESH_MEMCPY_POLICY::MEMCPY_KERNEL) {
            this->_tolerance = 0.0;  // set tolerance to 0 in a case of kernel & dense
        }

    }


    void MeshContext::_writeMeshGeometryInfo(mcutil::FortranOfstream& stream) const {
        std::vector<double> aff_1d = this->flatten();
        int    shape[3];
        shape[0] = (int)this->_shape.x;
        shape[1] = (int)this->_shape.y;
        shape[2] = (int)this->_shape.z;
        stream.write(reinterpret_cast<const unsigned char*>
            (shape), sizeof(int) * 3);
        stream.write(aff_1d);
        int    structure_type = static_cast<int>(this->_memory_structure);
        stream.write(reinterpret_cast<const unsigned char*>
            (&structure_type), sizeof(int));
    }


    void MeshContext::_writeMeshGeometryInfo(std::ofstream& stream) const {
        mclog::FormattedTable fmt_header({ 24, 8 });
        mclog::FormattedTable fmt(std::vector<size_t>(8, 12));

        std::string axis_str[3] = { "X", "Y", "Z" };
        double      ori[3]      = { this->origin().x, this->origin().y, this->origin().z };
        double      size[3]     = { this->size().x,   this->size().y,   this->size().z   };
        size_t      shape[3]    = { this->_shape.x,   this->_shape.y,   this->_shape.z   };

        for (size_t i = 0; i < 3; ++i) {
            // axis shape
            fmt_header << "Number of " + axis_str[i] + " bin" << shape[i];
            stream << fmt_header.str() << std::endl;
            fmt_header.clear();

            // lim
            fmt_header << axis_str[i] + " bin boundaries :";
            stream << fmt_header.str() << std::endl;
            fmt_header.clear();

            for (size_t j = 0; j <= this->_shape.x; ++j) {
                if ((j % 8 == 0) && j) {
                    stream << fmt.str() << std::endl;
                    fmt.clear();
                }
                double lim = ori[i] + size[i] * j;
                fmt << lim;
            }
            stream << fmt.str() << std::endl;
            stream << std::endl;
            fmt.clear();
        }

    }


    void MeshContext::_write3DAlignedData(
        std::ofstream& stream,
        const std::vector<double>& data,
        size_t offset,
        size_t stride
    ) const {
        mclog::FormattedTable fmt_header({ 24, 8 });
        mclog::FormattedTable fmt(std::vector<size_t>(8, 12));

        size_t count = 0u;
        for (size_t zz = 0; zz < this->_shape.z; ++zz) {

            fmt_header << "Z slice = " << zz;
            stream << fmt_header.str() << std::endl;
            fmt_header.clear();

            fmt_header << "XY (ix * ny + iy) :";
            stream << fmt_header.str() << std::endl;
            fmt_header.clear();

            for (size_t xx = 0; xx < this->_shape.x; ++xx) {
                for (size_t yy = 0; yy < this->_shape.y; ++yy) {
                    if ((count % 8 == 0) && count) {
                        stream << fmt.str() << std::endl;
                        fmt.clear();
                        count = 0u;
                    }
                    size_t idx = offset + stride * (zz + this->_shape.z * (yy + this->_shape.y * xx));
                    fmt << data[idx];
                    count++;
                }
            }
            stream << fmt.str() << std::endl;
            stream << std::endl;
            fmt.clear();
            count = 0u;
        }
    }


    void MeshContext::_writeDenseDataFromDeviceMemory(mcutil::FortranOfstream& stream, size_t max_length) const {
        size_t size;
        size  = (size_t)this->_shape.x;
        size *= (size_t)this->_shape.y;
        size *= (size_t)this->_shape.z;
        std::vector<float> data(size);
        std::vector<float> unc(size);

        CUDA_CHECK(cudaMemcpy(&data[0], this->_data_aligned,
            sizeof(float) * size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&unc[0], this->_unc_aligned,
            sizeof(float) * size, cudaMemcpyDeviceToHost));

        size_t size_data = max_length ? std::min(max_length, data.size()) : data.size();
        size_t size_unc  = max_length ? std::min(max_length, unc.size())  : unc.size();
        stream.write(reinterpret_cast<const unsigned char*>
            (&data[0]), sizeof(float) * size_data);
        stream.write(reinterpret_cast<const unsigned char*>
            (&unc[0]), sizeof(float) * size_unc);
    }


    void MeshContext::_writeCOOSparseDataFromDeviceMemory(mcutil::FortranOfstream& stream, size_t /*max_length*/) const {

        std::vector<int>   index(this->_nnz);
        std::vector<float> data(this->_nnz);
        std::vector<float> unc(this->_nnz);

        if (this->_nnz) {
            CUDA_CHECK(cudaMemcpy(&index[0], this->_sparse_index,
                sizeof(int) * this->_nnz, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&data[0], this->_sparse_data,
                sizeof(float) * this->_nnz, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&unc[0], this->_sparse_unc,
                sizeof(float) * this->_nnz, cudaMemcpyDeviceToHost));
        }
        
        stream.write(index);
        stream.write(data);
        stream.write(unc);
    }


    void MeshContext::_writeTallyDataFromDeviceMemory(mcutil::FortranOfstream& stream, size_t max_length) const {
        switch (this->_memory_structure) {
        case MESH_MEMORY_STRUCTURE::MEMORY_DENSE:
            this->_writeDenseDataFromDeviceMemory(stream, max_length);
            break;
        case MESH_MEMORY_STRUCTURE::MEMORY_SPARSE_COO:
            this->_writeCOOSparseDataFromDeviceMemory(stream, max_length);
            break;
        default:
            assert(false);
        }
        return;
    }


    void MeshContext::_summaryMeshContext() const {
        mclog::FormattedTable fmt({ 12, 12, 12 });
        double3 origin = this->origin();
        double3 size   = this->size();

        fmt << origin.x << origin.y << origin.z;
        mclog::printVar("Origin", fmt.str(), "cm");
        fmt.clear();

        fmt << size.x << size.y << size.z;
        mclog::printVar("Size", fmt.str(), "cm");
        fmt.clear();

        fmt << this->_shape.x << this->_shape.y << this->_shape.z;
        mclog::printVar("Shape", fmt.str());
        fmt.clear();

        double volume = size.x * size.y * size.z;
        volume *= (double)this->_shape.x;
        volume *= (double)this->_shape.y;
        volume *= (double)this->_shape.z;
        mclog::printVar("Total volume", volume, "cm3");
    }


    void MeshContext::_initializeStaticArgs(mcutil::ArgumentCard& card) {
        card.insert<double>("xrange", 2);
        card.insert<double>("yrange", 2);
        card.insert<double>("zrange", 2);
        card.insert<size_t>("nx", { 1 });
        card.insert<size_t>("ny", { 1 });
        card.insert<size_t>("nz", { 1 });
        card.insert<std::string>("memcpy", { "auto"  });
        card.insert<std::string>("sparse", { "dense" });
        card.insert<double>("sparse_tolerance", { 1e-4 }, { 0.0 }, { 1.0 });
    }


    void MeshContext::setKernelDimension(size_t block, size_t thread) {
        this->_block  = block;
        this->_thread = thread;
    }


    double3 MeshContext::origin() const {
        double3 origin_arr = { -0.5, -0.5, -0.5 };
        return mcutil::transform(origin_arr, *this);
    }


    double3 MeshContext::size() const {
        return this->scale();
    }


    uint3 MeshContext::shape() const {
        return this->_shape;
    }


#ifndef RT2QMD_STANDALONE

    void MeshContext::generateSparseCOO() {
        // convert dense to sparse by using CUSPARSE 
        this->_nnz = this->_coo_sparse_handler->convert();
        // build uncertainty sparse vector from the coo index vector
        __host__buildUncertaintyCOOSparse(
            this->_unc_aligned,
            this->_sparse_index,
            this->_sparse_unc,
            (int)this->_nnz,
            (int)this->_block,
            (int)this->_thread
        );
        return;
    }


    size_t MeshContext::_prepareAlignedMemory() {
        // aligned data
        if (this->_memcpy_policy == MESH_MEMCPY_POLICY::MEMCPY_KERNEL) {
            size_t dim = this->_aligned_length;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_data_aligned),
                sizeof(float) * dim));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_unc_aligned),
                sizeof(float) * dim));
            return 2 * dim * sizeof(float);
        }
        return 0u;
    }


    size_t MeshContext::_prepareSparseMemory() {
        // sparse data
        if (this->_memory_structure != MESH_MEMORY_STRUCTURE::MEMORY_DENSE) {
            size_t dim_max = this->_aligned_length;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_sparse_index),
                sizeof(int) * dim_max));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_sparse_data),
                sizeof(float) * dim_max));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_sparse_unc),
                sizeof(float) * dim_max));

            // sparse coo handler
            this->_coo_sparse_handler = std::make_shared<mcutil::DenseToCOOSparseHandler>(
                dim_max, this->_data_aligned, this->_sparse_index, this->_sparse_data
            );
            return 2 * dim_max * sizeof(float) + dim_max * sizeof(int);
        }
        return 0u;
    }


#endif


    MeshContext::~MeshContext() {
        mcutil::DeviceVectorHelper(this->_data_aligned).free();
        mcutil::DeviceVectorHelper(this->_unc_aligned).free();
        mcutil::DeviceVectorHelper(this->_sparse_index).free();
        mcutil::DeviceVectorHelper(this->_sparse_data).free();
        mcutil::DeviceVectorHelper(this->_sparse_unc).free();
    }


    FluenceContext::FluenceContext(mcutil::ArgInput& args) {
        // check energy bin interpolation type
        std::string type_str = args["escale"].cast<std::string>()[0];
        this->_etype = ENERGY_TYPE::ENERGY_LINEAR;
        if (type_str == "log")
            this->_etype = ENERGY_TYPE::ENERGY_LOG;
        else if (type_str == "linear");
        else
            mclog::fatal("'escale' must be 'linear' or 'log'");

        // interpret energy structure
        std::vector<double> erange = args["erange"].cast<double>();
        int ne = args["ne"].cast<int>()[0];
        if (erange[0] >= erange[1])
            mclog::fatal("'erange[1]' must be larger than 'erange[0]'");
        if (this->_etype == ENERGY_TYPE::ENERGY_LOG) {
            if (erange[0] <= 0)
                mclog::fatal("'erange[0]' must be positive value");
        }
        else {
            if (erange[0] < 0)
                mclog::fatal("'erange[0]' cannot be a negative value");
        }
        if (ne <= 0)
            mclog::fatal("'ne' must be a positive value");
        this->_erange = { erange[0], erange[1] };
        this->_nbin   = (uint32_t)ne;
    }


    void FluenceContext::_writeTallyEnergyStructure(mcutil::FortranOfstream& stream) const {
        int    etype = static_cast<int>(this->_etype);
        double erange[2];
        int    nbin = (int)this->_nbin;
        erange[0] = this->_erange.x;
        erange[1] = this->_erange.y;
        stream.write(reinterpret_cast<const unsigned char*>
            (&etype), sizeof(int));
        stream.write(reinterpret_cast<const unsigned char*>
            (erange), sizeof(double) * 2);
        stream.write(reinterpret_cast<const unsigned char*>
            (&nbin), sizeof(int));
    }


    void FluenceContext::_writeTallyEnergyStructure(std::ofstream& stream) const {
        mclog::FormattedTable fmt_header({ 24, 8 });
        mclog::FormattedTable fmt(std::vector<size_t>(8, 12));

        // scale
        fmt_header << "Energy scale" << (this->_etype == ENERGY_TYPE::ENERGY_LINEAR ? "linear" : "log");
        stream << fmt_header.str() << std::endl;
        fmt_header.clear();

        // shape
        fmt_header << "Number of energy bin" << this->_nbin;
        stream << fmt_header.str() << std::endl;
        fmt_header.clear();

        // lim
        fmt_header << "Energy bin boundaries :";
        stream << fmt_header.str() << std::endl;
        fmt_header.clear();

        for (size_t j = 0; j <= this->_nbin; ++j) {
            if ((j % 8 == 0) && j) {
                stream << fmt.str() << std::endl;
                fmt.clear();
            }
            double lim;
            if (this->_etype == ENERGY_TYPE::ENERGY_LINEAR)
                lim = (this->_erange.x * (this->_nbin - j) + this->_erange.y * j) / (double)this->_nbin;
            else {
                lim = (std::log(this->_erange.x) * (this->_nbin - j) + std::log(this->_erange.y) * j) / (double)this->_nbin;
                lim = std::exp(lim);
            }
            fmt << lim;
        }
        stream << fmt.str() << std::endl;
        stream << std::endl;
        fmt.clear();
    }


    void FluenceContext::_write1DAlignedData(
        std::ofstream& stream,
        const std::vector<double>& data,
        const std::vector<double>& unc
    ) const {
        mclog::FormattedTable fmt({ 14, 14, 14, 14 });
        fmt << "Emin" << "Emax" << "Value" << "Uncertainty";
        stream << fmt.str() << std::endl;
        fmt.clear();

        double lim_low = this->_erange.x;
        double lim_high;
        for (size_t j = 1; j <= this->_nbin; ++j) {
            if (this->_etype == ENERGY_TYPE::ENERGY_LINEAR)
                lim_high = (this->_erange.x * (this->_nbin - j) + this->_erange.y * j) / (double)this->_nbin;
            else {
                lim_high = (std::log(this->_erange.x) * (this->_nbin - j) + std::log(this->_erange.y) * j) / (double)this->_nbin;
                lim_high = std::exp(lim_high);
            }
            fmt << lim_low << lim_high << data[j - 1] << unc[j - 1];
            stream << fmt.str() << std::endl;
            fmt.clear();

            lim_low = lim_high;
        }
    }


    void FluenceContext::_summaryFluenceContext() const {
        mclog::printVar("Energy from", this->_erange.x, "MeV");
        mclog::printVar("Energy to",   this->_erange.y, "MeV");
        std::string method = this->_etype == tally::ENERGY_TYPE::ENERGY_LINEAR
            ? "linear spacing" : "log spacing";
        mclog::printVar("Binning method", method);
        mclog::printVar("Number of bin", (size_t)this->_nbin);
    }


    void FluenceContext::_initializeStaticArgs(mcutil::ArgumentCard& card) {
        card.insert<std::string>("escale", 1);
        card.insert<double>("erange", 2);
        card.insert<int>("ne", (size_t)1);
    }


    ENERGY_TYPE FluenceContext::interpType() const {
        return this->_etype;
    }


    double FluenceContext::emin() const {
        return this->_erange.x;
    }


    double FluenceContext::emax() const {
        return this->_erange.y;
    }


    uint32_t FluenceContext::nbin() const {
        return this->_nbin;
    }


    double2 FluenceContext::getInterpCoeff() const {
        double2 eihd;
        eihd.y = (double)this->_nbin;
        if (this->_etype == ENERGY_TYPE::ENERGY_LINEAR) {
            eihd.y /= (this->_erange.y - this->_erange.x);
            eihd.x = -eihd.y * this->_erange.x;
        }
        else {
            eihd.y /= log(this->_erange.y / this->_erange.x);
            eihd.x = -eihd.y * log(this->_erange.x);
        }
        return eihd;
    }


    void BoundaryCrossingContext::_summaryBoundaryCrossingContext() const {
        mclog::printVar("Total area",  this->_area, "cm2");
        mclog::printVar("Region from", this->_from.name(), "");
        mclog::printVar("Region to",   this->_to.name(), "");
    }


    BoundaryCrossingContext::BoundaryCrossingContext(mcutil::ArgInput& args)
        : _from(args["from"].cast<std::string>()[0]),
          _to(args["to"].cast<std::string>()[0]) {
        this->_area = args["area"].cast<double>()[0];
    }


    void BoundaryCrossingContext::_initializeStaticArgs(mcutil::ArgumentCard& card) {
        card.insert<std::string>("from", 1);
        card.insert<std::string>("to",   1);
        card.insert<double>("area", { 0.e0 }, { 0.e0 }, { 1.e30 });
    }


    double BoundaryCrossingContext::area() const {
        return this->_area;
    }


    void BoundaryCrossingContext::setArea(double area) {
        if (area <= 0.e0) {
            std::stringstream ss;
            ss << "Fail to initialize 'BoundaryCrossingContext'. "
               << "Its net crossing boundary area is 0. "
               << "Region '" << this->from().name() 
               << "' may not adjecent to region '" << this->to().name() << "'";
            mclog::fatal(ss);
        }
        this->_area = area;
    }


    VolumetricContext::VolumetricContext(mcutil::ArgInput& args) : 
        _where(args["where"].cast<std::string>()[0]) {
        this->_volume = args["volume"].cast<double>()[0];
    }


    void VolumetricContext::_summaryVolumetricContext() const {
        mclog::printVar("Total volume", this->_volume, "cm3");
        mclog::printVar("Region for",   this->_where.name(), "");
    }


    void VolumetricContext::_initializeStaticArgs(mcutil::ArgumentCard& card) {
        card.insert<std::string>("where", 1);
        card.insert<double>("volume", { 0.e0 }, { 0.e0 }, { 1.e30 });
    }


    geo::RegionContext& VolumetricContext::region() {
        return this->_where;
    }


    double VolumetricContext::volume() const {
        return this->_volume;
    }


    void VolumetricContext::setVolume(double volume) {
        if (volume <= 0.e0) {
            std::stringstream ss;
            ss << "Fail to initialize 'VolumetricContext'. "
                << "Its net volume is negative. "
                << "Region '" << this->region().name()
                << "' not be closed (may void)";
            mclog::fatal(ss);
        }
        this->_volume = volume;
    }


    
    DensityContext::DensityContext(mcutil::ArgInput& args) {
        std::string type_str = args["type"].cast<std::string>()[0];
        if (type_str == "depo")
            this->_dtype = DENSITY_TYPE::DENSITY_DEPO;
        else if (type_str == "dose")
            this->_dtype = DENSITY_TYPE::DENSITY_DOSE;
        else if (type_str == "rbe_dose")
            this->_dtype = DENSITY_TYPE::DENSITY_RBEDOSE;
        else if (type_str == "letd")
            this->_dtype = DENSITY_TYPE::DENSITY_LETD;
        else if (type_str == "activation")
            this->_dtype = DENSITY_TYPE::DENSITY_ACTIVATION;
        
        else
            mclog::fatal("'type' must be 'depo', 'dose', 'rbe_dose', 'letd', or 'activation'");
    }


    void DensityContext::_summaryDensityContext() const {
    }


    void DensityContext::_initializeStaticArgs(mcutil::ArgumentCard& card) {
        card.insert<std::string>("type", 1);
    }

    
    size_t tallyMemoryInitialize(double2** data_dev, size_t length) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(data_dev),
            sizeof(double2) * length));
        tallyMemoryReset(data_dev, length);
        return sizeof(double2) * length;
    }


    void tallyMemoryReset(double2** data_dev, size_t length) {
        // initialize T & T2 memory to zero
        size_t n = 10000;
        std::vector<double2> zeroing(n, { 0.e0, 0.e0 });
        for (size_t i = 0; i < length; i += n) {
            size_t stride = std::min(i + n, length) - i;
            CUDA_CHECK(cudaMemcpy(*data_dev + i, &zeroing[0],
                sizeof(double2) * stride, cudaMemcpyHostToDevice));
        }
    }


    double stdev(double ex, double ex2) {
        if (ex == 0.e0)  // 0 value
            return 0.e0;
        else
            return sqrt(std::max(ex2 - ex * ex, 0.0)) / ex;
    }


}