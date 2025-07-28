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
 * @file    mcutil/device/algorithm.cpp
 * @brief   Host side interpolation, sampling and integral algorithms
 * @author  CM Lee
 * @date    05/23/2023
 */

#ifdef RT2QMD_STANDALONE
#include "exception.h"
#else
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#endif

#include "algorithm.hpp"


namespace mcutil {


    LogLogCoeff::LogLogCoeff(double xmin, double xmax, size_t nbin) {
        this->_llx.y = (double)nbin / std::log(xmax / xmin);
        this->_llx.x = -this->_llx.y * std::log(xmin);
    }


    AliasTable::AliasTable(size_t size, const double* const prob) {
        this->_alias.resize(size, -1);
        this->_prob.resize(size);

        // normalize
        double total = 0.0;
        for (size_t i = 0; i < size; ++i) {
            total += prob[i];
            _prob[i] = prob[i];
        }
        for (size_t i = 0; i < size; ++i) {
            _prob[i] *= (double)size;
            _prob[i] /= total;
        }

        // set alias table
        for (size_t i = 0; i < size; ++i) {
            size_t j = 0;
            while (_prob[i] > 1.0 && j < size) {
                for (; j < size; ++j) {
                    if (_prob[j] < 1.0 && _alias[j] < 0) {
                        _prob[i] -= (1.0 - _prob[j]);
                        _alias[j] = (int)i;
                        break;
                    }
                }
            }
        }

        // avoid floating point error
        for (size_t i = 0; i < size; ++i) {
            if (_alias[i] < 0)
                _prob[i] = 10.0;
        }
    }


    size_t AliasTable::memcpyToHostStruct(DeviceAliasData* struct_host_member_device) const {
        size_t memsize = 0x0u;

        std::vector<float> prob
            = cvtVectorDoubleToFloat(this->prob());
        struct_host_member_device->dim   = (int)this->alias().size();

        DeviceVectorHelper alias_vector(this->alias());
        memsize += alias_vector.memoryUsage();
        struct_host_member_device->alias = alias_vector.address();

        DeviceVectorHelper prob_vector(prob);
        memsize += prob_vector.memoryUsage();
        struct_host_member_device->prob  = prob_vector.address();

        return memsize;
    }


    int AliasTable::sample(double rand) const {
        double aj;
        int j;

        aj = rand * this->_alias.size();
        j = (int)aj;
        aj -= (double)j;

        if (aj > this->_prob[j])
            j = this->_alias[j];

        return j;
    }


    std::vector<double> AliasTable::probLinear() const {
        size_t n = this->dimension();
        std::vector<double> prob_linear(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            double pseg  = 1.0 / (double)n;
            double prob  = this->_prob[i];
            int    alias = this->_alias[i];
            if (prob < 1.0) {
                prob_linear[i]     += pseg * prob;
                prob_linear[alias] += pseg * (1.0 - prob);
            }
            else
                prob_linear[i] += pseg;
        }
        return prob_linear;
    }


    AliasTableMap::AliasTableMap(size_t size, const double* const prob, const int* const map) {
        this->_table = AliasTable(size, prob);
        this->_map   = std::vector<int>(map, map + size);
    }


    size_t AliasTableMap::memcpyToHostStruct(DeviceAliasDataMap* struct_host_member_device) const {
        size_t memsize = 0x0u;

        memsize += this->_table.memcpyToHostStruct(&struct_host_member_device->table);

        DeviceVectorHelper map_vector(this->map());
        memsize += map_vector.memoryUsage();
        struct_host_member_device->map = map_vector.address();

        return memsize;
    }


    int AliasTableMap::sample(double rand) const {
        return this->_map[this->_table.sample(rand)];
    }


    AliasTableEGS::AliasTableEGS() :
        _dim(make_int2(0, 0)) {};


    // Prepare EGS alias table, with 1-D domain
    AliasTableEGS::AliasTableEGS(
        const std::vector<double>& xs_array,
        const std::vector<double>& fs_array
    ) : _dim(make_int2(0, 0)), _fdata(fs_array), _xdata(xs_array) {
        if (_fdata.size() % _xdata.size())  // dimension mismatched case
            std::runtime_error("Alias table 'xdata' and 'fdata' must have same dimension");
        _dim.y = (int)_xdata.size() - 1;
        _dim.x = (int)(_fdata.size() / _xdata.size());
        for (size_t i = 0; i < _dim.x; ++i) {
            std::vector<double> aux(_dim.y, 0.5);
            for (size_t j = 0; j < _dim.y; ++j) {
                aux[j] *= (_fdata[i * ((size_t)_dim.y + 1) + j + 1] +
                    _fdata[i * ((size_t)_dim.y + 1) + j]);
                aux[j] *= (_xdata[j + 1] - _xdata[j]);
            }
            AliasTable segment(_dim.y, &aux[0]);
            _wdata.insert(_wdata.end(), segment.prob().begin(), segment.prob().end());
            _idata.insert(_idata.end(), segment.alias().begin(), segment.alias().end());
        }
        return;
    }


    AliasTableEGS::AliasTableEGS(
        const std::vector<double>& xs_array,
        const std::vector<double>& fs_array,
        int xs_dim
    ) : _dim(make_int2(0, 0)), _fdata(fs_array), _xdata(xs_array) {
        if (_fdata.size() != _xdata.size())  // dimension mismatched case
            std::runtime_error("Alias table 'xdata' and 'fdata' must have same dimension");
        if (_xdata.size() % xs_dim + 1)
            std::runtime_error("Alias table 'xdata' length must same with 'xs_dim' + 1");
        _dim.y = xs_dim;
        _dim.x = (int)(_xdata.size() / (xs_dim + 1));
        for (size_t i = 0; i < _dim.x; ++i) {
            std::vector<double> aux(_dim.y, 0.5);
            for (size_t j = 0; j < _dim.y; ++j) {
                aux[j] *= (_fdata[i * ((size_t)_dim.y + 1) + j + 1] +
                    _fdata[i * ((size_t)_dim.y + 1) + j]);
                aux[j] *= (_xdata[i * ((size_t)_dim.y + 1) + j + 1] - 
                    _xdata[i * ((size_t)_dim.y + 1) + j]);
            }
            AliasTable segment(_dim.y, &aux[0]);
            _wdata.insert(_wdata.end(), segment.prob().begin(), segment.prob().end());
            _idata.insert(_idata.end(), segment.alias().begin(), segment.alias().end());
        }
        return;
    }


    size_t AliasTableEGS::memcpyToHostStruct(DeviceAliasDataEGS* struct_host_member_device) const {
        size_t memsize = 0x0u;

        struct_host_member_device->dim   = this->_dim;

        DeviceVectorHelper xdata_vector(mcutil::cvtVectorDoubleToFloat(this->_xdata));
        memsize += xdata_vector.memoryUsage();
        struct_host_member_device->xdata = xdata_vector.address();

        DeviceVectorHelper fdata_vector(mcutil::cvtVectorDoubleToFloat(this->_fdata));
        memsize += fdata_vector.memoryUsage();
        struct_host_member_device->fdata = fdata_vector.address();

        DeviceVectorHelper wdata_vector(mcutil::cvtVectorDoubleToFloat(this->_wdata));
        memsize += wdata_vector.memoryUsage();
        struct_host_member_device->wdata = wdata_vector.address();

        DeviceVectorHelper idata_vector(this->_idata);
        memsize += idata_vector.memoryUsage();
        struct_host_member_device->idata = idata_vector.address();

        return memsize;
    }


    __host__ DeviceAliasData::DeviceAliasData() :
        dim(0), alias(nullptr), prob(nullptr) {}


    __host__ void DeviceAliasData::free() {
        DeviceVectorHelper(this->alias).free();
        DeviceVectorHelper(this->prob).free();
    }


    __host__ DeviceAliasDataMap::DeviceAliasDataMap() :
        map(nullptr) {}


    __host__ void DeviceAliasDataMap::free() {
        this->table.free();
        DeviceVectorHelper(this->map).free();
    }


    __host__ DeviceAliasDataEGS::DeviceAliasDataEGS() :
        dim({ 0, 0 }), xdata(nullptr), fdata(nullptr), idata(nullptr), wdata(nullptr) {}


    __host__ void DeviceAliasDataEGS::free() {
        DeviceVectorHelper(this->xdata).free();
        DeviceVectorHelper(this->fdata).free();
        DeviceVectorHelper(this->idata).free();
        DeviceVectorHelper(this->wdata).free();
    }


    Interp1d::Interp1d(
        const gsl_interp_type* type,
        const std::vector<double>& x,
        const std::vector<double>& y,
        bool log_x,
        bool log_y
    ) : _log_x(log_x),
        _log_y(log_y) {
        size_t dim_x = x.size();

        if (dim_x < 2)
            throw std::runtime_error("Interp1d x dimension must be larger than 1");
        if (y.size() != dim_x)
            throw std::runtime_error("Interp1d dimension mismatched");

        // ascending order test
        if (!std::is_sorted(x.begin(), x.end()))
            throw std::runtime_error("Interp1d <x> must be sorted to ascending order");
        if (log_x && x.front() <= 0.e0)
            throw std::runtime_error("Interp1d <x> must be positive in logscale case");
        if (log_y && *std::min_element(y.begin(), y.end()) <= 0.e0)
            throw std::runtime_error("Interp1d <y> must be positive in logscale case");
        this->_dim = dim_x;

        // logarithm
        std::vector<double> _x, _y;
        if (log_x) {
            for (const double& xe : x)
                _x.push_back(std::log(xe));
        }
        else
            _x = std::vector(x.begin(), x.end());
        if (log_y) {
            for (const double& ye : y)
                _y.push_back(std::log(ye));
        }
        else
            _y = std::vector(y.begin(), y.end());

        this->_xrange = { _x.front(), _x.back() };

        // set spline matrix
        this->_spline = gsl_spline_alloc(type, dim_x);
        this->_acc = gsl_interp_accel_alloc();

        gsl_spline_init(this->_spline, &_x[0], &_y[0], dim_x);
    }


    Interp1d::~Interp1d() {
        gsl_spline_free(this->_spline);
        gsl_interp_accel_free(this->_acc);
    }


    size_t Interp1d::dimension() const {
        return this->_dim;
    }


    double2 Interp1d::domain() const {
        return this->_xrange;
    }


    double Interp1d::get(double x) const {
        double y;
        if (this->_log_x)
            x = std::log(x);

        // range cap
        x = std::min(_xrange.y, std::max(_xrange.x, x));

        y = gsl_spline_eval(this->_spline, x, this->_acc);
        if (this->_log_y)
            y = std::exp(y);
        return y;
    }


    Interp2d::Interp2d(
        const std::vector<double>& x,
        const std::vector<double>& y,
        const std::vector<double>& z,
        bool log_x,
        bool log_y,
        bool log_z
    ) : _log_x(log_x), 
        _log_y(log_y), 
        _log_z(log_z) {
        size_t dim_x = x.size();
        size_t dim_y = y.size();

        if (dim_x < 2)
            throw std::runtime_error("Interp2d x dimension must be larger than 1");
        if (dim_y < 2)
            throw std::runtime_error("Interp2d y dimension must be larger than 1");
        if (z.size() != dim_x * dim_y)
            throw std::runtime_error("Interp2d dimension mismatched");

        // ascending order test
        if (!std::is_sorted(x.begin(), x.end()))
            throw std::runtime_error("Interp2d <x> must be sorted to ascending order");
        if (log_x && x.front() <= 0.e0)
            throw std::runtime_error("Interp2d <x> must be positive in logscale case");
        if (!std::is_sorted(y.begin(), y.end()))
            throw std::runtime_error("Interp2d <y> must be sorted to ascending order");
        if (log_y && y.front() <= 0.e0)
            throw std::runtime_error("Interp2d <y> must be positive in logscale case");
        if (log_z && *std::min_element(z.begin(), z.end()) <= 0.e0)
            throw std::runtime_error("Interp2d <z> must be positive in logscale case");
        this->_dim_x = dim_x;
        this->_dim_y = dim_y;

        // logarithm
        std::vector<double> _x, _y, _z;
        if (log_x) {
            for (const double& xe : x)
                _x.push_back(std::log(xe));
        }
        else
            _x = std::vector(x.begin(), x.end());
        if (log_y) {
            for (const double& ye : y)
                _y.push_back(std::log(ye));
        }
        else
            _y = std::vector(y.begin(), y.end());
        if (log_z) {
            _z.clear();
            for (const double& ze : z)
                _z.push_back(std::log(ze));
        }
        else
            _z = std::vector(z.begin(), z.end());

        this->_xrange = { _x.front(), _x.back() };
        this->_yrange = { _y.front(), _y.back() };

        // set spline matrix
        this->_spline = gsl_spline2d_alloc(this->_t, dim_x, dim_y);
        this->_xacc   = gsl_interp_accel_alloc();
        this->_yacc   = gsl_interp_accel_alloc();

        this->_za = new double[dim_x * dim_y * sizeof(double)];

        for (size_t i = 0; i < dim_x; ++i)
            for (size_t j = 0; j < dim_y; ++j)
                gsl_spline2d_set(this->_spline, this->_za, i, j, _z[i * dim_y + j]);

        gsl_spline2d_init(this->_spline, &_x[0], &_y[0], this->_za, dim_x, dim_y);
    }


    Interp2d::~Interp2d() {
        gsl_spline2d_free(this->_spline);
        gsl_interp_accel_free(this->_xacc);
        gsl_interp_accel_free(this->_yacc);
        delete[] this->_za;
    }


    int2 Interp2d::dimension() const {
        return { (int)this->_dim_x, (int)this->_dim_y };
    }


    double2 Interp2d::xdomain() const {
        return this->_xrange;
    }


    double2 Interp2d::ydomain() const {
        return this->_yrange;
    }


    double Interp2d::get(double x, double y) const {
        double z;
        if (this->_log_x)
            x = std::log(x);
        if (this->_log_y)
            y = std::log(y);

        // range cap
        x = std::min(_xrange.y, std::max(_xrange.x, x));
        y = std::min(_yrange.y, std::max(_yrange.x, y));

        z = gsl_spline2d_eval(this->_spline, x, y, this->_xacc, this->_yacc);
        if (this->_log_z)
            z = std::exp(z);
        return z;
    }


    double legendre(double x, void* params) {
        size_t* na = (size_t*)params;
        size_t  n  = *na;
        double p_n, p_n1 = x, p_n2 = 1.e0;
        switch (n) {
        case 0:
            return p_n2;
        case 1:
            return p_n1;
        default:
            double dn = (double)n;
            for (size_t i = 1; i < n; ++i) {
                p_n = (2.e0 * dn - 1.e0) / dn * x * p_n1
                    - (dn - 1.e0) / dn * p_n2;
                p_n2 = p_n1;
                p_n1 = p_n;
            }
            return p_n;
        }
    }


    std::vector<double2> gaussLegendreQuad(double x1, double x2, size_t n) {
        std::vector<double2> quad(n);
        size_t m = (n + 1) / 2;  // Legendre symmetry
        double xm = (x2 + x1) / 2.e0;
        double xl = (x2 - x1) / 2.e0;
        for (size_t i = 0; i < m; ++i) {
            // first set root bracket
            double x = std::cos(M_PI * (i + 0.75) / (n + 0.5));

            // using Newton method
            double pp;
            while (true) {
                double p3 = 1.e0, p2 = 0.e0, p1 = 1.e0;
                for (size_t j = 1; j <= n; ++j) {
                    double dj = (double)j;
                    p3 = p2;
                    p2 = p1;
                    p1 = ((2.e0 * dj - 1.e0) * x * p2
                        - (dj - 1.e0) * p3) / dj;
                }
                pp     = (double)n * (x * p1 - p2) / (x * x - 1.e0);
                double x_last = x;
                x = x_last - p1 / pp;
                if (std::abs(x - x_last) < 3.e-14) break;
            }
            // move range
            quad[i].x = xm - xl * x;
            quad[i].y = 2.e0 * xl / ((1.e0 - x * x) * pp * pp);

            // symmetry
            quad[n - i - 1].x = xm + xl * x;
            quad[n - i - 1].y = quad[i].y;
        }
        return quad;
    }


    std::string Affine::_tostr() const {
        std::stringstream ss;
        Affine::matrix mat = this->affine();
        for (size_t i = 0; i < 3; ++i) {
            ss << "    | ";
            for (size_t j = 0; j < 4; ++j) {
                ss << std::setw(12) << std::setprecision(4) << mat[i][j] << ' ';
            }
            ss << '|' << std::endl;
        }
        return ss.str();
    }


    Affine::Affine() {
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                this->_matrix[i][j] = (i == j) ? 1.e0 : 0.e0;
    }


    Affine::Affine(
        double m00, double m01, double m02, double m03,
        double m10, double m11, double m12, double m13,
        double m20, double m21, double m22, double m23
    ) {
        this->_matrix[0][0] = m00;
        this->_matrix[0][1] = m01;
        this->_matrix[0][2] = m02;
        this->_matrix[0][3] = m03;
        this->_matrix[1][0] = m10;
        this->_matrix[1][1] = m11;
        this->_matrix[1][2] = m12;
        this->_matrix[1][3] = m13;
        this->_matrix[2][0] = m20;
        this->_matrix[2][1] = m21;
        this->_matrix[2][2] = m22;
        this->_matrix[2][3] = m23;
    }


    Affine::Affine(Affine::matrix m) {
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                this->_matrix[i][j] = m[i][j];
    }


    Affine Affine::operator*(const Affine& other) {
        Affine out(*this);
        out.transform(other);
        return out;
    }


    void Affine::transform(const Affine& other) {
        Affine::matrix other_mat = other.affine();
        double new_mat[3][4];

        // Dot product
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                new_mat[i][j] = 0.e0;
                for (size_t k = 0; k < 3; ++k)
                    new_mat[i][j] += other_mat[i][k] * this->_matrix[k][j];
            }
            new_mat[i][3] += other_mat[i][3];
        }
        
        // Copy
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                this->_matrix[i][j] = new_mat[i][j];
        return;
    }


    void Affine::translate(double x, double y, double z) {
        this->transform(Affine(
            1.0, 0.0, 0.0, x,
            0.0, 1.0, 0.0, y,
            0.0, 0.0, 1.0, z
        ));
    }


    void Affine::rotate(double theta, AFFINE_AXIS axis) {
        double affine[3][4];
        // Initialize affine matrix
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                affine[i][j] = (i == j) ? 1.e0 : 0.e0;

        double cost, sint;
        theta = M_PI * theta / 180.0;
        cost = std::cos(theta);
        sint = std::sin(theta);
        size_t a, b;
        switch (axis) {
        case AFFINE_AXIS::X:
            a = 1;
            b = 2;
            break;
        case AFFINE_AXIS::Y:
            a = 2;
            b = 0;
            break;
        case AFFINE_AXIS::Z:
            a = 0;
            b = 1;
            break;
        default:
            break;
        }
        affine[a][a] = +cost;
        affine[a][b] = -sint;
        affine[b][a] = +sint;
        affine[b][b] = +cost;
        this->transform(Affine(
            affine[0][0], affine[0][1], affine[0][2], affine[0][3],
            affine[1][0], affine[1][1], affine[1][2], affine[1][3],
            affine[2][0], affine[2][1], affine[2][2], affine[2][3]
        ));
    }


    Affine::matrix Affine::affine() const {
        return this->_matrix;
    }


    std::vector<double> Affine::flatten() const {
        std::vector<double> out;
        out.reserve(12);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                out.push_back(this->_matrix[i][j]);
        return out;
    }


    Affine Affine::inverse() const {
        double det = this->determinant();
        double affine[3][4];
        
        det = 1.e0 / det;

        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                affine[j][i] = (
                    (this->_matrix[(i + 1) % 3][(j + 1) % 3] * this->_matrix[(i + 2) % 3][(j + 2) % 3]) -
                    (this->_matrix[(i + 1) % 3][(j + 2) % 3] * this->_matrix[(i + 2) % 3][(j + 1) % 3])
                    ) * det;
            }
        }
        for (size_t i = 0; i < 3; ++i) {
            affine[i][3] = 0.0;
            for (size_t j = 0; j < 3; ++j)
                affine[i][3] -= affine[i][j] * this->_matrix[j][3];
        }
        return Affine(affine);
    }


    double Affine::determinant() const {
        double det = 0.0;
        for (size_t i = 0; i < 3; ++i) {  // Determinant
            det += this->_matrix[0][i] * (
                this->_matrix[1][(i + 1) % 3] * this->_matrix[2][(i + 2) % 3] -
                this->_matrix[1][(i + 2) % 3] * this->_matrix[2][(i + 1) % 3]
                );
        }
        return det;
    }


    double3 Affine::origin() const {
        return { this->_matrix[0][3], this->_matrix[1][3], this->_matrix[2][3] };
    }


    double3 Affine::scale() const {
        return { this->_matrix[0][0], this->_matrix[1][1], this->_matrix[2][2] };
    }


    double3 transform(double3 point, const mcutil::Affine& affine) {
        mcutil::Affine::matrix mat = affine.affine();
        double pn[3] = { 0.0, 0.0, 0.0 };
        double po[3] = { point.x, point.y, point.z };
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j)
                pn[i] += mat[i][j] * po[j];
            pn[i] += mat[i][3];
        }
        return { pn[0], pn[1], pn[2] };
    }


    double dot(const double3& v1, const double3& v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }


    float dot(const float3& v1, const float3& v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }


    double3 cross(const double3& v1, const double3& v2) {
        double3 res;
        res.x = v1.y * v2.z - v1.z * v2.y;
        res.y = v1.z * v2.x - v1.x * v2.z;
        res.z = v1.x * v2.y - v1.y * v2.x;
        return res;
    }


    double norm(double3& v) {
        return std::sqrt(dot(v, v));
    }


    float norm(float3& v) {
        return std::sqrt(dot(v, v));
    }


    double3& normalize(double3& v) {
        double n = norm(v);
        v.x /= n;
        v.y /= n;
        v.z /= n;
        return v;
    }


    float3& normalize(float3& v) {
        float n = norm(v);
        v.x /= n;
        v.y /= n;
        v.z /= n;
        return v;
    }


}