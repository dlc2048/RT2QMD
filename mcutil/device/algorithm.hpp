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
 * @file    mcutil/device/algorithm.hpp
 * @brief   Host side interpolation, sampling and integral algorithms
 * @author  CM Lee
 * @date    05/23/2023
 */

#pragma once

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iomanip>

#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "memory.hpp"
#include "algorithm.cuh"


namespace mcutil {


    class LogLogCoeff {
    private:
        double2 _llx;
    public:


        LogLogCoeff(double xmin, double xmax, size_t nbin);


        double2 llx() const { return this->_llx; }


    };


    /**
    * @brief Simple one dimensional alias table, host side
    */
    class AliasTable {
    private:
        std::vector<int>    _alias;  //!< @brief Alias index table
        std::vector<double> _prob;   //!< @brief Alias probability table
    public:


        AliasTable() {}


        /**
        * @brief Simple one dimensional alias table, host side
        * @param size Total size of domain
        * @param prob Sampling probability
        */
        AliasTable(size_t size, const double* const prob);


        AliasTable(const std::vector<int>& alias, const std::vector<double>& prob) :
            _alias(alias), _prob(prob) {}


        /**
        * @brief Alias index table
        */
        const std::vector<int>& alias() const { return this->_alias; }


        /**
        * @brief Alias probability table
        */
        const std::vector<double>& prob() const { return this->_prob; }


        /**
        * @brief Copy entire data from host class (this)
        *        to host struct with device member (DeviceAliasData)
        * @param struct_host_member_dev Host side alias table structure
        * 
        * @return Size of allocated memory [bytes]
        */
        size_t memcpyToHostStruct(DeviceAliasData* struct_host_member_dev) const;


        /**
        * @brief Sample alias index from table
        * @param rand Random number [0,1)
        * 
        * @return Alias index [0,_alias.size)
        */
        int sample(double rand) const;


        size_t dimension() const { return this->_alias.size(); }


        /**
        * @brief Original probability matrix
        */
        std::vector<double> probLinear() const;


    };


    /**
    * @brief Simple one dimensional alias table with index map, host side
    */
    class AliasTableMap {
    private:
        AliasTable       _table;
        std::vector<int> _map;
    public:


        AliasTableMap() {}


        AliasTableMap(size_t size, const double* const prob, const int* const map);


        const AliasTable&       table() const { return this->_table; }
        const std::vector<int>& map()   const { return this->_map; }


        size_t memcpyToHostStruct(DeviceAliasDataMap* struct_host_member_device) const;


        int sample(double rand) const;


    };


    /**
    * @brief Two dimensional EGS-style alias table, host side
    */
    class AliasTableEGS {
    private:
        int2                _dim;    //!< @brief Alias table dimension {group, domain}
        std::vector<double> _xdata;  //!< @brief Alias table domain
        std::vector<double> _fdata;  //!< @brief Alias table value
        std::vector<int>    _idata;  //!< @brief Alias table index
        std::vector<double> _wdata;  //!< @brief Alias probability
    public:


        AliasTableEGS();


        /**
        * @brief Initialize host side, two dimensional EGS-style alias table.
        *        All group share same domain (xdata)
        * @param xs_array Alias table domain
        * @param fs_array Alias table value. Length must be multiple of length of xs_array
        */
        AliasTableEGS(
            const std::vector<double>& xs_array,
            const std::vector<double>& fs_array
        );


        /**
        * @brief Initialize host side, two dimensional EGS-style alias table.
        *        All group has their's own domain (xdata)
        * @param xs_array Alias table domain. Length must be multiple of xs_dim
        *        and same with length of fs_array
        * @param fs_array Alias table value. Length must be multiple of xs_dim
        *        and same with length of fs_array
        * @param xs_dim   Length of alias table domain
        */
        AliasTableEGS(
            const std::vector<double>& xs_array,
            const std::vector<double>& fs_array,
            int xs_dim
        );


        /**
        * @brief Alias table dimension {group, domain}
        */
        int2 dimension() const { return this->_dim; }


        /**
        * @brief Alias table domain
        */
        const std::vector<double>& xdata() const { return this->_xdata; }


        /**
        * @brief Alias table value
        */
        const std::vector<double>& fdata() const { return this->_fdata; }


        /**
        * @brief Alias table index
        */
        const std::vector<int>&    idata() const { return this->_idata; }


        /**
        * @brief Alias probability
        */
        const std::vector<double>& wdata() const { return this->_wdata; }


        size_t memcpyToHostStruct(DeviceAliasDataEGS* struct_host_member_device) const;


    };


    /**
    * @brief One-dimensional spline interpolator
    */
    class Interp1d {
    private:
        size_t            _dim;     //!< @brief Length of domain and value
        bool              _log_x;   //!< @brief Do log scale interpolation in domain if true, linear otherwise
        bool              _log_y;   //!< @brief Do log scale interpolation in range if true, linear otherwise
        double2           _xrange;  //!< @brief Range of x {min, max}
        gsl_spline*       _spline;  //!< @brief GSL spline engine
        gsl_interp_accel* _acc;     //!< @brief GSL interpolation accelerator
    public:


        Interp1d() {};


        /**
        * @brief One-dimensional spline interpolator
        * @param type  GSL interpolation type
        * @param x     Interpolation domain
        * @param y     Interpolation range
        * @param log_x Do log scale interpolation in domain if true, linear otherwise
        * @param log_y Do log scale interpolation in range if true, linear otherwise
        */
        Interp1d(
            const gsl_interp_type* type,
            const std::vector<double>& x,
            const std::vector<double>& y,
            bool log_x=false,
            bool log_y=false
        );


        ~Interp1d();


        /**
        * @brief Length of domain and value
        */
        size_t dimension() const;


        double2 domain() const;


        /**
        * @brief calculate interpolated function range from given x
        * @param x Function domain
        * 
        * @return y of given x
        */
        double get(double x) const;
    };


    /**
    * @brief Two-dimensional spline interpolator
    */
    class Interp2d {
    private:
        size_t  _dim_x;             //!< @brief Length of first domain, x
        size_t  _dim_y;             //!< @brief Length of second domain, y
        bool    _log_x;             //!< @brief Do log scale interpolation in x if true, linear otherwise
        bool    _log_y;             //!< @brief Do log scale interpolation in y if true, linear otherwise
        bool    _log_z;             //!< @brief Do log scale interpolation in z if true, linear otherwise
        double2 _xrange;            //!< @brief Range of x {min, max}
        double2 _yrange;            //!< @brief Range of y {min, max}
        const gsl_interp2d_type* _t 
            = gsl_interp2d_bilinear;
        gsl_spline2d*     _spline;  //!< @brief GSL spline engine
        gsl_interp_accel* _xacc;    //!< @brief GSL interpolation accelerator, x-axis
        gsl_interp_accel* _yacc;    //!< @brief GSL interpolation accelerator, y-axis
        double* _za;                //!< @brief Function z value
    public:


        /**
        * @brief Two-dimensional spline interpolator
        * @param x     Interpolation domain, abscissa
        * @param y     Interpolation domain, ordinate
        * @param z     Interpolation range
        * @param log_x Do log scale interpolation in domain x if true, linear otherwise
        * @param log_y Do log scale interpolation in domain y if true, linear otherwise
        * @param log_z Do log scale interpolation in range if true, linear otherwise
        */
        Interp2d(
            const std::vector<double>& x,
            const std::vector<double>& y,
            const std::vector<double>& z,
            bool log_x=false,
            bool log_y=false,
            bool log_z=false
        );


        ~Interp2d();


        /*
        * @brief Dimension of domain {dim_x, dim_y}
        */
        int2 dimension() const;


        double2 xdomain() const;


        double2 ydomain() const;


        /**
        * @brief calculate interpolated function range from given x and y
        * @param x Function domain, abscissa 
        * @param y Function domain, ordinate
        *
        * @return z of given x and y
        */
        double get(double x, double y) const;
    };


    enum class AFFINE_AXIS {
        X,
        Y,
        Z
    };


    /**
    * @brief Affine transform
    */
    class Affine {
    private:
        double _matrix[3][4];
    protected:
        /**
        * @brief Get the affine matrix by string format
        *
        * @return summary string
        */
        std::string _tostr() const;


    public:
        typedef const double(&matrix)[3][4];


        Affine();


        Affine(
            double m00, double m01, double m02, double m03,
            double m10, double m11, double m12, double m13,
            double m20, double m21, double m22, double m23
        );


        Affine(Affine::matrix m);


        /**
        * @brief Get the nested Affine transformation
        * @param other Secondary transform
        * 
        * @return Nested Affine transformation object
        */
        Affine operator*(const Affine& other);


        /**
        * @brief Get the nested Affine transformation.
        *        This object will be conveted directly
        * @param other Secondary transform
        */
        virtual void transform(const Affine& other);


        /**
        * @brief Translate this Affine transformation
        * @param x Delta-x 
        * @param y Delta-y
        * @param z Delta-z
        */
        void translate(double x, double y, double z);


        /**
        * @brief Rotate this Affine transformation
        * @param theta Rotation angle (degree)
        * @param axis  Rotation axis (x, y or z)
        */
        void rotate(double theta, AFFINE_AXIS axis);


        /**
        * @brief Get the affine matrix
        * 
        * @return 3 x 4 affine matrix
        */
        Affine::matrix affine() const;


        /**
        * @brief Get the flattened affine matrix
        * 
        * @return length 12 affine matrix
        */
        std::vector<double> flatten() const;


        /**
        * @brief Get the inverse affine object
        * 
        * @return 4 x 3 inverse affine matrix
        */
        Affine inverse() const;


        /**
        * @brief Calculate the determinant of affine matrix
        * 
        * @return determinant
        */
        double determinant() const;


        double3 origin() const;


        double3 scale() const;


    };


    /**
    * @brief Get the value of Legendre polynomial P_n(x)
    * @param x      Domain in range[-1,1]
    * @param params Order of Legendre polynomial
    * 
    * @return Legendre polynomial P_n(x) for given x and n
    */
    double legendre(double x, void* params);


    /**
    * @brief Calculate the abscissas and weight of the Gauss-Legendre 
    *        n-point quadrature formula from x1 to x2
    * @param x1 Floor of quadrature
    * @param x2 Ceil of quadrature
    * @param n Number of points
    * 
    * @return Vector of {abscissa, weight}, size of n
    */
    std::vector<double2> gaussLegendreQuad(double x1, double x2, size_t n);


    double3 transform(double3 point, const mcutil::Affine& affine);


    double dot(const double3& v1, const double3& v2);


    float dot(const float3& v1, const float3& v2);


    double3 cross(const double3& v1, const double3& v2);


    double norm(double3& v);


    float norm(float3& v);


    double3& normalize(double3& v);


    float3& normalize(float3& v);


}