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
 * @file    mcutil/mclog/logger.tpp
 * @brief   RT2 log system
 * @author  CM Lee
 * @date    05/23/2023
 */


#include <assert.h>


namespace mclog {


    // default
    //template <typename T, typename Enable>
    //FormattedTable& operator<<(FormattedTable& cls, T _val) {
    //    size_t len = cls._len[cls._lpos];
    //    cls._lpos++;
    //    cls._ss << std::setw(len);

        //else if (std::is_integral<T>::value)
        //    cls._ss << (std::abs(_val) >= 10000 ? std::scientific : std::fixed);
    //    cls._ss << _val;
    //    cls._ss << " ";
    //    return cls;
    //}


   //template <typename T, typename Enable>
    ///FormattedTable& operator<<(FormattedTable& cls, T& _val) {
   //     size_t len = cls._len[cls._lpos];
    //    cls._lpos++;
    //    cls._ss << std::setw(len);
    //    cls._ss << _val;
    //    cls._ss << " ";
    //    return cls;
   // }


    template <typename T>
    void fatalListElementAlreadyExist(const T& element) {
        std::stringstream message;
        message << "Element '" << element << "' is duplicated";
        fatal(message);
    }


    template <typename T, typename std::enable_if_t<std::is_floating_point<T>::value>*>
    FormattedTable& FormattedTable::operator<<(T _val) {
        size_t len = this->_len[this->_lpos];
        this->_lpos++;
        this->_ss << std::setw(len);

        // numeric (1st try)
        std::stringstream ss;
        std::string ss_str;
        switch (this->_type) {
        case FORMAT_TYPE::AUTO:
            ss << ((_val != 0.0 && (std::abs(_val) < 1e-3 || std::abs(_val) > 1e+4)) ? std::scientific : std::fixed);
            break;
        case FORMAT_TYPE::NUMERIC:
            ss << std::fixed;
            break;
        case FORMAT_TYPE::SCIENTIFIC:
            ss << std::scientific;
            break;
        default:
            assert(false);
        }
        ss << _val;
        ss_str = ss.str();

        if (ss_str.length() > len) {  // forced scientific (2nd try)
            ss.str("");
            ss.clear();
            ss << std::scientific << std::setprecision(len - 6) << _val;
            ss_str = ss.str();
        }

        this->_ss << ss_str;
        this->_ss << " ";

        return *this;
    }


    template <typename T, typename std::enable_if_t<std::is_integral<T>::value>*>
    FormattedTable& FormattedTable::operator<<(T _val) {
        size_t len = this->_len[this->_lpos];
        this->_lpos++;
        this->_ss << std::setw(len);

        // numeric (1st try)
        std::stringstream ss;
        std::string ss_str;
        switch (this->_type) {
        case FORMAT_TYPE::AUTO:
            ss << ((std::llabs(_val) >= 10000) ? std::scientific : std::fixed);
            break;
        case FORMAT_TYPE::NUMERIC:
            ss << std::fixed;
            break;
        case FORMAT_TYPE::SCIENTIFIC:
            ss << std::scientific;
            break;
        default:
            assert(false);
        }
        ss << _val;
        ss_str = ss.str();

        if (ss_str.length() > len) {  // forced scientific (2nd try)
            ss.str("");
            ss.clear();
            ss << std::scientific << std::setprecision(len - 6) << _val;
            ss_str = ss.str();
        }

        this->_ss << ss_str;
        this->_ss << " ";

        return *this;
    }


}
