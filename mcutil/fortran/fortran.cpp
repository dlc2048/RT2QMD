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
 * @file    mcutil/fortran/fortran.cpp
 * @brief   Fortran-style binary file I/O
 * @author  CM Lee
 * @date    05/23/2023
 */


#include "fortran.hpp"


namespace mcutil {


    FortranIfstream::FortranIfstream(const std::string& file_name) {
        std::stringstream ss;
        ss << "Read fortran binary '" << file_name << "'";
        mclog::debug(ss);
        this->_istream.open(file_name, std::ios::binary | std::ios::in);
        if (this->fail()) {
            std::string ecode = "Fail to open file '" + file_name + "'";
            throw std::ios_base::failure(ecode);
        }
    }


    FortranIfstream::~FortranIfstream() {
        this->_istream.close();
    }


    void FortranIfstream::skip() {
        int blen1 = 0, blen2 = 0;
        std::vector<unsigned char> buffer;
        this->_istream.read(reinterpret_cast<char*>(&blen1), sizeof(int));

        buffer = std::vector<unsigned char>(blen1);
        this->_istream.read(reinterpret_cast<char*>(&buffer[0]), blen1);
        this->_istream.read(reinterpret_cast<char*>(&blen2), sizeof(int));
        if (blen1 != blen2)
            throw std::length_error("Fortran block delimiter mismatched");
    }


    bool FortranIfstream::fail() {
        return this->_istream.fail();
    }


    bool FortranIfstream::eof() {
        return this->_istream.eof();
    }


    FortranIfunformatted::FortranIfunformatted(const std::string& file_name, size_t recl) 
        : FortranIfstream(file_name), _recl(recl) {
    }


    FortranOfstream::FortranOfstream(const std::string file_name, SAVE_POLICY save_policy) {
        std::stringstream  ss;
        std::ios::openmode mode;
        ss << "Write ";
        if (save_policy == SAVE_POLICY::SAVE_NEW) {
            ss << "(overwrite)";
            mode = std::ios::binary | std::ios::out;
        }
        else {
            ss << "(append)";
            mode = std::ios::binary | std::ios::app;
        }
        ss << " fortran binary '" << file_name << "'";
        mclog::debug(ss);

        this->_ostream.open(file_name, mode);
        if (this->fail()) {
            std::string ecode = "Fail to open file '" + file_name + "'";
            throw std::ios_base::failure(ecode);
        }
    }


    FortranOfstream::~FortranOfstream() {
        this->_ostream.close();
    }


    bool FortranOfstream::fail() {
        return this->_ostream.fail();
    }


    void FortranOfstream::write(const unsigned char* buffer, size_t size) {
        int blen = (int)size;
        this->_ostream.write(reinterpret_cast<char*>(&blen), sizeof(int));
        this->_ostream.write(reinterpret_cast<const char*>(buffer), blen);
        this->_ostream.write(reinterpret_cast<char*>(&blen), sizeof(int));
    }


    FortranOfunformatted::FortranOfunformatted(const std::string file_name, SAVE_POLICY save_policy) :
        FortranOfstream(file_name, save_policy) {}


    void FortranOfunformatted::write(const unsigned char* buffer, size_t size) {
        int blen = (int)size;
        this->_ostream.write(reinterpret_cast<const char*>(buffer), blen);
    }


}
