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
 * @file    mcutil/fortran/fortran.tpp
 * @brief   Fortran-style binary file I/O
 * @author  CM Lee
 * @date    05/23/2023
 */


#pragma once


namespace mcutil {

    template <typename T>
    std::vector<T> FortranIfstream::read() {
        int blen1, blen2;
        std::vector<unsigned char> buffer;
        this->_istream.read(reinterpret_cast<char*>(&blen1), sizeof(int));

        buffer = std::vector<unsigned char>(blen1);
        this->_istream.read(reinterpret_cast<char*>(&buffer[0]), blen1);
        this->_istream.read(reinterpret_cast<char*>(&blen2), sizeof(int));

        if (blen1 != blen2)
            throw std::length_error("Fortran block delimiter mismatched");
        if (blen1 % sizeof(T))
            throw std::length_error("Fortran block type casting violation");
        T* buffer_cast = reinterpret_cast<T*>(&buffer[0]);
        std::vector<T> out;
        out.resize(blen1 / sizeof(T));
        std::memcpy(&out[0], buffer_cast, blen1);

        return out;
    }


    template <typename T>
    std::vector<T> FortranIfunformatted::read(size_t rec, size_t size) {
        std::vector<unsigned char> buffer;
        _istream.seekg(this->_recl * rec);
        buffer = std::vector<unsigned char>(size);
        if (size % sizeof(T))
            throw std::length_error("Fortran record type casting violation");
        _istream.read(reinterpret_cast<char*>(&buffer[0]), size);
        T* buffer_cast = reinterpret_cast<T*>(&buffer[0]);
        std::vector<T> out;
        out.resize(size / sizeof(T));
        std::memcpy(&out[0], buffer_cast, size);

        return out;
    }


    template <typename T>
    void FortranOfstream::write(const std::vector<T>& list) {
        if (list.empty()) {
            unsigned char dummy;
            FortranOfstream::write(reinterpret_cast<const unsigned char*>(&dummy), sizeof(T) * list.size());
        }
        else
            FortranOfstream::write(reinterpret_cast<const unsigned char*>(&list[0]), sizeof(T) * list.size());
    }


    template <typename T>
    void FortranOfunformatted::write(const std::vector<T>& list) {
        if (list.empty()) {
            unsigned char dummy;
            FortranOfunformatted::write(reinterpret_cast<const unsigned char*>(&dummy), sizeof(T) * list.size());
        }
        else
            FortranOfunformatted::write(reinterpret_cast<const unsigned char*>(&list[0]), sizeof(T) * list.size());
    }

}