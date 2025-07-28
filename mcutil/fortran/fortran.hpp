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
 * @file    mcutil/fortran/fortran.hpp
 * @brief   Fortran-style binary file I/O
 * @author  CM Lee
 * @date    05/23/2023
 */


#pragma once

#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <exception>

#include "mclog/logger.hpp"


namespace mcutil {


    enum class SAVE_POLICY {
        SAVE_NEW    = 0,  // overwrite data
        SAVE_APPEND = 1   // append data
    };


    /**
    * @brief Formatted Fortran binary input
    */
    class FortranIfstream {
    protected:
        std::ifstream _istream;
    public:


        /**
        * @brief Open formatted Fortran binary input
        * @param file_name Binary file name
        */
        FortranIfstream(const std::string& file_name);


        ~FortranIfstream();


        /**
        * @brief Read and abandon formatted block
        */
        void skip();


        /**
        * @brief Check stream error
        */
        bool fail();


        /**
        * @brief Check end of file (EOF)
        */
        bool eof();


        /**
        * @brief Read formatted fortran block.
        *        Block length must be the multiple of sizeof(T)
        * 
        * @return Vector<T> in size of size/sizeof<T>
        */
        template <typename T>
        std::vector<T> read();
    };


    /**
    * @brief Unformatted Fortran binary input
    */
    class FortranIfunformatted : public FortranIfstream {
    private:
        size_t _recl;  //!< @brief Record length
    public:


        /**
        * @brief Unformatted Fortran binary input
        * @param file_name Binary file name
        * @param recl      Bytes record length
        */
        FortranIfunformatted(const std::string& file_name, size_t recl = 1);
        void skip() = delete;
        template <typename T>
        std::vector<T> read() = delete;


        /**
        * @brief Read unformatted fortran block.
        *        Bytes record is begin in position (recl * rec)
        * @param rec  Record position
        * @param size Size of block. Size must be the multiple of sizeof(T)
        * 
        * @return Vector<T> in size of size/sizeof<T>
        */
        template <typename T>
        std::vector<T> read(size_t rec, size_t size);
    };


    /**
    * @brief Formatted Fortran binary output
    */
    class FortranOfstream {
    protected:
        std::ofstream _ostream;
    public:


        /**
        * @brief Formatted Fortran binary output
        * @param file_name Binary file name
        */
        FortranOfstream(const std::string file_name, SAVE_POLICY save_policy = SAVE_POLICY::SAVE_NEW);


        ~FortranOfstream();


        /**
        * @brief Check stream error
        */
        bool fail();


        /**
        * @brief Write formatted fortran block.
        * @param buffer Byte buffer
        * @param size   Record size
        */
        void write(const unsigned char* buffer, size_t size);


        /**
        * @brief Write formatted fortran block. 
        *        Vector is written in length of sizeof(T) * list.size()
        * @param list 1-D vector
        */
        template <typename T>
        void write(const std::vector<T>& list);
    };


    /**
    * @brief Unformatted Fortran binary output
    */
    class FortranOfunformatted : public FortranOfstream {
    public:


        /**
        * @brief Unformatted Fortran binary output
        * @param file_name Binary file name
        */
        FortranOfunformatted(const std::string file_name, SAVE_POLICY save_policy = SAVE_POLICY::SAVE_NEW);


        /**
        * @brief Write unformatted fortran block.
        * @param buffer Byte buffer
        * @param size   Record size
        */
        void write(const unsigned char* buffer, size_t size);


        /**
        * @brief Write unformatted fortran block.
        *        Vector is written in length of sizeof(T) * list.size()
        * @param list 1-D vector
        */
        template <typename T>
        void write(const std::vector<T>& list);
    };

}


// template definitions
#include "fortran.tpp"