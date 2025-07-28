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
 * @file    mcutil/parser/input.hpp
 * @brief   Input stream handler
 * @author  CM Lee
 * @date    05/23/2023
 */


#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <regex>
#include <deque>
#include <exception>
#include <filesystem>

#include "mclog/logger.hpp"
#include "parser/parser.hpp"


namespace mcutil {


    typedef std::map<std::string, std::string> Macro;


    /**
    * @brief Input stream handler
    */
    class Input {
    private:
        bool          _psyntax;
        std::ifstream _stream;       //! @brief text file stream
        size_t        _pos_current;  //! @brief current stream position
        size_t        _pos_last;     //! @brief last stream position
        std::string   _line_last;    //! @brief last line index
        std::map<std::string, std::string> _macro;  //!< @brief Macro statements 


        /**
        * @brief Set macro statements, which has format '$KEY$ value'
        */
        void _setMacro();


    public:


        /**
        * @brief Input stream handler
        * @param file_name Textfile name
        */
        Input(const std::string& file_name, bool psyntax);


        ~Input() { this->close(); }


        /**
        * @brief Go to target line index
        * @param line Line index
        */
        void seek(size_t line);


        /**
        * @brief Initialize file stream
        */
        void init() { this->seek(0); }


        void close();


        /**
        * @brief Get single character from the current stream position
        *
        * @return Character
        */
        char get();


        /**
        * @brief Check stream have hit end of file or not
        *
        * @return True if stream hit EOF
        */
        bool eof() { return this->_stream.eof(); }


        /**
        * @brief Print previous line contents on logger
        */
        void printPreviousLine() const { mclog::line(this->_pos_last, this->_line_last); }


        bool printSyntax() const { return this->_psyntax; }


        /**
        * @brief Parse next lines
        *
        * @return Splitted line (Deque of string)
        */
        std::deque<std::string> nextLine();


        /**
        * @brief Parse next lines and convert them to argument type context
        * @details Argument type contexts are have following structure:
        *          CARDNAME --field1 value1 --field2 value2 ...
        *
        * @return Argument type context <Cardname, Set of field-value pair>
        */
        std::pair<std::string, ArgInput> nextCard();


    };


    // Card factory (type T must be RT2 card context)
    template <typename T>
    class InputCardFactory {
    private:


        // Factory strategy
        static ArgumentCard _setCard();


    public:


        /**
        * @brief Read all cards those have cardname of this context
        *        and convert them to argtype card deque
        * @param input    Input text file stream handler
        * @param max_inst Maximum length of output deque if positive,
        *        size is unlimited elsewhere
        *
        * @return Argtype card deque
        */
        static std::deque<T> readAll(Input& input, int max_inst = -1);


        static ArgumentCard getCardDefinition() { return InputCardFactory<T>::_setCard(); }


    };


}


#include "input.tpp"