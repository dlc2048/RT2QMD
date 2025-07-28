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
 * @file    mcutil/parser/parser.hpp
 * @brief   RT2 input text line parser
 * @author  CM Lee
 * @date    05/23/2023
 */


#pragma once

#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <deque>
#include <regex>
#include <map>
#include <filesystem>

#include <cuda_runtime.h>

#include "mclog/logger.hpp"
#include "prompt/env.hpp"


namespace mcutil {


    inline const std::filesystem::path SYNTAX_HOME = std::filesystem::path("resource/syntax");


    inline const std::regex FIELD_PATTERN(R"(^--(.+))");


    // text and equation parsing 


    std::deque<std::string> split(const std::string& line, const char sep = '\0', const int max_split = -1);


    std::deque<std::string> split(const std::string& line, const std::string& delimiters);


    std::string join(const std::deque<std::string>&  segs, const std::string sep = " ");
    std::string join(const std::vector<std::string>& segs, const std::string sep = " ");


    /**
    * @brief String to numeric type casting exception
    */
    class StringCastException : public std::exception {
    protected:
        std::string _msg;
    public:
        explicit StringCastException(const char* message)
            : _msg(message) {}
        explicit StringCastException(const std::string& message)
            : _msg(message) {}
        virtual ~StringCastException() noexcept {}
        virtual const char* what() const noexcept {
            return _msg.c_str();
        }
    };


    /**
    * @brief Unformatted string value container of argument type input
    */
    class ArgumentContainer {
    private:
        std::vector<std::string> _data_literal;   //!< @brief string data
    public:


        ArgumentContainer() {}


        /**
        * @brief Get the literal data size of this container
        */
        size_t size() const { return this->_data_literal.size(); }


        /**
        * @brief Push back a literal data
        * @param data Literal data
        */
        void push_back(const std::string& data) { this->_data_literal.push_back(data); }


        /**
        * @brief Erase all literal and byte data
        */
        void clear() { this->_data_literal.clear(); }


        /**
        * @brief Cast the array of byte data to array of T
        *
        * @return Array of type T
        */
        template <typename T>
        std::vector<T> cast() const;


    };


    template <>
    std::vector<std::string> ArgumentContainer::cast() const;

    template <>
    std::vector<bool> ArgumentContainer::cast() const;


    typedef std::map<std::string, ArgumentContainer> ArgInput;


    /**
    * @brief Argument field interface class for the argument type polymorphism
    */
    class ArgumentInterface {
    protected:
        std::string _field;  //! @brief Field name
        bool _is_required;  //! @brief This field is necessary if true
        bool _has_range;    //! @brief This field has range if true
        int  _size;         //! @brief Size of argument lists (-1 = flexible)


        /**
        * @brief Argument field interface class for the argument type polymorphism
        * @param field       Field name
        * @param is_required It will become essential if true
        * @param has_range   It will have range if true
        */
        ArgumentInterface(
            const std::string& field,
            bool is_required,
            bool has_range,
            int  size
        ) : _field(field), _is_required(is_required), _has_range(has_range), _size(size) {}


    public:


        /**
        * @brief Get the field name
        */
        const std::string& field() const { return this->_field; }


        int size() const { return this->_size; }


        virtual void find(ArgInput& card) const;


    };


    /**
    * @brief Pre-defined field argument
    */
    template <typename T>
    class Argument : public ArgumentInterface {
    private:
        std::vector<T> _value_default;  //!< @brief Default value
        std::vector<T> _value_minimum;  //!< @brief Range floor
        std::vector<T> _value_maximum;  //!< @brief Range ceil


        /**
        * @brief Check the set of value of target container
        *        are fall within value range or not
        * @details Logger will raise fatal error and
        *          program will be terminated if fail
        */
        void _rangeTest(const ArgumentContainer& container) const;


    public:


        /**
        * @brief Pre-defined   field argument
        * @param field         Field name
        * @param size          Argument size
        * @param is_required   It will become essential if true
        * @param has_range     It will have range if true
        * @param value_default Default value
        * @param value_minimum Value range floor
        * @param value_maximum Value range ceil
        */
        Argument(
            const std::string& field,
            int  size,
            bool is_required,
            bool has_range,
            const std::vector<T>& value_default,
            const std::vector<T>& value_minimum,
            const std::vector<T>& value_maximum
        );


        /**
        * @brief Find the field and values from argtype card input.
        *        Check the essential and test the range condition.
        *        If fail, logger raise fatal error and program will
        *        be terminated. Elsewhere, argtype card contents are
        *        filled automatically by range condition
        */
        void find(ArgInput& card) const;


    };


    template <>
    void Argument<std::string>::_rangeTest(const ArgumentContainer& container) const;


    typedef std::map<std::string, std::shared_ptr<ArgumentInterface>>  ArgList;
    typedef std::pair<std::string, std::shared_ptr<ArgumentInterface>> ArgPair;

    
    /**
    * @brief Argument type card context
    */
    class ArgumentCard {
    private:
        std::string _key;   //!< @brief Card name
        ArgList     _args;  //!< @brief List of field arguments


        /**
        * @brief Insert field algument, internal method
        * @param field         Field name
        * @param is_required   It will become essential if true
        * @param has_range     It will have range if true
        * @param value_default Default value 
        * @param value_minimum Value range floor
        * @param value_maximum Value range ceil
        */
        template <typename T>
        void insert(
            const std::string& field,
            int  size,
            bool is_required,
            bool has_range,
            const std::vector<T>& value_default,
            const std::vector<T>& value_minimum,
            const std::vector<T>& value_maximum
        );


    public:


        /**
        * @brief Argument type card
        * @param key Card name
        */
        ArgumentCard(const std::string& key) : _key(key) {}


        /**
        * @brief Insert field algument, optional but range bounded
        * @param field         Field name
        * @param value_default Default value
        * @param value_minimum Value range floor
        * @param value_maximum Value range ceil
        */
        template <typename T>
        void insert(
            const std::string& field,
            const std::vector<T>& value_default,
            const std::vector<T>& value_minimum,
            const std::vector<T>& value_maximum
        );


        /**
        * @brief Insert field algument, essential and range bounded
        * @param field         Field name
        * @param value_minimum Value range floor
        * @param value_maximum Value range ceil
        */
        template <typename T>
        void insert(
            const std::string& field,
            const std::vector<T>& value_minimum,
            const std::vector<T>& value_maximum
        );


        /**
        * @brief Insert field algument, optional and unbounded
        * @param field         Field name
        * @param value_default Default value
        */
        template <typename T>
        void insert(
            const std::string& field,
            const std::vector<T>& value_default
        );


        /**
        * @brief Insert field algument, essential but unbounded
        * @param field Field name
        * @param size  Value size
        */
        template <typename T>
        void insert(
            const std::string& field,
            int size = -1
        );


        template <typename T>
        void insertUnlimitedFieldWithDefault(
            const std::string& field,
            const std::vector<T>& value_default
        );


        void insert(
            const std::string& field,
            std::shared_ptr<ArgumentInterface> arg
        );


        void erase(const std::string& field) { this->_args.erase(field); }


        const std::string& key() const { return this->_key; }


        size_t size() const { return this->_args.size(); }


        std::vector<std::string> fieldNameList() const;


        const std::shared_ptr<ArgumentInterface> field(const std::string& field) const;


        void get(ArgInput& container);


        void printHelp();


    };


    // naming convention


    /**
    * @brief Check whether entered string has special
    *        character except the underscore or not
    *
    * @return Boolean condition
    */
    bool isHasSpecialSymbol(const std::string& str);


    bool isHasDigitHeader(const std::string& str);


    const std::string& ordinalSuffix(int n);


    template <typename T>
    T stringTo(const std::string& str);


}


// template definitions
#include "parser.tpp"