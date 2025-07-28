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
 * @file    mcutil/parser/parser.tpp
 * @brief   RT2 input text line parser
 * @author  CM Lee
 * @date    05/23/2023
 */


#pragma once


namespace mcutil {


    template <typename T>
    std::vector<T> ArgumentContainer::cast() const {
        std::vector<T> formatted;
        for (const std::string& data : this->_data_literal) {
            std::stringstream converter(data);
            T value;
            converter >> value;
            if (converter.fail())
                throw StringCastException(data);
            formatted.push_back(value);
        }
        return formatted;
    }

    
    template <typename T>
    void Argument<T>::_rangeTest(const ArgumentContainer& container) const {
        std::vector<T> data = container.cast<T>();
        int smax = std::min(this->size(), (int)this->_value_minimum.size());
        for (int i = 0; i < smax; ++i) {
            std::stringstream ss_value, ss_floor, ss_ceil;
            ss_value << data[i];
            ss_floor << this->_value_minimum[i];
            ss_ceil  << this->_value_maximum[i];
            if (data[i] > this->_value_maximum[i])
                mclog::fatalOutOfRangeCeil(
                    this->_field,
                    ss_value.str(),
                    ss_ceil.str()
                );
            if (data[i] < this->_value_minimum[i])
                mclog::fatalOutOfRangeFloor(
                    this->_field,
                    ss_value.str(),
                    ss_floor.str()
                );
        }
    }


    template <typename T>
    Argument<T>::Argument(
        const std::string& field,
        int  size,
        bool is_required,
        bool has_range,
        const std::vector<T>& value_default,
        const std::vector<T>& value_minimum,
        const std::vector<T>& value_maximum
    ) :
        ArgumentInterface(field, is_required, has_range, size),
        _value_default(value_default),
        _value_minimum(value_minimum),
        _value_maximum(value_maximum) 
    {
        // check value size
        size_t value_size[3];
        value_size[0] = this->_value_default.size();
        value_size[1] = this->_value_maximum.size();
        value_size[2] = this->_value_minimum.size();
        if ((value_size[0] != value_size[1]) || 
            (value_size[0] != value_size[2]))
            throw std::length_error("Argument<T> constructor");
    }


    template <typename T>
    void Argument<T>::find(ArgInput& card) const {
        ArgInput::iterator iter = card.find(this->_field);
        if (iter == card.end()) {  // field is not found
            if (this->_is_required)
                mclog::fatalFieldRequired(this->_field);
            else {  // set to default value
                ArgumentContainer new_container;
                for (const T& data: this->_value_default) {
                    std::stringstream ss;
                    ss << data;
                    new_container.push_back(ss.str());
                    mclog::warningUseDefaultField(this->_field, ss.str());
                }
                card.insert({ this->_field, new_container });
                iter = card.find(this->_field);
            }
        }
        // now field is included in container, let's cast datatype
        try {
            iter->second.cast<T>();
        }
        catch (StringCastException& e) {
            mclog::fatalTypeCasting(this->_field, std::string(e.what()));
        }
        // check data length
        if (this->size() > 0 && this->size() != iter->second.size())
            mclog::fatalValueSize(this->_field, this->size(), iter->second.size());
        // check data range
        if (this->_has_range) {
            this->_rangeTest(iter->second);
        }
    }

    
    template <typename T>
    void ArgumentCard::insert(
        const std::string& field,
        int  size,
        bool is_required,
        bool has_range,
        const std::vector<T>& value_default,
        const std::vector<T>& value_minimum,
        const std::vector<T>& value_maximum
    ) {
        std::shared_ptr<Argument<T>> arg;
        arg = std::make_shared<Argument<T>>
            (
                field,
                size,
                is_required,
                has_range,
                value_default,
                value_minimum,
                value_maximum
            );
        this->_args.insert({ field, arg });
    }


    template <typename T>
    void ArgumentCard::insert(
        const std::string& field,
        const std::vector<T>& value_default,
        const std::vector<T>& value_minimum,
        const std::vector<T>& value_maximum
    ) {
        this->insert(
            field,
            (int)value_minimum.size(),
            false,
            true,
            value_default,
            value_minimum,
            value_maximum
        );
    }


    template <typename T>
    void ArgumentCard::insert(
        const std::string& field,
        const std::vector<T>& value_minimum,
        const std::vector<T>& value_maximum
    ) {
        this->insert(
            field,
            (int)value_minimum.size(),
            true,
            true,
            value_minimum,
            value_minimum,
            value_maximum
        );
    }


    template <typename T>
    void ArgumentCard::insert(
        const std::string& field,
        const std::vector<T>& value_default
    ) {
        this->insert(
            field,
            (int)value_default.size(),
            false,
            false,
            value_default,
            value_default,
            value_default
        );
    }


    template <typename T>
    void ArgumentCard::insert(
        const std::string& field,
        int size
    ) {
        std::vector<T> dummy(std::max(0, size));
        this->insert(
            field,
            size,
            true,
            false,
            dummy,
            dummy,
            dummy
        );
    }


    template <typename T>
    void ArgumentCard::insertUnlimitedFieldWithDefault(
        const std::string& field,
        const std::vector<T>& value_default
    ) {
        this->insert(
            field,
            -1,
            false,
            false,
            value_default,
            value_default,
            value_default
        );
    }


    template <typename T>
    T stringTo(const std::string& str) {
        std::stringstream converter(str);
        T value = 0x0;
        converter >> value;
        if (converter.fail()) {
            std::stringstream ss;
            ss << "Type conversion error detected at value '"
                << str << "'";
            mclog::fatal(ss);
        }
        return value;
    }


}