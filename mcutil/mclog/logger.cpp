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
 * @file    mcutil/mclog/logger.cpp
 * @brief   RT2 log system
 * @author  CM Lee
 * @date    05/23/2023
 */


#include "logger.hpp"


namespace mclog {


    McrtLogger::McrtLogger() :
        _max_length(80),
        _log_level(LEVEL::LINE)
    {
        this->_header.insert({ LEVEL::OFF,   "" });
        this->_header.insert({ LEVEL::FATAL, " [FATAL] " });
        this->_header.insert({ LEVEL::WARN,  " [WARNING] " });
        this->_header.insert({ LEVEL::PRINT, "" });
        this->_header.insert({ LEVEL::INFO,  " [INFO] " });
        this->_header.insert({ LEVEL::LINE,  " [LINE] " });
        this->_header.insert({ LEVEL::DEBUG, " [DEBUG] " });
    };


    McrtLogger::~McrtLogger() {
        this->close();
    }


    void McrtLogger::_printCout(std::stringstream& ss) {
        std::string token;
        while (std::getline(ss, token, '\n')) {
            std::cout << token << std::endl;
        }
    }


    void McrtLogger::_printFile(std::stringstream& ss) {
        std::string token;
        while (std::getline(ss, token, '\n')) {
            this->_file << token << std::endl;
        }
    }


    void McrtLogger::setMaxLength(size_t length) {
        this->_max_length = length;
    }


    void McrtLogger::setLevel(LEVEL level) {
        this->_log_level = level;
    }


    bool McrtLogger::open(const std::string& file_name) {
        this->_file = std::ofstream(file_name);
        return this->_file.is_open();
    }


    void McrtLogger::close() {
        if (this->_file.is_open())
            this->_file.close();
    }


    void McrtLogger::print(LEVEL level, std::stringstream& ss) {
        bool is_empty_line = true;
        const std::string& header = this->_header.find(level)->second;
        size_t header_size = header.size();
        std::stringstream ns;

        // generate empty spaceline
        std::string space("");
        {
            std::stringstream space_ss;
            for (size_t i = 0; i < header_size; ++i) {
                space_ss << ' ';
            }
            space = space_ss.str();
        }

        // calculate message stride
        if (this->_max_length <= header_size)
            throw std::length_error("member variable '_max_length' must be larger than header size");

        size_t stride = this->_max_length - header_size;
        std::string message;
        while (std::getline(ss, message, '\n')) {
            is_empty_line = false;
            for (size_t i = 0; i < message.size(); i += stride) {
                if (!i) ns << header;
                else    ns << space;
                ns << message.substr(i, stride);
                ns << std::endl;
            }
        }

        if (is_empty_line)
            ns << std::endl << std::endl;

        if (level <= this->_log_level) {
            if (this->_file.is_open())
                this->_printFile(ns);
            else
                this->_printCout(ns);
        }

        if (level == LEVEL::FATAL)
            throw McrtLoggerException("Program terminated by bad input");

    }


    void McrtLogger::time() {
        std::time_t time = std::time({});
        char time_string[std::size("yyyy-mm-dd hh:mm:ss")];
        std::strftime(std::data(time_string), std::size(time_string), "%F %T", std::localtime(&time));
        std::stringstream ss;
        ss << " [" << time_string << "] " << std::endl;
        if (this->_file.is_open())
            this->_printFile(ss);
        else
            this->_printCout(ss);
    }


    bool setLogger(const std::string& file_name) {
        return McrtLogger::getInstance().open(file_name);
    }


    bool setLogger() {
        McrtLogger::getInstance().close();
        return true;
    }


    void setMaxLength(size_t length) {
        McrtLogger::getInstance().setMaxLength(length);
    }


    void setLevel(LEVEL level) {
        McrtLogger::getInstance().setLevel(level);
    }


    void fatal(std::stringstream& ss) {
        McrtLogger::getInstance().print(LEVEL::FATAL, ss);
    }


    void fatal(const std::string& message) {
        std::stringstream ss(message);
        fatal(ss);
    }


    void fatalTypeCasting(
        const std::string& field,
        const std::string& value
    ) {
        std::stringstream message;
        message << "Type casting error detected in '"
                << field << "', its value '"
                << value << "' has strange format";
        fatal(message);
    }


    void fatalOutOfRangeCeil(
        const std::string& field,
        const std::string& value,
        const std::string& ceil
    ) {
        std::stringstream message;
        message << "Value " << value 
                << " must be equal to or less than "
                << ceil << " in field '" << field << "'";
        fatal(message);
    }


    void fatalOutOfRangeFloor(
        const std::string& field,
        const std::string& value,
        const std::string& floor
    ) {
        std::stringstream message;
        message << "Value " << value
                << " must be equal to or larger than "
                << floor << " in field '" << field << "'";
        fatal(message);
    }


    void fatalFieldRequired(const std::string& field) {
        std::stringstream message;
        message << "Field '" << field << "' is necessary";
        fatal(message);
    }


    void fatalValueSize(
        const std::string& field,
        size_t required_size,
        size_t entered_size
    ) {
        std::stringstream message;
        message << "Field '" << field << "' requires "
                << required_size << " values, but "
                << entered_size << " values are entered";
        fatal(message);
    }


    void fatalInvalidNameFormat(const std::string& name) {
        std::stringstream message;
        message << "Name '" << name << "' has invalid format";
        fatal(message);
    }


    void fatalNameAlreadyExist(const std::string& name) {
        std::stringstream message;
        message << "Name '" << name << "' is already defined";
        fatal(message);
    }


    void fatalNameNotExist(const std::string& name) {
        std::stringstream message;
        message << "Name '" << name << "' is not defined";
        fatal(message);
    }


    void fatalFileNotExist(const std::string& name) {
        std::stringstream message;
        message << "Fail to open file '" << name << "'";
        fatal(message);
    }


    void fatalNecessary(const std::string& key) {
        std::stringstream message;
        message << "At least one '" << key << "'"
                << " card is necessary";
        fatal(message);
    }


    void fatalInsufficient(const std::string& key) {
        std::stringstream message;
        message << "Card '" << key
                << "' has insufficient parameters";
        fatal(message);
    }


    void warning(std::stringstream& ss) {
        McrtLogger::getInstance().print(LEVEL::WARN, ss);
    }


    void warning(const std::string& message) {
        std::stringstream ss(message);
        warning(ss);
    }


    void warningUseDefaultField(const std::string& field, const std::string& value) {
        std::stringstream message;
        message << "Field '" << field << "' is not found. "
                << "Use default value (" << value << ")";
        warning(message);
    }


    void line(size_t idx, std::stringstream& ss) {
        std::stringstream ss_new;
        ss_new << std::setw(6) << idx;
        ss_new << " ";
        ss_new << ss.str();
        McrtLogger::getInstance().print(LEVEL::LINE, ss_new);
    }


    void line(size_t idx, const std::string& message) {
        std::stringstream ss(message);
        line(idx, ss);
    }


    void info(std::stringstream& ss) {
        McrtLogger::getInstance().print(LEVEL::INFO, ss);
    }


    void info(const std::string& message) {
        std::stringstream ss(message);
        info(ss);
    }


    void print(std::stringstream& ss) {
        McrtLogger::getInstance().print(LEVEL::PRINT, ss);
    }


    void print(const std::string& message) {
        std::stringstream ss(message);
        print(ss);
    }


    void printName(const std::string& name, size_t max_length) {
        std::stringstream ss;
        ss << " [ " << std::setw(max_length) << mcutil::truncate(name, max_length) << " ]";
        print(ss);
    }


    void printVar(const std::string& var, const std::string& value, const std::string& unit) {
        std::stringstream ss;
        size_t sh;
        ss << "  " << var << " ";
        sh = ss.str().size();
        ss << std::setw(50 - sh) << value;
        if (!unit.empty())
            ss << " (" << unit << ")";
        print(ss);
    }


    void printVar(const std::string& var, double value, const std::string& unit) {
        std::stringstream ss;
        ss << value;
        printVar(var, ss.str(), unit);
    }


    void printVar(const std::string& var, size_t value, const std::string& unit) {
        std::stringstream ss;
        ss << value;
        printVar(var, ss.str(), unit);
    }


    void printVar(const std::string& var, int    value, const std::string& unit) {
        std::stringstream ss;
        ss << value;
        printVar(var, ss.str(), unit);
    }


    void printVar(const std::string& var, float  value, const std::string& unit) {
        std::stringstream ss;
        ss << value;
        printVar(var, ss.str(), unit);
    }


    void debug(std::stringstream& ss) {
        McrtLogger::getInstance().print(LEVEL::DEBUG, ss);
    }


    void debug(const std::string& message) {
        std::stringstream ss(message);
        debug(ss);
    }


    void time() {
        McrtLogger::getInstance().time();
    }


    FormattedTable::FormattedTable(const std::vector<size_t>& length, size_t offset, FORMAT_TYPE type, int digit) :
        _len(length), _lpos(0), _offset(offset), _type(type), _digit(digit) {
        this->_ss << std::string(_offset, ' ');
    }


    void FormattedTable::clear() {
        this->_lpos = 0x0u;
        this->_ss   = std::stringstream();
        this->_ss << std::string(_offset, ' ');
    }


    //template <>
    //FormattedTable& operator<< <std::string>(FormattedTable& cls, std::string _val) {
    //    size_t len = cls._len[cls._lpos];
    //    cls._lpos++;
    //    cls._ss << std::setw(len);
    //    cls._ss << _val;
    //    cls._ss << " ";
    //    return cls;
    //}
    

    FormattedTable& FormattedTable::operator<<(const std::string& _val) {
        size_t len = this->_len[this->_lpos];
        this->_lpos++;
        this->_ss << std::setw(len);
        this->_ss << mcutil::truncate(_val, len);
        this->_ss << " ";
        return *this;
    }


    FormattedTable& FormattedTable::operator<<(const char _val[]) {
        return this->operator<<(std::string(_val));
    }


}


namespace mcutil {

    std::string truncate(const std::string& str, size_t size) {
        std::string out;
        size_t str_size = str.size();
        if (str_size > size) {
            out = str.substr(0, size - 2);
            out += "..";
        }
        else
            out = str;
        return out;
    }

}
