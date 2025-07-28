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
 * @file    mcutil/parser/input.cpp
 * @brief   Input stream handler
 * @author  CM Lee
 * @date    05/23/2023
 */


#include "input.hpp"


namespace mcutil {


    void Input::_setMacro() {
        this->init();
        this->_macro.clear();
        mclog::debug("Set macro statements definition ...");
        while (!this->eof()) {
            std::deque<std::string> list = this->nextLine();
            if (list.empty()) continue;
            std::string key = list[0];
            if (key.front() == '$' && key.back() == '$') {
                if (list.size() < 2) {
                    std::stringstream ss;
                    ss << "Macro statement '" << key << "' needs definition";
                    mclog::fatal(ss);
                }
                Macro::iterator iter = this->_macro.find(key);
                if (iter != this->_macro.end()) {
                    std::stringstream ss;
                    ss << "Macro statement '" << key << "' is already defined";
                    mclog::fatal(ss);
                }
                std::stringstream ss;
                list.pop_front();
                std::string mac_def = mcutil::join(list);
                ss << key << " = " << mac_def;
                mclog::debug(ss);
                this->_macro.insert({ key.substr(1, key.size() - 2), mac_def });
            }
        }
    }


    Input::Input(const std::string& file_name, bool psyntax) :
        _pos_current(0), _pos_last(0), _psyntax(psyntax) {
        this->_stream.open(file_name);
        if (!this->_stream) {
            std::stringstream ss;
            ss << "Input file '" << file_name << "' is not found";
            mclog::fatal(ss);
        }
        else {
            std::stringstream ss;
            ss << "Read input file stream '" << file_name << "'";
            mclog::info(ss);
        }
        this->_setMacro();
        this->init();
    }


    void Input::seek(size_t line) {
        this->_stream.clear();
        this->_stream.seekg(std::ios::beg);
        for (uint32_t i = 0; i < line; ++i) {
            this->_stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        if (!this->_stream)
            throw std::runtime_error("Input::seek(line) out of range");
        this->_pos_current = line;
    }


    void Input::close() {
        if (this->_stream.is_open()) {
            mclog::info("Close input file stream");
            this->_stream.close();
        }
    }


    char Input::get() {
        char c = 0;
        this->_stream.get(c);
        return c;
    }


    std::deque<std::string> Input::nextLine() {
        std::deque<std::string> list;
        std::string line;
        bool is_header_found  = false;
        bool is_newline_found = false;

        while ((!is_header_found || is_newline_found) && !this->eof()) {
            std::deque<std::string> seg;
            std::getline(this->_stream, line);
            // replace macro statement
            for (const std::pair<std::string, std::string>& macro : this->_macro) {
                std::regex macro_regex("\\$" + macro.first + "\\$");
                line = std::regex_replace(line, macro_regex, macro.second);
            }
            ++this->_pos_current;
            seg = mcutil::split(line, '#', 1);  // remove annotation
            if (seg.empty()) continue;

            line = seg.front();
            seg = mcutil::split(line, '\\', 1);  // find newline character
            if (seg.empty()) continue;

            is_newline_found = (seg.size() == 1) ? false : true;
            seg = mcutil::split(seg.front());

            // merge segments to list
            for (std::string& element : seg) {
                if (element.empty())
                    continue;
                is_header_found = true;
                list.push_back(element);
            }
            if (!is_newline_found)
                this->_pos_last = this->_pos_current;  // save the current line position
        }
        this->_line_last = mcutil::join(list);
        return list;
    }


    std::pair<std::string, ArgInput> Input::nextCard() {
        std::deque<std::string> list = this->nextLine();
        if (list.empty())
            return { "", ArgInput() };

        ArgInput          args;
        ArgumentContainer container;
        
        std::string key   = list[0];
        std::string field = "";
        for (size_t i = 1; i < list.size(); ++i) {
            std::smatch match;
            if (std::regex_match(list[i], match, FIELD_PATTERN)) {  // new field
                if (!field.empty()) {
                    if (args.find(field) != args.end()) {
                        std::stringstream ss;
                        ss << "Field '" << field << "' is duplicated";
                        mclog::fatal(ss);
                    }
                    args.insert({ field, container });
                }
                field = match[1];
                container.clear();
            }
            else if (!field.empty()) {
                container.push_back(list[i]);
            }
        }
        // last argument
        if (!field.empty()) {
            if (args.find(field) != args.end()) {
                std::stringstream ss;
                ss << "Field '" << field << "' is duplicated";
                mclog::fatal(ss);
            }
            args.insert({ field, container });
        }
        return { key, args };
    }



}