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


#include "parser.hpp"


namespace mcutil {


    std::deque<std::string> split(
        const std::string& line,
        const char         sep,
        const int          max_split
    ) {
        std::stringstream stream(line);
        std::deque<std::string> list;
        int n_split = 0;
        while (stream.good() && n_split != max_split) {
            std::string segment;
            if (sep == '\0')
                stream >> segment;
            else
                std::getline(stream, segment, sep);
            list.push_back(segment);
            n_split++;
        }
        if (stream.good()) {
            list.push_back(stream.str().substr(stream.tellg()));
        }
        return list;
    }


    std::deque<std::string> split(
        const std::string& line,
        const std::string& delimiter
    ) {
        std::deque<std::string> list;
        std::vector<int64_t> de_pos;
        de_pos.push_back(-1);
        for (int i = 0; i < (int)line.size(); ++i) {
            if ((int)delimiter.find(line[i]) == -1)
                continue;
            de_pos.push_back(i);
        }
        de_pos.push_back((int)line.size());
        for (int64_t i = 1; i < (int64_t)de_pos.size(); ++i) {
            int64_t p1 = de_pos[i - 1] + 1;
            int64_t p2 = de_pos[i] - 1;
            if (p1 <= p2)
                list.push_back(line.substr(p1, p2 - p1 + 1));
            if (p2 + 1 < de_pos.back())
                list.push_back(line.substr(p2 + 1, 1));
        }
        return list;
    }


    std::string join(const std::deque<std::string>& segs, const std::string sep) {
        std::string st;
        bool is_first = true;
        for (const auto& iter : segs) {
            if (!is_first)
                st += sep + iter;
            else {
                st += iter;
                is_first = false;
            }
        }
        return st;
    }


    std::string join(const std::vector<std::string>& segs, const std::string sep) {
        std::string st;
        bool is_first = true;
        for (const auto& iter : segs) {
            if (!is_first)
                st += sep + iter;
            else {
                st += iter;
                is_first = false;
            }
        }
        return st;
    }


    template <>
    std::vector<std::string> ArgumentContainer::cast() const {
        return this->_data_literal;
    }


    template <>
    std::vector<bool> ArgumentContainer::cast() const {
        std::vector<bool> formatted;
        for (const std::string& data : this->_data_literal) {
            std::stringstream converter(data);
            int value;
            converter >> value;
            if (converter.fail()) {  // literal
                std::string lower = data;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                if (lower == "true")
                    formatted.push_back(true);
                else if (lower == "false")
                    formatted.push_back(false);
                else
                    throw StringCastException(data);
            }
            else   // numeric
                formatted.push_back((bool)value);
        }
        return formatted;
    }


    template <>
    void Argument<std::string>::_rangeTest(const ArgumentContainer& container) const {
        // rangetest for string is not allowed
        mclog::fatal("Range test for std::string Argument is not allowed");
    }


    void ArgumentInterface::find(ArgInput& card) const {
        ArgInput::iterator iter = card.find(this->_field);
        if (iter == card.end()) {  // field is not found
            if (this->_is_required)
                mclog::fatalFieldRequired(this->_field);
        }
        // now field is included in container, let's cast datatype
        try {
            iter->second.cast<std::string>();
        }
        catch (StringCastException& e) {
            mclog::fatalTypeCasting(this->_field, std::string(e.what()));
        }
    }


    void ArgumentCard::insert(
        const std::string& field,
        std::shared_ptr<ArgumentInterface> arg
    ) {
        this->_args.insert({ field, arg });
    }


    std::vector<std::string> ArgumentCard::fieldNameList() const {
        std::vector<std::string> name_list;
        for (const ArgPair& arg : this->_args)
            name_list.push_back(arg.first);
        return name_list;
    }


    const std::shared_ptr<ArgumentInterface> ArgumentCard::field(const std::string& field) const {
        ArgList::const_iterator iter = this->_args.find(field);
        if (iter == this->_args.end())
            throw std::exception();
        else {
            return iter->second;
        }
    }


    void ArgumentCard::get(ArgInput& container) {
        // check container
        for (const auto& element : container) {
            std::string field = element.first;
            if (this->_args.find(field) == this->_args.end()) {
                std::stringstream ss;
                ss << "Unknown argument '--" << field << "'";
                mclog::fatal(ss);
            }
        }
        for (const ArgPair& arg : this->_args)
            arg.second->find(container);
    }


    void ArgumentCard::printHelp() {
        namespace fp = std::filesystem;
        std::string home = getMCRT2HomePath();
        if (home.empty())
            mclog::fatal("Environment variable 'MCRT2_HOME' is missing");
        fp::path file(home);
        file = file / SYNTAX_HOME / this->key();

        std::stringstream ss;

        ss << "Card '" << this->key() << "' field entry";
        mclog::info(ss);
        ss.str(""); ss.clear();

        std::ifstream data(file.string());
        if (!data.is_open()) {
            mclog::warning("Card syntax file not found");
            return;
        }

        char line[256];
        while (data.getline(line, 256)) {
            std::string sline(line);
            mclog::print(sline);
        }

        mclog::info("End of field entry");
    }


    bool isHasSpecialSymbol(const std::string& str) {
        for (std::string::const_iterator ch = str.begin();
            ch != str.end(); ++ch) {
            if (*ch > 47 && *ch < 58);        // digit
            else if (*ch > 64 && *ch < 91);   // capital alphabet
            else if (*ch > 96 && *ch < 123);  // small alphabet
            else if (*ch == 95);
            else return true;
        }
        return false;
    }


    bool isHasDigitHeader(const std::string& str) {
        if (!str.empty()) {
            if (str[0] > 47 && str[0] < 58) return true;
        }
        return false;
    }


    const std::string& ordinalSuffix(int n) {
        static const std::vector<std::string> suffixes = { "th", "st", "nd", "rd" };
        int ord = n % 100;
        if (ord / 10 == 1)
            ord = 0;
        ord = ord % 10;
        if (ord > 3)
            ord = 0;
        return suffixes[ord];
    }


}