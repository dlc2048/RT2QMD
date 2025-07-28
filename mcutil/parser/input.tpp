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
 * @file    mcutil/parser/input.tpp
 * @brief   Input stream handler
 * @author  CM Lee
 * @date    05/23/2023
 */


#pragma once


namespace mcutil {


    template <typename T>
    std::deque<T> InputCardFactory<T>::readAll(Input& input, int max_inst) {
        std::stringstream ss;


        ArgumentCard card_definition = InputCardFactory<T>::_setCard();


        ss << "Search '" << card_definition.key() << "' cards ...";
        if (input.printSyntax())
            mclog::info(ss);
        else
            mclog::debug(ss);
        ss.str(""); ss.clear();

        ss << "Possible fields: ";
        for (const std::string& field :
            card_definition.fieldNameList()) {
            ss << field << ", ";
        }
        mclog::debug(ss);
        ss.str(""); ss.clear();

        if (input.printSyntax())
            card_definition.printHelp();

        input.init();
        std::deque<T> context_series;

        while (!input.eof()) {
            std::pair<std::string, ArgInput> card = input.nextCard();
            if (card.first == card_definition.key()) {
                input.printPreviousLine();
                card_definition.get(card.second);
                context_series.push_back(T(card.second));
                if (context_series.size() >= max_inst && max_inst >= 0)
                    break;
            }
        }
        return context_series;
    }


}