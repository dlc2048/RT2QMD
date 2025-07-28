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
 * @file    mcutil/prompt/prompt.cpp
 * @brief   CLI commands
 * @author  CM Lee
 * @date    05/23/2023
 */


#include "prompt.hpp"

namespace mcutil {

	FileTarget interpretCommandLine(int argc, char* argv[]) {

		FileTarget  target;
		target.level  = mclog::LEVEL::WARN;
		target.syntax = false;

		for (int i = 1; i < argc; ++i) {
			const std::string arg = argv[i];
			if (arg == "--help" || arg == "-h")
				printHelp(1);
			else if (arg == "--input" || arg == "-i") {
				if (i == argc - 1) {
					std::cout << "Option '" << argv[i] << "' needs following argument" << std::endl;
					printHelp(1);
				}
				target.input = argv[++i];
			}
			else if (arg == "--output" || arg == "-o") {
				if (i == argc - 1) {
					std::cout << "Option '" << argv[i] << "' needs following argument" << std::endl;
					printHelp(1);
				}
				target.output = argv[++i];
			}
			else if (arg == "--result" || arg == "-r") {
				if (i == argc - 1) {
					std::cout << "Option '" << argv[i] << "' needs following argument" << std::endl;
					printHelp(1);
				}
				target.result = argv[++i];
			}
			else if (arg == "--syntax" || arg == "-s")
				target.syntax = true;
			else if (arg == "--debug"  || arg == "-d")
				target.level  = mclog::LEVEL::DEBUG;
			else {
				std::cout << "Unknown option '" << argv[i] << "'" << std::endl;
				printHelp(1);
			}
		}

		if (target.input == "") {
			std::cout << "Input file must be specified" << std::endl;
			printHelp(1);
		}
		target.project = target.input.substr(0, target.input.find_last_of("."));

		if (target.result == "") {
			target.result = ".";
		}

		return target;
	}

	void printHelp(int err) {
		std::cout << "Parameters: --input   | -i <filename>     File for MC input         " << std::endl;
		std::cout << "            --output  | -o <filename>     File for MC output        " << std::endl;
		std::cout << "            --result  | -r <directory>    Where output & tally saved" << std::endl;
		std::cout << "            --debug   | -d                Print detail              " << std::endl;
		std::cout << "            --syntax  | -s                Print MC parameter info   " << std::endl;
		std::cout << "            --help    | -h                Print this message        " << std::endl;
		exit(err);
	}

}