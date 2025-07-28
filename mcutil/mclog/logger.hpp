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
 * @file    mcutil/mclog/logger.hpp
 * @brief   RT2 log system
 * @author  CM Lee
 * @date    05/23/2023
 */


#pragma once

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <exception>

#include "singleton/singleton.hpp"


namespace mclog {


	static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
	{
		std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
			<< message << "\n";
	}


	/**
	* @brief Log level
	*/
	enum class LEVEL {
		OFF  ,
		FATAL,
		PRINT,
		INFO ,
		LINE ,
		WARN ,
		DEBUG
	};


	class McrtLoggerException : public std::exception {
	protected:
		std::string _msg;
	public:
		explicit McrtLoggerException(const char* message)
			: _msg(message) {}
		explicit McrtLoggerException(const std::string& message)
			: _msg(message) {}
		virtual ~McrtLoggerException() noexcept {}
		virtual const char* what() const noexcept {
			return _msg.c_str();
		}
	};


	/**
	* @brief RT2 log system, which shared by all modules
	*/
	class McrtLogger : public Singleton<McrtLogger> {
		friend class Singleton<McrtLogger>;
	private:
		std::ofstream                _file;        //!< @brief Output stream file
		size_t                       _max_length;  //!< @brief Max length in one line
		std::map<LEVEL, std::string> _header;      //!< @brief Log level header
		LEVEL                        _log_level;   //!< @brief Current level


		McrtLogger();


		~McrtLogger();


		/**
		* @brief Print stringstream on terminal
		* @param ss Line segment
		*/
		void _printCout(std::stringstream& ss);


		/**
		* @brief Print stringstream on file stream
		* @param ss Line segment
		*/
		void _printFile(std::stringstream& ss);
		
	public:


		/**
		* @brief Set maximum length of single line
		* @param length maximum length of line
		*/
		void setMaxLength(size_t length);


		/**
		* @brief Set minimum level of log.
		*        Message under this level will be ignored
		* @param level Minimum log level
		*/
		void setLevel(LEVEL level);


		/**
		* @brief Open log stream
		* @param file_name logfile name
		*/
		bool open(const std::string& file_name);


		/**
		* @brief Close log stream
		*/
		void close();


		/**
		* @brief Print log on terminal or stream
		* @param level Log level
		* @param ss    Log message
		*/
		void print(LEVEL level, std::stringstream& ss);


		/**
		* @brief Print local time on terminal or stream
		*/
		void time();
	};


	/**
	* @brief Set McrtLogger singleton instance to targeting filestream
	* @param file_name Name of filestream
	* 
	* @return true if success, false elsewhere
	*/
	bool setLogger(const std::string& file_name);


	/**
	* @brief Set McrtLogger singleton instance to targeting terminal
	* 
	* @return true if success, false elsewhere
	*/
	bool setLogger();


	/**
	* @brief Set maximum line length of McrtLogger singleton instance
	*/
	void setMaxLength(size_t length);


	/**
	* @brief Set minimum level of McrtLogger singleton instance
	*/
	void setLevel(LEVEL level);


	/* Fatal families */ 


	void fatal(std::stringstream& ss);


	void fatal(const std::string& message);


	void fatalTypeCasting(
		const std::string& field, 
		const std::string& value
	);


	void fatalOutOfRangeCeil(
		const std::string& field,
		const std::string& value,
		const std::string& ceil
	);


	void fatalOutOfRangeFloor(
		const std::string& field,
		const std::string& value,
		const std::string& floor
	);


	void fatalFieldRequired(const std::string& field);


	void fatalValueSize(
		const std::string& field,
		size_t required_size,
		size_t entered_size
	);


	void fatalInvalidNameFormat(const std::string& name);


	void fatalNameAlreadyExist(const std::string& name);


	template <typename T>
	void fatalListElementAlreadyExist(const T& element);


	void fatalNameNotExist(const std::string& name);


	void fatalFileNotExist(const std::string& name);


	void fatalNecessary(const std::string& key);


	void fatalInsufficient(const std::string& key);


	/* Warining families */


	void warning(std::stringstream& ss);


	void warning(const std::string& message);


	//template <typename T>
	//void warningUseDefaultField(const std::string& field, T value);


	void warningUseDefaultField(const std::string& field, const std::string& value);

	
	/* Line families */


	void line(size_t idx, std::stringstream& ss);


	void line(size_t idx, const std::string& message);
	

	/* Info families */


	void info(std::stringstream& ss);


	void info(const std::string& message);


	/* Print families */


	void print(std::stringstream& ss);


	void print(const std::string& message);


	void printName(const std::string& name, size_t max_length=16);


	void printVar(const std::string& var, const std::string& value, const std::string& unit = "");


	void printVar(const std::string& var, double value, const std::string& unit = "");


	void printVar(const std::string& var, size_t value, const std::string& unit = "");


	void printVar(const std::string& var, int    value, const std::string& unit = "");


	void printVar(const std::string& var, float  value, const std::string& unit = "");


	/* Debug families */


	void debug(std::stringstream& ss);


	void debug(const std::string& message);
	

	/**
	* @brief Print local time
	*/
	void time();


	enum class FORMAT_TYPE {
		AUTO,
		NUMERIC,
		SCIENTIFIC
	};


	/**
	* @brief Table helper
	*/
	class FormattedTable  {
	private:
		std::vector<size_t> _len;
		size_t              _lpos;
		size_t              _offset;
		FORMAT_TYPE         _type;
		int                 _digit;
		std::stringstream   _ss;
	public:


		FormattedTable(const std::vector<size_t>& length, size_t offset=0x0u, FORMAT_TYPE type=FORMAT_TYPE::AUTO, int digit=-1);


		std::string str() const { return this->_ss.str(); }


		void clear();


		FormattedTable& operator<<(const std::string& _val);


		FormattedTable& operator<<(const char _val[]);


		template <typename T, typename std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
		FormattedTable& operator<<(T _val);


		template <typename T, typename std::enable_if_t<std::is_integral<T>::value>* = nullptr>
		FormattedTable& operator<<(T _val);



	};


	//template <typename T, typename Enable = void>
	//FormattedTable& operator<<(FormattedTable& cls, T& _val);


	//template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
	//FormattedTable& operator<<(FormattedTable& cls, T _val);


	//template <typename T, typename std::enable_if_t<std::is_integral_v<T>, bool> = true>
	//FormattedTable& operator<<(FormattedTable& cls, T _val);


}


namespace mcutil {


	std::string truncate(const std::string& str, size_t size);


}


#include "logger.tpp"
