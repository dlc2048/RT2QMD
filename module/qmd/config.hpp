/**
 * @file    module/qmd/config.hpp
 * @brief   QMD config handler
 * @author  CM Lee
 * @date    02/15/2024
 */

#pragma once

#include "mclog/logger.hpp"
#include "parser/input.hpp"


namespace RT2QMD {


    enum class NN_SCATTERING_TYPE {
        NN_SCATTERING_GEANT4 = 0,
        NN_SCATTERING_INCL   = 1
    };


    namespace Host {


        //! @brief QMD global configurations
        class Config : public Singleton<Config> {
            friend class Singleton<Config>;
        private:
            NN_SCATTERING_TYPE _scattering_type;
            bool _measure_time;    // Measure the execution time of the initialization per block
            int  _timer_size;      // Maximum size of timer
            bool _do_dump_action;  // Phase-space dump for the leading model
            int  _dump_size;       // Maximum size of phase-space dump


            Config();  // singleton


        public:


            Config(mcutil::ArgInput& args);


            bool measureTime() {
                return this->_measure_time;
            }


            int timerSize() {
                return this->_timer_size;
            }


            bool usingINCLNNScattering() {
                return this->_scattering_type == NN_SCATTERING_TYPE::NN_SCATTERING_INCL;
            }


            bool doDumpAction() {
                return this->_do_dump_action;
            }


            int dumpSize() {
                return this->_dump_size;
            }


        };


    }
}