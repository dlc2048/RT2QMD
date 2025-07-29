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
 * @file    module/particles/define.tpp
 * @brief   RT2 particle definitions
 * @author  CM Lee
 * @date    04/08/2024
 */


#pragma once


namespace Define {


    template <typename T>
    void ParticleInterface<T>::_readHeader(mcutil::ArgInput& args) {
        // T::_ARGCARD.get(args);
        this->_library   = args["library"].cast<std::string>()[0];
        this->_activated = args["activate"].cast<bool>()[0];
        this->_t_cutoff  = args["transport_cutoff"].cast<double>()[0];
        this->_p_cutoff  = args["production_cutoff"].cast<double>()[0];
    }


    template <typename T>
    int ParticleInterface<T>::pid() {
        return this->_pid;
    }


    template <typename T>
    bool ParticleInterface<T>::activated() {
        return this->_activated;
    }


    template <typename T>
    double ParticleInterface<T>::transportCutoff() {
        return this->_t_cutoff;
    }


    template <typename T>
    double ParticleInterface<T>::productionCutoff() {
        return this->_p_cutoff;
    }


    template <typename T>
    double ParticleInterface<T>::transportCeil() {
        return this->_t_ceil;
    }


    template <typename T>
    void ParticleInterface<T>::setTransportCeil(double ceil) {
        this->_t_ceil = ceil;
    }


    template <typename T>
    void ParticleInterface<T>::setTransportCutoff(double floor) {
        this->_t_cutoff = floor;
    }


    template <typename T>
    void ParticleInterface<T>::setProductionCutoff(double floor) {
        this->_p_cutoff = floor;
    }


    template <typename T>
    void ParticleInterface<T>::setActivation(bool option) {
        this->_activated = option;
    }


    template <typename T>
    std::string ParticleInterface<T>::library() {
        return this->_library;
    }


    template <typename T>
    mcutil::ArgumentCard ParticleInterface<T>::controlCard() {
        return mcutil::InputCardFactory<T>::getCardDefinition();
    }



}