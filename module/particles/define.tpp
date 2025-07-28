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