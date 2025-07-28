#pragma once

#include "tally_interface.hpp"


namespace tally {


    class IonFilter {
    protected:
        std::string   _where;  // target tally
        std::set<int> _za;      // target genion ZA lists


    public:


        IonFilter(mcutil::ArgInput& args);


        const std::string& target() const { return this->_where; }


        const std::set<int>& za() const { return this->_za; }


    };


}