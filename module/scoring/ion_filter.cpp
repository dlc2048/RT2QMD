
#include "ion_filter.hpp"


namespace mcutil {


    template <>
    ArgumentCard InputCardFactory<tally::IonFilter>::_setCard() {
        ArgumentCard arg_card("ION_FILTER");
        arg_card.insert<std::string>("tally", 1);
        arg_card.insert<int>("za");
        return arg_card;
    }


}


namespace tally {


    IonFilter::IonFilter(mcutil::ArgInput& args) {
        this->_where = args["tally"].cast<std::string>()[0];

        std::vector<int> za_list = args["za"].cast<int>();
        for (int za_this : za_list) {
            if (this->_za.find(za_this) != this->_za.end()) {
                std::stringstream ss;
                mclog::fatalListElementAlreadyExist(za_this);
            }
            this->_za.insert(za_this);
        }
    }


}