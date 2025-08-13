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
 * @file    module/transport/buffer_struct.hpp
 * @brief   Enums related to the transport system
 * @author  CM Lee
 * @date    05/23/2023
 */


#pragma once

#include <map>
#include <string>

#include "particles/define_struct.hpp"


namespace mcutil {

    
	enum BUFFER_TYPE {
		// virtual
		SOURCE,
		QMD,       // for the test
		// Optix transport
		ELECTRON,
		PHOTON,
		POSITRON,
		NEUTRON,   // Neutron < 20 MeV
		GNEUTRON,  // Generic neutron
		GENION,
		// interactions
		RELAXATION,
		RAYLEIGH,
		PHOTO,
		COMPTON,
		PAIR,
		EBREM,
		EBREM_SP,
		MOLLER,
		PBREM,
		PBREM_SP,
		BHABHA,
		ANNIHI,
		NEU_SECONDARY,   // neutron secondary
		DELTA,           // generic ion delta
		ION_NUCLEAR,     // Nucleus-nucleus inelastic (determine isotope target & rejection)
		// ion-nuclear models
		BME,
		CN_FORMATION,     // compound nucleus formation (INCL)
		ABRASION,
		NUC_SECONDARY,   // nuclear model secondaries
		DEEXCITATION,    // de-excitation branching
		PHOTON_EVAP,     // photon evaporation
		COMP_FISSION,    // competitive fission
		// buffer size indicator
		EOB
	};


	static const std::map<int, BUFFER_TYPE>& getPidHash() {
		static const std::map<int, BUFFER_TYPE> PID_HASH = {
			{ Define::PID::PID_ELECTRON, BUFFER_TYPE::ELECTRON    },
			{ Define::PID::PID_PHOTON,   BUFFER_TYPE::PHOTON      },
			{ Define::PID::PID_POSITRON, BUFFER_TYPE::POSITRON    },
			{ Define::PID::PID_NEUTRON,  BUFFER_TYPE::GNEUTRON    },
			{ Define::PID::PID_GENION,   BUFFER_TYPE::GENION      },
			{ Define::PID::PID_VACANCY,  BUFFER_TYPE::RELAXATION  },
			{ Define::PID::PID_IONNUC,   BUFFER_TYPE::ION_NUCLEAR },
		};
		return PID_HASH;
	}
    

	static const std::map<int, std::string>& getPidName() {
		static const std::map<int, std::string> PID_NAME = {
			{ Define::PID::PID_ELECTRON, "Electron"       },
			{ Define::PID::PID_PHOTON,   "Photon"         },
			{ Define::PID::PID_POSITRON, "Positron"       },
			{ Define::PID::PID_NEUTRON,  "Neutron"        },
			{ Define::PID::PID_GENION,   "Generic Ion"    },
			{ Define::PID::PID_VACANCY,  "Vacancy"        },
			{ Define::PID::PID_IONNUC,   "Heavy Ion Local"},
		};
		return PID_NAME;
	}


	static const std::map<BUFFER_TYPE, std::string>& getBidName() {
		static std::map<BUFFER_TYPE, std::string> BID_NAME = {
			{BUFFER_TYPE::SOURCE,        "Monte Carlo Primary"           },
			{BUFFER_TYPE::QMD,           "Quantum Molecular Dynamics"    },
			{BUFFER_TYPE::ELECTRON,      "Electron Transport"            },
			{BUFFER_TYPE::PHOTON,        "Photon Transport"              },
			{BUFFER_TYPE::POSITRON,      "Positron Transport"            },
			{BUFFER_TYPE::NEUTRON,       "Low Energy Neutron Transport"  },
			{BUFFER_TYPE::GNEUTRON,      "Generic Neutron Transport"     },
			{BUFFER_TYPE::GENION,        "Generic Ion Transport"         },
			{BUFFER_TYPE::RELAXATION,    "Atomic Relaxation"             },
			{BUFFER_TYPE::RAYLEIGH,      "Rayleigh Scattering"           },
			{BUFFER_TYPE::PHOTO,         "Photoelectric Effect"          },
			{BUFFER_TYPE::COMPTON,       "Compton Scattering"            },
			{BUFFER_TYPE::PAIR,          "Pair production"               },
			{BUFFER_TYPE::EBREM,         "e- Bremsstrahlung"             },
			{BUFFER_TYPE::EBREM_SP,      "e- Bremsstrahlung, split"      },
			{BUFFER_TYPE::MOLLER,        "Moller Scattering"             },
			{BUFFER_TYPE::PBREM,         "e+ Bremsstrahlung"             },
			{BUFFER_TYPE::PBREM_SP,      "e+ Bremsstrahlung, split"      },
			{BUFFER_TYPE::BHABHA,        "Bhabha Scattering"             },
			{BUFFER_TYPE::ANNIHI,        "e+e- Annihilation"             },
			{BUFFER_TYPE::NEU_SECONDARY, "Low-energy Neutron Reaction"   },
			{BUFFER_TYPE::DELTA,         "Generic ion delta-ray"         },
			{BUFFER_TYPE::ION_NUCLEAR,   "Ion-Nuclear Reaction"          },
			{BUFFER_TYPE::BME,           "Boltzmann Master Equation"     },
			{BUFFER_TYPE::CN_FORMATION,  "Compound-Nucleus Formation"    },
			{BUFFER_TYPE::ABRASION,      "Wilson-Abrasion"               },
			{BUFFER_TYPE::NUC_SECONDARY, "Frag Secondary Particles"      },
			{BUFFER_TYPE::DEEXCITATION,  "Nuclear De-excitation"         },
			{BUFFER_TYPE::PHOTON_EVAP,   "Photon Evaporation"            },
			{BUFFER_TYPE::COMP_FISSION,  "Competitive Fission"           },
		};
		return BID_NAME;
	}


}
