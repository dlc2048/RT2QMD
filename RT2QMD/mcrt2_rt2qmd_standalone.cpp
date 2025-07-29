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
 * @file    RT2QMD/mcrt2_rt2qmd_standalone.cpp
 * @brief   RT2QMD, GPU-based QMD evert generator. 
 *          This project is part of RT2, Ray-Tracing accelerated 
 *          Radiation Transport Monte Carlo code.
 * @author  CM Lee
 * @date    07/23/2025
 */


#include <string>
#include <exception>

#include "device/exception.h"

#include "prompt/prompt.hpp"
#include "parser/input.hpp"
#include "device/tuning.hpp"
#include "device/prng.hpp"

#include "transport/buffer.hpp"

#include "deexcitation/handler.hpp"
#include "deexcitation/deexcitation.cuh"

#include "qmd/reaction.hpp"
#include "qmd/buffer.cuh"
#include "qmd/config.hpp"

#include "nuc_secondary/secondary.cuh"

#include "genericion/auxiliary.hpp"

#include "auxiliary.hpp"
#include "aux_score.hpp"


int main(int argc, char* argv[]) {

	// Parse command line options
	mcutil::FileTarget target = mcutil::interpretCommandLine(argc, argv);
	std::filesystem::path res = target.result;
	if (!res.is_absolute())
		res = std::filesystem::current_path() / res;

	// make folder
	if (!std::filesystem::exists(res))
		std::filesystem::create_directory(res);

	// setup output
	std::filesystem::path out = std::filesystem::path(target.output);
	if (out.string() == "")
		mclog::setLogger();
	else {
		out = res / out;
		mclog::setLogger(out.string());
	}

	mclog::setLevel(target.level);

	// file startup time
	mclog::time();

	// setup input
	mcutil::Input input(target.input, target.syntax);

	// read gpu settings
	mcutil::DeviceController controller;
	{
		std::deque<mcutil::DeviceController> controller_lists 
			= mcutil::InputCardFactory<mcutil::DeviceController>::readAll(input);  // use the last card
		if (controller_lists.empty())
			mclog::fatalNecessary(mcutil::InputCardFactory<mcutil::DeviceController>::getCardDefinition().key());
		controller = controller_lists.back();
	}

	// turn off all transport except ion
	Define::Electron  ::getInstance().setActivation(false);
	Define::Photon    ::getInstance().setActivation(false);
	Define::Positron  ::getInstance().setActivation(false);
	Define::Neutron   ::getInstance().setActivation(false);

	Define::GenericIon::getInstance().setActivation(true);

	Define::IonInelastic::getInstance().setModeHigh(Define::ION_INELASTIC_METHOD_HIGH::ION_INELASTIC_HIGH_QMD);
	Define::IonInelastic::getInstance().setModeLow (Define::ION_INELASTIC_METHOD_LOW ::ION_INELASTIC_LOW_BME);

	// QMD settings
	mcutil::InputCardFactory<RT2QMD::Host::Config>::readAll(input);

	// Event generator
	auxiliary::EventGenerator event_generator(input);

	/*
	* Memory
	*/

	// set prng
	mcutil::RandState state_handler(
		{ (unsigned int)controller.block(),
		  (unsigned int)controller.thread() },
		(int)controller.seed(), 0
	);


	/*
	* Scoring
	*/

	// set scoring data
	// 
	// transportable ion table in RT2 (genion <-> tally / neutron)
	genion::IsoProjectileTable& ion_table = genion::IsoProjectileTable::getInstance();

	tally::DeviceMemoryHandlerMT tally_handler(
		input,
		target.project,
		controller,
		ion_table.listProjectileZA()
	);
	tally_handler.summary();

	bool tally_has_detector = event_generator.settings().hid();

	auxiliary::DummyDeviceMemoryHandler dummy_memory_handler;

	// file end time
	mclog::time();

	// END of input
	input.close();


	/*
	* Physics
	*/

	// De-excitation data
	std::shared_ptr<deexcitation::DeviceMemoryHandler> deexcitation_handler 
		= std::make_shared<deexcitation::DeviceMemoryHandler>();

	// QMD data
	std::shared_ptr<RT2QMD::DeviceMemoryHandler> qmd_handler = 
		std::make_shared<RT2QMD::DeviceMemoryHandler>(
			tally_has_detector,
			controller,
			event_generator.maxDimQMD()
		);

	/*
	* Buffer
	*/
	mcutil::DeviceBufferHandler buffer_handler(controller, event_generator.settings().hid());

	// MT YIELD
	tally_handler.prepareYieldHandle();

	mclog::time();
	mclog::info("All data are prepared!");

	// symbol link

	// tally
	// no device-side tally in RT2QMD

	// PRNG
	mclog::debug("Link PRNG device symbols ...");
	CUDA_CHECK(tally       ::setPrngHandle(state_handler.deviceHandle()));
	CUDA_CHECK(deexcitation::setPrngHandle(state_handler.deviceHandle()));
	CUDA_CHECK(Hadron      ::setPrngHandle(state_handler.deviceHandle()));
	CUDA_CHECK(RT2QMD      ::setPrngHandle(state_handler.deviceHandle()));

	// buffer system
	mclog::debug("Link buffer device symbols ...");
	CUDA_CHECK(deexcitation::setBufferHandle(buffer_handler.handle(), tally_has_detector));
	CUDA_CHECK(Hadron      ::setBufferHandle(buffer_handler.handle(), tally_has_detector));
	CUDA_CHECK(RT2QMD      ::setBufferHandle(buffer_handler.handle(), tally_has_detector));

	// transport
	mclog::debug("Link transport device symbols ...");
	CUDA_CHECK(Hadron::setMassTableHandle(Nucleus::MassTableHandler::getInstance().deviceptr()));

	// set auxiliary device symbol
	CUDA_CHECK(auxiliary::setBufferHandle(buffer_handler.handle(), buffer_handler.hasHid()));
	CUDA_CHECK(auxiliary::setQMDBuffer(qmd_handler->ptrProblemBuffer()));
	CUDA_CHECK(auxiliary::setPrngHandle(state_handler.deviceHandle()));
	
	// hadron dummy
	CUDA_CHECK(Hadron::setRegionMaterialTable(dummy_memory_handler.ptrDummyRegionMaterialTable()));
	CUDA_CHECK(Hadron::setTallyHandle(dummy_memory_handler.getDummyTallyHandle()));

	/*
	* Event Generator Loop
	*/

	int    block;
	int    thread = (int)controller.thread();

	mcutil::BUFFER_TYPE btype_target = static_cast<mcutil::BUFFER_TYPE>(event_generator.settings().bid());
	mcutil::BUFFER_TYPE btype;

	size_t next_stamp = (size_t)(0.05 * event_generator.settings().nps());

	mclog::time();
	std::stringstream target_bid_ss;
	target_bid_ss << "Module tester target intraction: " << mcutil::getBidName().find(btype_target)->second;
	mclog::info(target_bid_ss.str());
	std::stringstream tester_mode;
	tester_mode << "Module tester mode: " << (event_generator.settings().fullMode() ? "Full" : "Event");
	mclog::info(tester_mode.str());
	mclog::info("Start kernel iteration");

	// Fortran unformatted file (output), consider compatibility for the MCRT2 main interface 
	std::unique_ptr<mcutil::FortranOfstream> ofstream;
	if (event_generator.settings().writePS()) {
		ofstream = std::make_unique<mcutil::FortranOfstream>("phase_space.bin");
		// Generic ion infomations
		ofstream->write(ion_table.listProjectileZA());
		// Relaxation informations (shell binding energy [MeV]), dummy
		ofstream->write(std::vector<int>{0});
		ofstream->write(std::vector<int>{0});
		ofstream->write(std::vector<double>{0.0});
		ofstream->write(std::vector<double>{0.0});
		// INCL info
		bool use_incl = false;
		ofstream->write(std::vector<int>{use_incl});
		//if (use_incl) {
		//	ofstream->write(incl_handler->ZATableTarget());
		//	ofstream->write(incl_handler->ZATableProjectile());
		//}
		// Number of iterations
		ofstream->write(std::vector<int>{event_generator.settings().nIterKernel()});
	}

	for (int i = 0; i < event_generator.settings().nIterKernel(); ++i) {
		std::stringstream ss;
		ss << i << mcutil::ordinalSuffix(i + 1) << " iteration";
		mclog::info(ss.str());
		buffer_handler.clearAll();
		
		// init batch
		bool first_stage_flag = true;
		block = (int)controller.block();
		buffer_handler.setResidualStage((size_t)block);

		std::vector<geo::PhaseSpace> ps_transport;

		// reset QMD metadata
		qmd_handler->reset();

		while (true) {

			// QMD
			bool qmd_applied = false;
			qmd_applied = qmd_handler->launch();

			// Standard MC 
			if (!qmd_applied) {

				btype = buffer_handler.getBufferPriority();
				if (event_generator.settings().hid_now() >= event_generator.settings().nps()
					&& btype == mcutil::BUFFER_TYPE::SOURCE) {
					if (block > (int)controller.blockDecayLimit()) {
						// second stage
						block = (double)block * controller.blockDecayRate();
						block = std::max(block, 1);
						if (first_stage_flag) {
							mclog::time();
							mclog::info("End of main Monte Carlo stage");
							mclog::info("Start the residual stage");
						}
						mclog::info("Block is set to " + std::to_string(block));
						buffer_handler.setResidualStage((size_t)block);
						first_stage_flag = false;
						continue;
					}
					else {
						break;
					}
				}

				if (event_generator.settings().hid_now() >= next_stamp) {
					mclog::time();
					mclog::info("NPS " + std::to_string(next_stamp));
					next_stamp += (size_t)(0.05 * event_generator.settings().nps());
				}

				switch (btype) {
				case mcutil::BUFFER_TYPE::SOURCE:
					event_generator.sample(block, thread);
					break;
				case mcutil::BUFFER_TYPE::PHOTON:
				case mcutil::BUFFER_TYPE::ELECTRON:
				case mcutil::BUFFER_TYPE::POSITRON:
				case mcutil::BUFFER_TYPE::NEUTRON:
				case mcutil::BUFFER_TYPE::GNEUTRON:
				case mcutil::BUFFER_TYPE::GENION:
				case mcutil::BUFFER_TYPE::RELAXATION:
					tally_handler.appendYield(buffer_handler.deviceptr(), btype);
					if (event_generator.settings().writePS()) {
						std::vector<geo::PhaseSpace> ps_seg = buffer_handler.getPhaseSpace(btype, buffer_handler.hasHid());
						ps_transport.insert(ps_transport.begin(), ps_seg.begin(), ps_seg.end());
					}
					buffer_handler.clearTarget(btype);  // not supported in RT2QMD standalone, just flush
					// buffer_handler.pullVector(btype); 
					break;
				case mcutil::BUFFER_TYPE::RAYLEIGH:
				case mcutil::BUFFER_TYPE::PHOTO:
				case mcutil::BUFFER_TYPE::COMPTON:
				case mcutil::BUFFER_TYPE::PAIR:
				case mcutil::BUFFER_TYPE::EBREM:
				case mcutil::BUFFER_TYPE::EBREM_SP:
				case mcutil::BUFFER_TYPE::MOLLER:
				case mcutil::BUFFER_TYPE::PBREM:
				case mcutil::BUFFER_TYPE::PBREM_SP:
				case mcutil::BUFFER_TYPE::BHABHA:
				case mcutil::BUFFER_TYPE::ANNIHI:
				case mcutil::BUFFER_TYPE::DELTA:
				case mcutil::BUFFER_TYPE::ION_NUCLEAR:
				case mcutil::BUFFER_TYPE::BME:
				case mcutil::BUFFER_TYPE::CN_FORMATION:
				case mcutil::BUFFER_TYPE::ABRASION:
				{
					std::stringstream ss;
					ss << "Unsupported  buffer index " << btype;
					mclog::fatal(ss);
				}
					break;
				case mcutil::BUFFER_TYPE::NUC_SECONDARY:
					Hadron::secondaryStep(block, thread);
					break;
				case mcutil::BUFFER_TYPE::DEEXCITATION:
					deexcitation::deexcitationStep(block, thread);
					break;
				case mcutil::BUFFER_TYPE::PHOTON_EVAP:
					deexcitation::photon::continuumEvaporationStep(block, thread);
					break;
				case mcutil::BUFFER_TYPE::COMP_FISSION:
					deexcitation::fission::fissionStep(block, thread);
					break;
				default:
				{
					std::stringstream ss;
					ss << "Unsupported  buffer index " << btype;
					mclog::fatal(ss);
				}
				}

			}

			// CUDA_CHECK(cudaDeviceSynchronize());

			if (!event_generator.settings().fullMode() && (btype != mcutil::BUFFER_TYPE::SOURCE || qmd_applied))
				break;

			// CUDA_CHECK(cudaDeviceSynchronize());
			
			// buffer_handler.clearTransportBuffer(btype_target);
		}

		mclog::info("*** Launch History ***");
		mclog::printVar("NPS setup",          event_generator.settings().nps());
		mclog::printVar("Launched histories", event_generator.settings().hid_now());
		mclog::printVar("Total weight",       event_generator.settings().hid_now());

		buffer_handler.getBufferHistories();
		buffer_handler.result();

		tally_handler.pull(event_generator.settings().hid_now());
		tally_handler.write(res, i);

		// extract phase-space data
		if (event_generator.settings().writePS()) {
			std::vector<geo::PhaseSpace> ps_host = buffer_handler.getAllPhaseSpace(buffer_handler.hasHid());
			ps_host.insert(ps_host.end(), ps_transport.begin(), ps_transport.end());

			if (!ps_host.empty())
				ofstream->write(ps_host);
		}

		event_generator.settings().init();
		next_stamp = (size_t)(0.05 * event_generator.settings().nps());
	}

	mclog::time();
	mclog::info("End of iteration");


	return 0;
}