// Adapted from example/bgen_to_vcf.cpp by Gavin Band
// From project at http://bitbucket.org/gavinband/bgen/get/master.tar.gz

#define EIGEN_USE_MKL_ALL

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <memory>
#include "parse_arguments.hpp"
#include "class.h"
#include "data.hpp"
#include "vbayes_x2.hpp"
#include "pve.hpp"


int main( int argc, char** argv ) {
	parameters p;

	std::cout << "============="<< std::endl;
	std::cout << "LEMMA v" << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << std::endl;
	std::cout << "=============" << std::endl << std::endl;

	auto start = std::chrono::system_clock::now();

	try {
		auto data_start = std::chrono::system_clock::now();
		parse_arguments(p, argc, argv);

		std::time_t start_time = std::chrono::system_clock::to_time_t(start);
		std::cout << "Starting analysis at " << std::ctime(&start_time) << std::endl;

		Data data( p );

		data.apply_filters();

		// Simple approach for the moment; don't bother about covariates etc

		data.read_non_genetic_data();

		// Also regresses out covariables if necessary
		data.standardise_non_genetic_data();

		data.read_full_bgen();

		// Variance components
		// Eigen::VectorXd eta = data.E.col(0).cast<double>();

		if(p.mode_vb){
			if (data.n_effects > 1) {
				data.calc_dxteex();
			}
			if (p.env_weights_file == "NULL" && p.init_weights_with_snpwise_scan) {
				data.calc_snpstats();
			}
			if (p.vb_init_file != "NULL") {
				data.read_alpha_mu();
			}

			auto data_end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_reading_data = data_end - data_start;


			// Pass data to VBayes object
			VBayesX2 VB(data);

			VB.check_inputs();
			// VB.output_init();

			// Run inference
			auto vb_start = std::chrono::system_clock::now();
			VB.run();
			auto vb_end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_vb = vb_end - vb_start;
			// VB.output_results();

			// eta = vp.eta.cast<double>();

			std::cout << std::endl << "Time expenditure:" << std::endl;
			std::cout << "Reading data: " << elapsed_reading_data.count() << " secs" << std::endl;
			std::cout << "VB inference: " << elapsed_vb.count() << " secs" << std::endl;
			std::cout << "runInnerLoop: " << VB.elapsed_innerLoop.count() << " secs" << std::endl;

			// Write time log to file
			boost_io::filtering_ostream outf_time;
			std::string ofile_map = VB.fstream_init(outf_time, "", "_time_elapsed");
			outf_time << "function time" << std::endl;
			outf_time << "read_data " << elapsed_reading_data.count() << std::endl;
			outf_time << "full_inference " << elapsed_vb.count() << std::endl;
			outf_time << "vb_outer_loop " << VB.elapsed_innerLoop.count() << std::endl;
		}

		if(p.mode_pve_est){

			// Random seed
			if(p.random_seed == -1){
				std::random_device rd;
				p.random_seed = rd();
			}
			std::cout << "Initialising random sample generator with seed " << p.random_seed << std::endl;

			// Eigen::VectorXd eta = data.E.col(0).cast<double>();
			Eigen::VectorXd Y = data.Y.cast<double>();

			PVE pve(p, data.G, Y);
			pve.run();

			// Write output
			pve.to_file(p.out_file);
		}
	}
	catch( genfile::bgen::BGenError const& e ) {
		std::cerr << "!! Uh-oh, error parsing bgen file.\n" ;
		return -1 ;
	}

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Analysis finished at " << std::ctime(&end_time);
	std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
	return 0;
}
