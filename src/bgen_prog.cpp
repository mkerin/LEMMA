// Adapted from example/bgen_to_vcf.cpp by Gavin Band
// From project at http://bitbucket.org/gavinband/bgen/get/master.tar.gz

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <memory>
#include "parse_arguments.hpp"
#include "class.h"
#include "data.hpp"
#include "vbayes_x2.hpp"

void read_directory(const std::string& name, std::vector<std::string>& v);

// TODO: Sensible restructuring of interaction code
// TODO: Should not require grid_probs
// TODO: Use high precision double for pval
// TODO: implement tests for info filter
// TODO: tests for read_pheno, read_covar? Clarify if args for these are compulsory.
// TODO: copy argument_sanity()

// Efficiency changes:
// 1) Use --range to edit query before reading bgen file
// 2) Is there an option to skip sample ids?
//    If so we could fill n_samples at start
//    - read_covar() & read_pheno()
//    - then edit query with incomplete cases before reading bgen

// This example program reads data from a bgen file specified as the first argument
// and outputs it as a VCF file.
int main( int argc, char** argv ) {
	parameters p;

	try {
		auto data_start = std::chrono::system_clock::now();
		parse_arguments(p, argc, argv);
		Data data( p );

		data.apply_filters();

		// Summary info
		data.bgenView->summarise(std::cout);

		// Simple approach for the moment; don't bother about covariates etc

		data.read_non_genetic_data();

		// Also regresses out covariables if necessary
		data.standardise_non_genetic_data();

		data.read_full_bgen();
		data.calc_dxteex();
		if(p.env_weights_file == "NULL" && p.init_weights_with_snpwise_scan) {
			data.calc_snpstats();
		}
		if(p.vb_init_file != "NULL"){
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

		std::cout << std::endl << "Time expenditure:" << std::endl;
		std::cout << "Reading data: " << elapsed_reading_data.count() << " secs" << std::endl;
		std::cout << "VB inference: " << elapsed_vb.count() << " secs" << std::endl;
		std::cout << "runOuterLoop: " << VB.elapsed_outerLoop.count() << " secs" << std::endl;

		// Write time log to file
		boost_io::filtering_ostream outf_time;
		std::string ofile_map   = VB.fstream_init(outf_time, "", "_time_elapsed");
		outf_time << "function time" << std::endl;
		outf_time << "read_data " << elapsed_reading_data.count() << std::endl;
		outf_time << "full_inference " << elapsed_vb.count() << std::endl;
		outf_time << "vb_outer_loop " << VB.elapsed_outerLoop.count() << std::endl;

		return 0 ;
	}
	catch( genfile::bgen::BGenError const& e ) {
		std::cerr << "!! Uh-oh, error parsing bgen file.\n" ;
		return -1 ;
	}
}
