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
#include "genfile/bgen/bgen.hpp"
#include "genfile/bgen/View.hpp"
#include "bgen_parser.hpp"
#include "vbayes.hpp"
#include "vbayes_x2.hpp"
#include "version.h"

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
		parse_arguments(p, argc, argv);
		Data data( p );

		// filter - incl sample ids
		// filter - range
		// filter - incl rsids
		// filter - select single rsid
		data.apply_filters();

		// Summary info
		data.bgenView->summarise(std::cout);

		if (p.mode_vb){
			// Simple approach for the moment; don't bother about covariates etc

			data.read_non_genetic_data();
			data.standardise_non_genetic_data();

			if(p.covar_file != "NULL"){
				data.regress_out_covars();
			}
			data.read_full_bgen();

			// Pass data to VBayes object
			if(p.interaction_analysis){
				VBayesX2 VB(data);

				VB.check_inputs();
				// VB.output_init();

				// Run inference
				VB.run();
				// VB.output_results();
			} else {
				vbayes VB(data);

				VB.check_inputs();

				// Run inference
				std::cout << "Starting to run variational inference" << std::endl;
				VB.run();
				VB.write_to_file( p.out_file );
			}
		}

		if (p.mode_vcf){
			throw std::runtime_error("Gutted from programme. See other branch.");
		}

		if(p.mode_joint_model){
			throw std::runtime_error("Gutted from programme. See other branch.");
		}

		if(p.mode_lm){
			throw std::runtime_error("Gutted from programme. See other branch.");
		}

		if (p.mode_lm2){
			throw std::runtime_error("Gutted from programme. See other branch.");
		}

		if (p.mode_joint_model){
			throw std::runtime_error("Gutted from programme. See other branch.");
		}

		return 0 ;
	}
	catch( genfile::bgen::BGenError const& e ) {
		std::cerr << "!! Uh-oh, error parsing bgen file.\n" ;
		return -1 ;
	}
}
