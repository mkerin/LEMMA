// Adapted from example/bgen_to_vcf.cpp by Gavin Band
// From project at http://bitbucket.org/gavinband/bgen/get/master.tar.gz

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
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
#include "bgen_parser.hpp"
#include "version.h"

// TODO: get --range to work off of .bgi files
// TODO: Sensible restructuring of interaction code
// TODO: --interaction option! Currently taking 1st column of covariates.
// TODO: Use high precision double for pval
// TODO: implement info filter
// TODO: tests for read_pheno, read_covar? Clarify if args for these are compulsory.
// TODO: copy argument_sanity()
// TODO: function to check that files exist

// Efficiency changes:
// 1) Use --range to edit query before reading bgen file
// 2) Is there an option to skip sample ids? 
//    If so we could fill n_samples at start
//    - read_covar() & read_pheno()
//    - then edit query with incomplete cases before reading bgen

// Notes:
// - Replace missing data in BGEN files with mean

// This example program reads data from a bgen file specified as the first argument
// and outputs it as a VCF file.
int main( int argc, char** argv ) {
	parameters p;

	try {
		parse_arguments(p, argc, argv);
		data Data( p.bgen_file );
		Data.params = p;
		Data.output_init();
		Data.n_samples = Data.bgenParser.number_of_samples();

		if(Data.params.incl_sids_file != "NULL"){
			Data.read_incl_sids();
		}

		if(p.mode_lm){
			// Start loading the good stuff
			Data.run();
		}

		if (p.mode_vcf){
			// Reading in from bgen file
			while (Data.read_bgen_chunk()) {
				std::cout << "Top of the bgen chunk while-loop" << std::endl;
				for (int jj = 0; jj < Data.n_var; jj++ ) {
					Data.outf << Data.chromosome[jj] << '\t'
						<< Data.position[jj] << '\t'
						<< Data.rsid[jj] << '\t' ;
					Data.outf << Data.alleles[jj][0] << '\t' ;
					for( std::size_t i = 1; i < Data.alleles[jj].size(); ++i ) {
						Data.outf << ( i > 1 ? "," : "" ) << Data.alleles[jj][i] ;
					}
					Data.outf << "\t.\t.\t.\tGP" ;

					for (int ii = 0; ii < Data.n_samples; ii++ ) {
						Data.outf << '\t' ;
						Data.outf << Data.G(ii,jj);
					}
					Data.outf << "\n" ;
				}
			}
		}

		return 0 ;
	}
	catch( genfile::bgen::BGenError const& e ) {
		std::cerr << "!! Uh-oh, error parsing bgen file.\n" ;
		return -1 ;
	}
}
