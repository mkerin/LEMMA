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
#include "genfile/bgen/View.hpp"
#include "bgen_parser.hpp"
#include "version.h"

// TODO: Sensible restructuring of interaction code
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

		// incl sample ids filter
		if(p.incl_sids_file != "NULL"){
			Data.read_incl_sids();
		}

		// range filter
		if (p.range){
			std::cout << "Selecting range..." << std::endl;
			genfile::bgen::IndexQuery::UniquePtr query = genfile::bgen::IndexQuery::create(p.bgi_file);
			genfile::bgen::IndexQuery::GenomicRange rr1(p.chr, p.start, p.end);
			query->include_range( rr1 ).initialise();
			Data.bgenView->set_query( query );
		}

		// incl rsids filter
		if(p.select_snps){
			Data.read_incl_rsids();
			std::cout << "Filtering SNPs by rsid..." << std::endl;
			genfile::bgen::IndexQuery::UniquePtr query = genfile::bgen::IndexQuery::create(p.bgi_file);
			query->include_rsids( Data.rsid_list ).initialise();
			Data.bgenView->set_query( query );
		}

		// Summary info
		Data.bgenView->summarise(std::cout);

		// if(p.mode_joint_model){
		// 	Data.calc_joint_model();
		// 	Data.output_results();
		// }

		if(p.mode_lm || p.mode_joint_model){
			// Start loading the good stuff
			Data.run();
		}

		if (p.mode_vcf){
			// Reading in from bgen file
			while (Data.read_bgen_chunk()) {
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
