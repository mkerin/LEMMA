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
#include "version.h"

void read_directory(const std::string& name, std::vector<std::string>& v);

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

		// filter - incl sample ids
		if(p.incl_sids_file != "NULL"){
			Data.read_incl_sids();
		}

		// filter - range
		if (p.range){
			std::cout << "Selecting range..." << std::endl;
			genfile::bgen::IndexQuery::UniquePtr query = genfile::bgen::IndexQuery::create(p.bgi_file);
			genfile::bgen::IndexQuery::GenomicRange rr1(p.chr, p.start, p.end);
			query->include_range( rr1 ).initialise();
			Data.bgenView->set_query( query );
		}

		// filter - incl rsids
		if(p.select_snps){
			Data.read_incl_rsids();
			std::cout << "Filtering SNPs by rsid..." << std::endl;
			genfile::bgen::IndexQuery::UniquePtr query = genfile::bgen::IndexQuery::create(p.bgi_file);
			query->include_rsids( Data.rsid_list ).initialise();
			Data.bgenView->set_query( query );
		}

		// filter - select single rsid
		if(p.select_rsid){
			std::sort(p.rsid.begin(), p.rsid.end());
			std::cout << "Filtering to rsids:" << std::endl;
			for (int kk = 0; kk < p.rsid.size(); kk++) std::cout << p.rsid[kk]<< std::endl;
			genfile::bgen::IndexQuery::UniquePtr query = genfile::bgen::IndexQuery::create(p.bgi_file);
			query->include_rsids( p.rsid ).initialise();
			Data.bgenView->set_query( query );
		}

		// Summary info
		Data.bgenView->summarise(std::cout);

		if(p.mode_joint_model){
			// // Step 1; Read in raw covariates and phenotypes
			// // - also makes a record of missing values
			// Data.read_covar();
			// Data.read_pheno();
			// 
			// // Step 2; Reduce raw covariates and phenotypes to complete cases
			// // - may change value of n_samples
			// // - will also skip these cases when reading bgen later
			// Data.reduce_to_complete_cases();
			// 
			// // Step 3; Center phenos, genotypes, normalise covars
			// Data.center_matrix( Y, n_pheno );
			// Data.center_matrix( W, n_covar );
			// Data.scale_matrix( W, n_covar, covar_names );
			// 
			// // Step 4; Regress covars out of phenos
			// Data.regress_covars();
			// 
			// // Step 5; Read all bgen data into single matrix
			// if(!p.bgen_wildcard){
			// 	// NB: maf/info filters apply during reading in.
			// 	// Hence will only know if n > p afterwards.
			// 	Data.params.chunk_size = Data.bgenView->number_of_variants();
			// 	Data.read_bgen_chunk();
			// } else {
			// 	// reading from several files
			// 	// std::vector<std::string> all_files;
			// 	// std::string dir, file_prefix;
			// 	// std::regex_search(str, m, re1);
			// 	// std::cout << " ECMA (depth first search) match: " << m[0] << '\n';
			// 	// read_directory("data/io_test/t5_bgen_wildcard", all_files);
			// 	// std::string wildcard = "chr*.bgen";
			// 	// std::regex dot_regex("\\.");
			// 	// std::string new_wildcard = std::regex_replace(wildcard, dot_regex, "\\.");
			// 	// std::cout << new_wildcard << std::endl;
			// 	// std::regex asterisk_regex("\\*");
			// 	// std::string new_wildcard2 = std::regex_replace(new_wildcard, asterisk_regex, "[[:digit:]]*");
			// 	// std::cout << new_wildcard2.c_str() << std::endl;
			// 	throw std::runtime_error("bgen wildcard functionality not yet implemented.")
			// }
		}

		// NB Might be better to write a different run() function for each mode.
		if(p.mode_lm){
			std::cout << "Running SNPwise regression" << std::endl;
			Data.run();
		}

		if (p.mode_lm2){
			std::cout << "Running the full dependency model:" << std::endl;
			std::cout << "Y = C + E + G + CxE + GxC + GxE" << std::endl;
			std::cout << "WARNING: this is likely to be slow for a large number of SNPs" << std::endl;
			Data.run();
		}

		if (p.mode_joint_model){
			Data.run();
		}

		if (p.mode_vb){
			// Simple approach for the moment; don't bother about covariates etc

			// Read in phenotypes
			Data.read_pheno();

			// Read in all genetic data
			Data.params.chunk_size = Data.bgenView->number_of_variants();
			Data.read_bgen_chunk();

			// Read in grids for importance sampling
			Data.read_grids();

			// Read starting point for VB approximation if provided
			if(p.alpha_file != "NULL" && p.mu_file != "NULL"){
				Data.read_alpha_mu();
			}

			// Pass data to VBayes object
			vbayes VB(Data);
			VB.check_inputs();

			// Run inference
			std::cout << "Starting to run variational inference" << std::endl;
			VB.run();

			VB.write_to_file( p.out_file );
		}

		if (p.mode_vcf){
			Data.output_init();
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


void read_directory(const std::string& name, std::vector<std::string>& v) {
	DIR* dirp = opendir(name.c_str());
	struct dirent * dp;
	while ((dp = readdir(dirp)) != NULL) {
		v.push_back(dp->d_name);
	}
	closedir(dirp);
}
