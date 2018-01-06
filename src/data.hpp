// File of Data class for use with src/bgen_prog.cpp
#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <vector>
#include <string>
#include "class.h"
#include "tools/eigen3.3/Dense"
#include "tools/eigen3.3/Sparse"
#include "tools/eigen3.3/Eigenvalues"

#include "bgen_parser.hpp"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>

namespace boost_io = boost::iostreams;

class data 
{
	public :
	parameters params;

	std::vector< std::string > chromosome, rsid, SNPID;
	std::vector< uint32_t > position;
	std::vector< std::vector< std::string > > alleles;
	
	int n_pheno; // number of phenotypes
	int n_covar; // number of covariates
	int n_samples; // number of samples
	long int n_snps; // number of snps
	bool bgen_pass;
	int G_ncol;

	std::vector<double> info;
	std::vector<double> maf;

	Eigen::MatrixXd G; // genotype matrix --
	BgenParser bgenParser; // hopefully initialised appropriately from
                           // initialisation list in Data constructor

	boost_io::filtering_ostream outf;
	
	// constructors/destructors	
	data( std::string filename ) : bgenParser( filename ){
		bgen_pass = true;
	}
	
	~data() {
	}

	void output_init() {
		// open output file
		std::string ofile, gz_str = ".gz";
		std::cout << "Opening output files ..." << std::endl;

		ofile = params.out_file;
		outf.push(boost_io::file_sink(ofile.c_str()));
		// outf << "chr rsid pos a_0 a_1 af info";
	}

	bool read_bgen_chunk() {
		// Assumed that:
		// - commandline args parsed and passed to params
		// - bgenParser initialised with correct filename
		std::cout << "Entering read_bgen_chunk()" << std::endl;
		// Exit function if last call ran out of variants to read in.
		if (!bgen_pass) return false;

		// Temporary variables to store info from read_variant()
		std::string my_chr ;
		uint32_t my_pos ;
		std::string my_rsid ;
		std::vector< std::string > my_alleles ;
		std::vector< std::vector< double > > probs ;

		double d1, af, x, dosage;

		// Wipe variant info from last chunk
		maf.clear();
		rsid.clear();
		chromosome.clear();
		position.clear();
		alleles.clear();

		// Resize genotype matrix
		G.resize(n_samples, params.chunk_size);

		int jj = 0;
		while ( jj < params.chunk_size && bgen_pass ) {
			bgen_pass = bgenParser.read_variant( &my_chr, &my_pos, &my_rsid, &my_alleles );
			if (!bgen_pass) break;
			assert( my_alleles.size() > 0 );

			// range filter
			if (params.range && (my_pos < params.start || my_pos > params.end)){
				bgenParser.ignore_probs();
				continue;
			}

			// maf + info filters
			d1 = 0.0;
			for( std::size_t i = 0; i < probs.size(); ++i ) {
				for( std::size_t j = 0; j < probs[i].size(); ++j ) {
					x = probs[i][j];
					d1 += x * j;
				}
			}
			af = d1 / (2.0 * n_samples);
			if (params.maf_lim && af < params.min_maf) {
				bgenParser.ignore_probs();
				continue;
			}

			// filters passed; write dosage to G
			maf.push_back(af);
			rsid.push_back(my_rsid);
			chromosome.push_back(my_chr);
			position.push_back(my_pos);
			alleles.push_back(my_alleles);
			
			bgenParser.read_probs( &probs ) ;
			for( std::size_t ii = 0; ii < probs.size(); ++ii ) {
				dosage = 0.0;
				for( std::size_t kk = 0; kk < probs[ii].size(); ++kk ) {
					x = probs[ii][kk];
					dosage += x * kk;
				}
				G(ii,jj) = dosage;
			}
			jj++;
		}

		// if while loop exits early due to EOF,
		// we need to resize G whilst retaining existing coefficients.
		G.conservativeResize(n_samples, jj);
		assert( rsid.size() == jj );
		assert( chromosome.size() == jj );
		assert( position.size() == jj );
		assert( alleles.size() == jj );
		G_ncol = jj;

		return true;
	}
};

#endif
