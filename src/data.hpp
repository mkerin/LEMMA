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

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>

namespace boost_io = boost::iostreams;

class data 
{
	public :
	parameters params;
	
	std::vector<std::string> chromosome, rsid, SNPID;
	std::vector<uint32_t> position;
	std::vector<std::vector<std::string>> alleles;
	
	int n_pheno; // number of phenotypes
	int n_covar; // number of covariates
	int n_samples; // number of samples
	long int n_snps; // number of snps

	std::vector<double> info;
	std::vector<double> maf;

	Eigen::MatrixXd G; // genotype matrix --

	boost_io::filtering_ostream outf;
	
	// constructors/destructors	
	data() {
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
};


#endif
