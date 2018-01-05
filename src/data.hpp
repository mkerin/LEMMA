// File of Data class for use with src/bgen_prog.cpp
#ifndef DATA_H
#define DATA_H

#include <iostream>
#include "class.h"
#include "tools/eigen3.3/Eigen/Dense"
#include "tools/eigen3.3/Eigen/Sparse"
#include "tools/eigen3.3/Eigen/Eigenvalues"

class data 
{
	public :
	parameters params;
	
	vector<string> chromosome, rsid, SNPID;
	vector<uint32_t> position;
	vector<vector<string>> alleles;
	
	int n_pheno; // number of phenotypes
	int n_covar; // number of covariates
	int n_samples; // number of samples
	long int n_snps; // number of snps

	vector<double> info;
	vector<double> maf;

	MatrixXd G; // genotype matrix --
	
	// constructors/destructors	
	data() {
		missing_phenotypes = false;
	}
	
	~data() {
	}
};


#endif
