// File of classes for use with src/bgen_prog.cpp
#ifndef CLASS_H
#define CLASS_H

#include <iostream>
#include <string>
#include <vector>

class parameters {
	public :
		std::string bgen_file, chr, out_file, pheno_file, covar_file, bgi_file;
		std::string incl_sids_file, x_param_name, incl_rsids_file;
		int chunk_size, missing_code;
		uint32_t start, end;
		bool range, maf_lim, info_lim, mode_vcf, mode_lm, test_2dof, select_snps;
		bool geno_check;
		double min_maf, min_info;
		std::vector < std::string > incl_sample_ids;
	
	// constructors/destructors	
	parameters() {
		bgen_file = "NULL";
		bgi_file = "NULL";
		pheno_file = "NULL";
		covar_file = "NULL";
		out_file = "NULL";
		incl_sids_file = "NULL";
		incl_rsids_file = "NULL";
		x_param_name = "NULL";
		chunk_size = 256;
		missing_code = -999;
		range = false;
		maf_lim = false;
		info_lim = false;
		mode_lm = false;
		mode_vcf = false;
		test_2dof = true;
		select_snps = false;
		geno_check = true; // check allele probs sum to 1 by default
	}

	~parameters() {
	}
};

#endif
