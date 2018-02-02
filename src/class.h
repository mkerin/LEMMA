// File of classes for use with src/bgen_prog.cpp
#ifndef CLASS_H
#define CLASS_H

#include <iostream>
#include <string>
#include <vector>

class parameters {
	public :
		std::string bgen_file, chr, out_file, pheno_file, covar_file, incl_sids_file;
		int chunk_size, missing_code;
		uint32_t start, end;
		bool range, maf_lim, mode_vcf, mode_lm;
		double min_maf;
		std::vector < std::string > incl_sample_ids;
	
	// constructors/destructors	
	parameters() {
		bgen_file = "NULL";
		pheno_file = "NULL";
		covar_file = "NULL";
		out_file = "NULL";
		incl_sids_file = "NULL";
		chunk_size = 256;
		missing_code = -999;
		range = false;
		maf_lim = false;
		mode_lm = false;
		mode_vcf = false;
	}

	~parameters() {
	}
};

#endif
