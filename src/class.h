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
		std::string r1_hyps_grid_file, r1_probs_grid_file, hyps_grid_file, hyps_probs_file, vb_init_file;
		std::vector< std::string > rsid;
		long int chunk_size;
		int missing_code, n_gconf, n_thread;
		uint32_t start, end;
		bool range, maf_lim, info_lim, mode_vcf, mode_lm, test_2dof, select_snps;
		bool geno_check, mode_joint_model, bgen_wildcard, mode_lm2, mode_vb;
		bool select_rsid, interaction_analysis, verbose, low_mem;
		bool elbo_tol_set_by_user, alpha_tol_set_by_user, mode_empirical_bayes;
		bool keep_constant_variants;
		double min_maf, min_info, elbo_tol, alpha_tol;
		std::vector < std::string > incl_sample_ids, gconf;
	
	// constructors/destructors	
	parameters() {
		bgen_file = "NULL";
		bgen_wildcard = false;
		bgi_file = "NULL";
		pheno_file = "NULL";
		covar_file = "NULL";
		out_file = "NULL";
		r1_hyps_grid_file = "NULL";
		r1_probs_grid_file = "NULL";
		hyps_grid_file = "NULL";
		hyps_probs_file = "NULL";
		vb_init_file = "NULL";
		incl_sids_file = "NULL";
		incl_rsids_file = "NULL";
		x_param_name = "NULL";
		n_thread = 1;
		interaction_analysis = false;
		chunk_size = 256;
		missing_code = -999;
		range = false;
		maf_lim = false;
		info_lim = false;
		mode_lm = false;
		mode_lm2 = false;
		mode_vcf = false;
		mode_empirical_bayes = false;
		low_mem = false;
		mode_joint_model = false;
		mode_vb = false;
		test_2dof = true;
		select_snps = false;
		select_rsid = false;
		geno_check = true; // check allele probs sum to 1 by default
		verbose = false;
		alpha_tol_set_by_user = false;
		elbo_tol_set_by_user = false;
		keep_constant_variants = false;
	}

	~parameters() {
	}
};

#endif
