// File of classes for use with src/bgen_prog.cpp
#ifndef CLASS_H
#define CLASS_H

#include <iostream>
#include <string>
#include <vector>

class parameters {
	public :
		std::string bgen_file, chr, out_file, pheno_file, env_file, covar_file, bgi_file;
		std::string incl_sids_file, x_param_name, incl_rsids_file, recombination_file;
		std::string r1_hyps_grid_file, r1_probs_grid_file, hyps_grid_file;
		std::string env_weights_file, hyps_probs_file, vb_init_file;
		std::string dxteex_file, snpstats_file;
		std::vector< std::string > rsid;
		long int chunk_size, vb_iter_max;
		int missing_code, n_gconf, n_thread, burnin_maxhyps, env_update_repeats;
		int vb_chunk_size;
		uint32_t start, end;
		bool range, maf_lim, info_lim, test_2dof, select_snps, xtra_verbose;
		bool geno_check, bgen_wildcard, mode_vb, use_vb_on_covars;
		bool select_rsid, interaction_analysis, verbose, low_mem;
		bool elbo_tol_set_by_user, alpha_tol_set_by_user, mode_empirical_bayes;
		bool keep_constant_variants, user_requests_round1, scale_pheno;
		bool mode_alternating_updates, mode_sgd, sgd_delay_set, sgd_forgetting_rate_set;
		bool sgd_minibatch_size_set, rescale_eta, restrict_gamma_updates;
		bool init_weights_with_snpwise_scan, flip_high_maf_variants, min_spike_diff_set;
		bool mode_mog_prior_beta, mode_mog_prior_gam;
		double min_maf, min_info, elbo_tol, alpha_tol, gamma_updates_thresh;
		double sgd_delay, sgd_forgetting_rate, spike_diff_factor, min_spike_diff_factor;
		std::vector < std::string > incl_sample_ids, gconf;
		long int sgd_minibatch_size;

	// constructors/destructors
	parameters() : bgen_file("NULL"),
		out_file("NULL"),
		pheno_file("NULL"),
		recombination_file("NULL"),
		covar_file("NULL"),
		env_file("NULL"),
		bgi_file("NULL"),
		snpstats_file("NULL"),
		r1_hyps_grid_file("NULL"),
		r1_probs_grid_file("NULL"),
		hyps_grid_file("NULL"),
		hyps_probs_file("NULL"),
		vb_init_file("NULL"),
		incl_sids_file("NULL"),
		incl_rsids_file("NULL"),
		x_param_name("NULL"),
		dxteex_file("NULL"),
		env_weights_file("NULL") {
		bgen_wildcard = false;
		rescale_eta = false;
		flip_high_maf_variants = true;
		init_weights_with_snpwise_scan = false;
		restrict_gamma_updates = false;
		n_thread = 1;
		burnin_maxhyps = 0;
		env_update_repeats = 1;
		interaction_analysis = false;
		chunk_size = 256;
		vb_chunk_size = 64;
		missing_code = -999;
		vb_iter_max = 10000;
		spike_diff_factor = 1000000.0; // Initial diff in variance of spike & slab
		range = false;
		min_spike_diff_set = false;
		maf_lim = false;
		info_lim = false;
		mode_empirical_bayes = false;
		mode_alternating_updates = false;
		low_mem = false;
		mode_vb = false;
		mode_mog_prior_beta = false;
		mode_mog_prior_gam = false;
		test_2dof = true;
		select_snps = false;
		select_rsid = false;
		geno_check = true; // check allele probs sum to 1 by default
		verbose = false;
		xtra_verbose = false;
		use_vb_on_covars = false;
		alpha_tol_set_by_user = false;
		elbo_tol_set_by_user = false;
		keep_constant_variants = false;
		user_requests_round1 = false;
		scale_pheno = true;
		mode_sgd = false;
		sgd_forgetting_rate_set = false;
		sgd_delay_set = false;
		sgd_minibatch_size_set = false;
	}

	~parameters() {
	}
};

#endif
