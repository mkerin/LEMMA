// File of classes for use with src/bgen_prog.cpp
#ifndef CLASS_H
#define CLASS_H

#include <iostream>
#include <string>
#include <vector>

#include "genfile/bgen/bgen.hpp"
#include "genfile/bgen/View.hpp"

class parameters {
public:
	std::string bgen_file, chr, out_file, pheno_file, env_file, covar_file, bgi_file;
	std::string incl_sids_file, incl_rsids_file, recombination_file;
	std::string r1_hyps_grid_file, r1_probs_grid_file, hyps_grid_file, rhe_random_vectors_file;
	std::string env_coeffs_file, covar_coeffs_file, hyps_probs_file, vb_init_file;
	std::string dxteex_file, snpstats_file, mog_weights_file, resume_prefix;
	std::string streamBgenOutFile, extra_pve_covar_file;
	std::vector< std::string > rsid;
//	std::string streamBgenFiles, streamBgiFiles;
	std::vector< std::string > streamBgenFiles, streamBgiFiles, RHE_groups_files;
	unsigned int random_seed;
	long chunk_size, vb_iter_max, vb_iter_start, param_dump_interval, n_pve_samples;
	long long int maxBytesPerRank;
	int missing_code, env_update_repeats, n_gconf, n_bgen_thread;
	unsigned int n_thread, main_chunk_size, gxe_chunk_size;
	std::uint32_t range_start, range_end;
	bool range, maf_lim, info_lim, xtra_verbose, joint_covar_update, mode_RHEreg_LM;
	bool geno_check, mode_vb, use_vb_on_covars;
	bool select_rsid, interaction_analysis, verbose, low_mem;
	bool elbo_tol_set_by_user, alpha_tol_set_by_user, mode_empirical_bayes;
	bool keep_constant_variants, user_requests_round1, scale_pheno;
	bool mode_alternating_updates, mode_RHE, mode_RHE_fast;
	bool restrict_gamma_updates, mode_no_gxe, mode_debug;
	bool init_weights_with_snpwise_scan, flip_high_maf_variants, min_spike_diff_set;
	bool mode_mog_prior_beta, mode_mog_prior_gam, mode_random_start, mode_calc_snpstats;
	bool mode_remove_squared_envs, mode_squarem, mode_incl_squared_envs, drop_loco;
	bool exclude_ones_from_env_sq, mode_RHEreg_NM;
	long redo_ym_interval, levenburgMarquardt_max_iter;
	double min_maf, min_info, elbo_tol, alpha_tol, gamma_updates_thresh;
	double beta_spike_diff_factor, gam_spike_diff_factor, min_spike_diff_factor;
	long LOSO_window, n_jacknife, streamBgen_print_interval, nelderMead_max_iter;
	bool RHE_multicomponent, mode_dump_processed_data, use_raw_env;
	std::vector < std::string > incl_sample_ids, gconf;

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
		extra_pve_covar_file("NULL"),
		vb_init_file("NULL"),
		incl_sids_file("NULL"),
		incl_rsids_file("NULL"),
		dxteex_file("NULL"),
		mog_weights_file("NULL"),
		covar_coeffs_file("NULL"),
		resume_prefix("NULL"),
		env_coeffs_file("NULL"),
		rhe_random_vectors_file("NULL"),
		streamBgenOutFile("NULL") {
		flip_high_maf_variants = false;
		init_weights_with_snpwise_scan = false;
		restrict_gamma_updates = false;
		n_thread = 1;
		n_jacknife = 2;
		redo_ym_interval = -1;
		n_bgen_thread = 1;
		random_seed = -1;
		env_update_repeats = 1;
		interaction_analysis = false;
		joint_covar_update = false;
		drop_loco = false;
		use_raw_env = false;
		chunk_size = 256;
		main_chunk_size = 64;
#ifdef DATA_AS_FLOAT
		gxe_chunk_size = 16;
#else
		gxe_chunk_size = 8;
#endif
		missing_code = -999;
		vb_iter_max = 10000;
		maxBytesPerRank = 14500000000;
		vb_iter_start = 0;
		n_pve_samples = 50;
		LOSO_window = 100000;
		// Initial difference in variance of spike & slab
		beta_spike_diff_factor = 1000;
		gam_spike_diff_factor = 1000;
		param_dump_interval = 50;
		streamBgen_print_interval = 100;
		range = false;
		min_spike_diff_set = false;
		mode_remove_squared_envs = true;
		mode_incl_squared_envs = false;
		exclude_ones_from_env_sq = false;
		maf_lim = false;
		info_lim = false;
		mode_empirical_bayes = true;
		mode_squarem = true;
		mode_alternating_updates = false;
		low_mem = true;
		mode_vb = false;
		mode_no_gxe = false;
		mode_calc_snpstats = false;
		mode_mog_prior_beta = true;
		mode_mog_prior_gam = true;
		mode_debug = false;
		mode_random_start = false;
		mode_RHE = false;
		mode_RHE_fast = false;
		mode_RHEreg_NM = false;
		mode_RHEreg_LM = false;
		levenburgMarquardt_max_iter = 100;
		nelderMead_max_iter = 100;
		RHE_multicomponent = false;
		select_rsid = false;
		mode_dump_processed_data = false;
		// check allele probs sum to 1 by default
		geno_check = true;
		verbose = false;
		xtra_verbose = false;
		use_vb_on_covars = true;
		alpha_tol_set_by_user = false;
		elbo_tol_set_by_user = false;
		elbo_tol = 0.01;
		alpha_tol = 0.001;
		keep_constant_variants = false;
		user_requests_round1 = false;
		scale_pheno = true;
	}

	~parameters() = default;
};

#endif
