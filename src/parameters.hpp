// File of classes for use with src/bgen_prog.cpp
#ifndef CLASS_H
#define CLASS_H

#include <iostream>
#include <string>
#include <vector>

class parameters {
public:
std::string bgen_file, chr, out_file, pheno_file, env_file, covar_file, bgi_file;
std::string incl_sids_file, incl_rsids_file, recombination_file;
std::string r1_hyps_grid_file, r1_probs_grid_file, hyps_grid_file;
std::string env_coeffs_file, covar_coeffs_file, hyps_probs_file, vb_init_file;
std::string dxteex_file, snpstats_file, mog_weights_file;
std::string beta_prior, gamma_prior;
std::vector< std::string > rsid;
unsigned int random_seed;
long int chunk_size, vb_iter_max, vb_iter_start, param_dump_interval;
int missing_code, env_update_repeats, n_gconf, n_bgen_thread, n_pve_samples;
unsigned int n_thread, main_chunk_size, gxe_chunk_size;
std::uint32_t range_start, range_end;
bool range, maf_lim, info_lim, select_snps, xtra_verbose;
bool geno_check, mode_vb, use_vb_on_covars;
bool select_rsid, interaction_analysis, verbose, low_mem;
bool elbo_tol_set_by_user, alpha_tol_set_by_user, mode_empirical_bayes;
bool keep_constant_variants, user_requests_round1, scale_pheno;
bool mode_alternating_updates, mode_pve_est;
bool restrict_gamma_updates, mode_no_gxe, mode_debug;
bool init_weights_with_snpwise_scan, flip_high_maf_variants, min_spike_diff_set;
bool mode_random_start;
bool mode_remove_squared_envs, mode_squarem, mode_incl_squared_envs;
double min_maf, min_info, elbo_tol, alpha_tol, gamma_updates_thresh;
double beta_spike_diff_factor, gam_spike_diff_factor, min_spike_diff_factor;
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
	vb_init_file("NULL"),
	incl_sids_file("NULL"),
	incl_rsids_file("NULL"),
	dxteex_file("NULL"),
	mog_weights_file("NULL"),
	covar_coeffs_file("NULL"),
	env_coeffs_file("NULL") {
	flip_high_maf_variants = false;
	init_weights_with_snpwise_scan = false;
	restrict_gamma_updates = false;
	n_thread = 1;
	n_bgen_thread = 1;
	random_seed = -1;
	env_update_repeats = 1;
	interaction_analysis = false;
	chunk_size = 256;
	main_chunk_size = 64;
#ifdef DATA_AS_FLOAT
	gxe_chunk_size = 16;
#else
	gxe_chunk_size = 8;
#endif
	missing_code = -999;
	vb_iter_max = 10000;
	vb_iter_start = 0;
	n_pve_samples = 100;
	beta_spike_diff_factor = 10000.0;                     // Initial diff in variance of spike & slab
	gam_spike_diff_factor = 10000.0;                     // Initial diff in variance of spike & slab
	param_dump_interval = -1;
	range = false;
	min_spike_diff_set = false;
	mode_remove_squared_envs = true;
	mode_incl_squared_envs = false;
	maf_lim = false;
	info_lim = false;
	mode_empirical_bayes = true;
	mode_squarem = true;
	mode_alternating_updates = false;
	low_mem = true;
	mode_vb = false;
	mode_no_gxe = false;
	beta_prior = "MixOfGaussian";
	gamma_prior = "MixOfGaussian";
	mode_debug = false;
	mode_random_start = false;
	mode_pve_est = false;
	select_snps = false;
	select_rsid = false;
	geno_check = true;                     // check allele probs sum to 1 by default
	verbose = false;
	xtra_verbose = false;
	use_vb_on_covars = true;
	alpha_tol_set_by_user = false;
	elbo_tol_set_by_user = false;
	keep_constant_variants = false;
	user_requests_round1 = false;
	scale_pheno = true;
}

~parameters() = default;
};

#endif
