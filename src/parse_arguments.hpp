// parse_arguments
#ifndef PARSE_ARGUMENTS_HPP
#define PARSE_ARGUMENTS_HPP

#include <iostream>
#include <iomanip>
#include <set>
#include <cstring>
#include <sys/stat.h>
#include "class.h"
#include "version.h"
#include "my_timer.hpp"
#include <regex>
#include <stdexcept>
#include "tools/eigen3.3/Dense" // For vectorise profiling

void check_counts(const std::string& in_str, int i, int num, int argc);
void parse_arguments(parameters &p, int argc, char *argv[]);
void check_file_exists(const std::string& filename);

// For vectorise profiling
void foo(const Eigen::VectorXi& aa1,
         const Eigen::VectorXi& aa2,
         Eigen::VectorXi& aa){
	asm("#it begins here!");
		aa = aa1 + aa2;
	asm("#it ends here!");
}

void foo(const Eigen::VectorXf& aa1,
         const Eigen::VectorXf& aa2,
         Eigen::VectorXf& aa){
	asm("#it begins here!");
		aa = aa1 + aa2;
	asm("#it ends here!");
}

void foo(const Eigen::VectorXf& aa1,
         const Eigen::VectorXd& aa2,
         Eigen::VectorXf& aa){
	asm("#it begins here!");
		aa = aa1 + aa2.cast<float>();
	asm("#it ends here!");
}

void foo(const Eigen::VectorXd& aa1,
         const Eigen::VectorXd& aa2,
         Eigen::VectorXd& aa){
	asm("#it begins here!");
		aa = aa1 + aa2;
	asm("#it ends here!");
}

void check_counts(const std::string& in_str, int i, int num, int argc) {
	// Stop overflow from argv
	if (i + num >= argc) {
		if (num == 1) {
			std::cout << "ERROR: flag " << in_str << " requres an argument. ";
			std::cout << "Please refer to the manual for usage instructions." << std::endl;
		} else {
			std::cout << "ERROR: flag " << in_str << " seems to require ";
 			std::cout << std::to_string(num) + " arguments. No arguments of ";
			std::cout << "this type should be implemented yet.." << std::endl;
		}
		std::exit(EXIT_FAILURE);
	}
}

void check_file_exists(const std::string& filename){
	// Throw error if given file does not exist.
	// NB: Doesn't check if file is empty etc.
	struct stat buf;
	if(stat(filename.c_str(), &buf) != 0){
		std::cout << "File " << filename << " does not exist" << std::endl;
		throw std::runtime_error("ERROR: file does not exist");
	}
}

void parse_arguments(parameters &p, int argc, char *argv[]) {
	char *in_str;
	int i;
	std::set<std::string> option_list {
		"--bgen",
		"--pheno",
		"--covar",
		"--recombination_map",
		"--environment",
		"--environment_weights",
		"--snpwise_scan",
		"--chunk",
		"--range",
		"--maf",
		"--info",
		"--out",
		"--mode_vb",
		"--mode_empirical_bayes",
		"--effects_prior_mog",
		"--use_vb_on_covars",
		"--threads",
		"--low_mem",
		"--high_mem",
		"--incl_sample_ids",
		"--incl_rsids",
		"--rsid",
		"--no_geno_check",
		"--genetic_confounders",
		"--r1_hyps_grid",
		"--r1_probs_grid",
		"--min_elbo_diff",
		"--min_alpha_diff",
		"--hyps_grid",
		"--hyps_probs",
		"--vb_init",
		"--verbose",
		"--xtra_verbose",
		"--keep_constant_variants",
		"--force_round1",
		"--raw_phenotypes",
		"--mode_alternating_updates",
		"--mode_no_gxe",
		"--vb_iter_max",
		"--burnin_maxhyps",
		"--env_update_repeats",
		"--gamma_updates_thresh",
		"--init_weights_with_snpwise_scan",
		"--dxteex",
		"--mode_mog_beta",
		"--mode_mog_gamma",
		"--gxe_chunk_size",
		"--main_chunk_size",
		"--spike_diff_factor",
		"--min_spike_diff_factor"
	};

	std::set<std::string>::iterator set_it;
	// for --version (& no other args), print splash screen and exit.
	if (argc == 2 && strcmp(argv[1], "--version") == 0) {
		printf("%s\n\n", splash);
		std::exit(EXIT_SUCCESS);
	}

	if (argc == 1) {
		std::cout << "======-----"<< std::endl;
		std::cout << "Matt's BGEN PROG" << std::endl;
		std::cout << "======-----" << std::endl << std::endl;

#ifdef EIGEN_VECTORIZE
		std::cout << "Details:" << std::endl;
		std::cout << "- vectorization with SSE is ON" << std::endl;
#else
		std::cout << "Details:" << std::endl;
		std::cout << "- vectorization with SSE is OFF" << std::endl;
#endif

#ifdef DEBUG
		std::cout << "- in DEBUG mode (multithreading not used)" << std::endl;
#endif

#ifdef OSX
		std::cout << "- OSX compatible" << std::endl;
#else
		std::cout << "- LINUX compatible" << std::endl;
#endif

#ifdef DATA_AS_FLOAT
		std::cout << "- Data encoded as float" << std::endl;
#else
		std::cout << "- Data encoded as doubles" << std::endl;
#endif
#ifdef DEBUG
		std::cout << "- in DEBUG mode" << std::endl;
#endif
#ifdef EIGEN_USE_MKL_ALL
		std::cout << "- compiled with Intel MKL backend" << std::endl;
#else
        std::cout << "- compiled with native Eigen backend" << std::endl;
#endif
#ifdef OSX
		std::cout << "- OSX compatible" << std::endl;
#endif
		// For vectorise profiling
		Eigen::VectorXi aa1 = Eigen::VectorXi::Random(256000);
		Eigen::VectorXi aa2 = Eigen::VectorXi::Random(256000);
		Eigen::VectorXi aa(256000);
		Eigen::VectorXf bb1 = Eigen::VectorXf::Random(256000);
		Eigen::VectorXf bb2 = Eigen::VectorXf::Random(256000);
		Eigen::VectorXf bb(256000);
		Eigen::VectorXd cc1 = Eigen::VectorXd::Random(256000);
		Eigen::VectorXd cc2 = Eigen::VectorXd::Random(256000);
		Eigen::VectorXd cc(256000);

		MyTimer t_testi("500 foo() calls in %ts (int)\n");
		t_testi.resume();
		for (int jj = 0; jj < 500; jj++){
			foo(aa1, aa2, aa);
		}
		t_testi.stop();
		t_testi.report();

		MyTimer t_testf("500 foo() calls in %ts (float)\n");
		t_testf.resume();
		for (int jj = 0; jj < 500; jj++){
			foo(bb1, bb2, bb);
		}
		t_testf.stop();
		t_testf.report();

		MyTimer t_testf_cast("500 foo() calls in %ts (float via cast)\n");
		t_testf_cast.resume();
		for (int jj = 0; jj < 500; jj++){
			foo(bb1, cc2, bb);
		}
		t_testf_cast.stop();
		t_testf_cast.report();


		MyTimer t_testd("500 foo() calls in %ts (double)\n");
		t_testd.resume();
		for (int jj = 0; jj < 500; jj++){
			foo(cc1, cc2, cc);
		}
		t_testd.stop();
		t_testd.report();

		std::exit(EXIT_SUCCESS);
	}

	// read in and check option flags
	for (i = 0; i < argc; i++) {
		in_str = argv[i];
		if (strcmp(in_str, "--version") == 0 || strcmp(in_str, "--help") == 0) {
			std::cout << "ERROR: flag '" << in_str << "' cannot be used with any other flags." << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	// Ensure some arguments only appear once
	bool check_out = 0;

	for(i = 0; i < argc; i++){
		if(*argv[i] == '-'){
			in_str = argv[i];
			set_it = option_list.find(in_str);

			if(set_it == option_list.end()) {
				std::cout << "ERROR: flag '" << in_str <<
					"' not valid. Please refer to the manual for usage instructions." <<
					std::endl;

				exit(EXIT_FAILURE);
			}

			// flags with parameters should eat their arguments
			// & also make sure they don't go over argc

			// Modes - a variety of different functionalities now included
			if(strcmp(in_str, "--gxe_chunk_size") == 0) {
				p.gxe_chunk_size = std::stoi(argv[i + 1]);
				i += 1;
			}

            if(strcmp(in_str, "--main_chunk_size") == 0) {
				p.main_chunk_size = std::stoi(argv[i + 1]);
				i += 1;
			}

			if(strcmp(in_str, "--burnin_maxhyps") == 0) {
				p.burnin_maxhyps = std::stoi(argv[i + 1]);
				i += 1;
			}

			if(strcmp(in_str, "--env_update_repeats") == 0) {
				p.env_update_repeats = std::stoi(argv[i + 1]);
				i += 1;
			}

			if(strcmp(in_str, "--init_weights_with_snpwise_scan") == 0) {
				p.init_weights_with_snpwise_scan = true;
				i += 0;
			}

			if(strcmp(in_str, "--gamma_updates_thresh") == 0) {
				p.restrict_gamma_updates = true;
				p.gamma_updates_thresh = std::stod(argv[i + 1]);
				i += 1;
			}

			if(strcmp(in_str, "--vb_iter_max") == 0) {
				p.vb_iter_max = std::stol(argv[i + 1]);
				i += 1;
			}

			if(strcmp(in_str, "--spike_diff_factor") == 0) {
				p.spike_diff_factor = std::stod(argv[i + 1]);
				std::cout << "Initial slab variance " << p.spike_diff_factor << "x spike variance" << std::endl;
				i += 1;
			}

			if(strcmp(in_str, "--min_spike_diff_factor") == 0) {
				p.min_spike_diff_factor = std::stod(argv[i + 1]);
				p.min_spike_diff_set = true;
				std::cout << "Slab variance constrained to atleast " << p.min_spike_diff_factor << "x spike variance" << std::endl;
				i += 1;
			}

			if(strcmp(in_str, "--mode_vb") == 0) {
				p.mode_vb = true;
				i += 0;
			}

			if(strcmp(in_str, "--effects_prior_mog") == 0) {
				p.mode_mog_prior_beta = true;
				p.mode_mog_prior_gam = true;
				i += 0;
			}

			if(strcmp(in_str, "--mode_mog_beta") == 0) {
				p.mode_mog_prior_beta = true;
				i += 0;
			}

            if(strcmp(in_str, "--mode_no_gxe") == 0) {
				p.mode_no_gxe = true;
				i += 0;
			}

			if(strcmp(in_str, "--mode_mog_gamma") == 0) {
				p.mode_mog_prior_gam = true;
				i += 0;
			}

			if(strcmp(in_str, "--use_vb_on_covars") == 0) {
				p.use_vb_on_covars = true;
				i += 0;
			}

			if(strcmp(in_str, "--mode_alternating_updates") == 0) {
				p.mode_alternating_updates = true;
				i += 0;
			}

			if(strcmp(in_str, "--keep_constant_variants") == 0) {
				p.keep_constant_variants = true;
				i += 0;
			}

			if(strcmp(in_str, "--force_round1") == 0) {
				p.user_requests_round1 = true;
				i += 0;
			}

			if(strcmp(in_str, "--mode_empirical_bayes") == 0) {
				p.mode_empirical_bayes = true;
				i += 0;
			}

			if(strcmp(in_str, "--verbose") == 0) {
				p.verbose = true;
				i += 0;
			}

			if(strcmp(in_str, "--xtra_verbose") == 0) {
				p.verbose = true;
				p.xtra_verbose = true;
				i += 0;
			}

			if(strcmp(in_str, "--low_mem") == 0) {
				p.low_mem = true;
				i += 0;
			}

			if(strcmp(in_str, "--high_mem") == 0) {
				p.low_mem = false;
				i += 0;
			}

			if(strcmp(in_str, "--raw_phenotypes") == 0) {
				p.scale_pheno = false;
				i += 0;
			}

			// Data inputs
			if(strcmp(in_str, "--bgen") == 0) {
				check_counts(in_str, i, 1, argc);
				p.bgen_file = argv[i + 1]; // bgen file

				check_file_exists(p.bgen_file);
				i += 1;
			}

			if(strcmp(in_str, "--pheno") == 0) {
				check_counts(in_str, i, 1, argc);
				p.pheno_file = argv[i + 1]; // pheno file
				check_file_exists(p.pheno_file);
				i += 1;
			}

			if(strcmp(in_str, "--recombination_map") == 0) {
				check_counts(in_str, i, 1, argc);
				p.recombination_file = argv[i + 1]; // pheno file
				check_file_exists(p.recombination_file);
				i += 1;
			}

			if(strcmp(in_str, "--environment") == 0) {
				check_counts(in_str, i, 1, argc);
				p.interaction_analysis = true;
				p.env_file = argv[i + 1]; // pheno file
				check_file_exists(p.env_file);
				i += 1;
			}

			if(strcmp(in_str, "--environment_weights") == 0) {
				check_counts(in_str, i, 1, argc);
				p.env_weights_file = argv[i + 1]; // pheno file
				check_file_exists(p.env_weights_file);
				i += 1;
			}

			if(strcmp(in_str, "--covar") == 0) {
				check_counts(in_str, i, 1, argc);
				p.covar_file = argv[i + 1]; // covar file
				check_file_exists(p.covar_file);
				i += 1;
			}

			if(strcmp(in_str, "--snpwise_scan") == 0) {
				check_counts(in_str, i, 1, argc);
				p.snpstats_file = argv[i + 1]; // covar file
				check_file_exists(p.snpstats_file);
				i += 1;
			}

			if(strcmp(in_str, "--out") == 0) {
				if (check_out == 1) {
					std::cout << "ERROR: flag '" << in_str << "' can only be provided once." << std::endl;
					exit(EXIT_FAILURE);
				}
				check_out = 1;
				check_counts(in_str, i, 1, argc);
				p.out_file = argv[i + 1];
				i += 1;
			}

			if(strcmp(in_str, "--hyps_grid") == 0) {
				check_counts(in_str, i, 1, argc);
				p.hyps_grid_file = argv[i + 1]; // covar file
				check_file_exists(p.hyps_grid_file);
				i += 1;
			}

			if(strcmp(in_str, "--r1_hyps_grid") == 0) {
				check_counts(in_str, i, 1, argc);
				p.r1_hyps_grid_file = argv[i + 1]; // covar file
				check_file_exists(p.r1_hyps_grid_file);
				i += 1;
			}

			if(strcmp(in_str, "--r1_probs_grid") == 0) {
				check_counts(in_str, i, 1, argc);
				p.r1_probs_grid_file = argv[i + 1]; // covar file
				check_file_exists(p.r1_probs_grid_file);
				i += 1;
			}

			if(strcmp(in_str, "--hyps_probs") == 0) {
				check_counts(in_str, i, 1, argc);
				p.hyps_probs_file = argv[i + 1]; // covar file
				check_file_exists(p.hyps_probs_file);
				i += 1;
			}

			if(strcmp(in_str, "--vb_init") == 0) {
				check_counts(in_str, i, 1, argc);
				p.vb_init_file = argv[i + 1]; // covar file
				check_file_exists(p.vb_init_file);
				i += 1;
			}

			if(strcmp(in_str, "--dxteex") == 0) {
				check_counts(in_str, i, 1, argc);
				p.dxteex_file = argv[i + 1]; // covar file
				check_file_exists(p.dxteex_file);
				i += 1;
			}

			// Filters
			if(strcmp(in_str, "--maf") == 0) {
				check_counts(in_str, i, 1, argc);
				p.maf_lim = true;
				p.min_maf = std::stod(argv[i + 1]); // bgen file
				i += 1;
			}

			if(strcmp(in_str, "--info") == 0) {
				check_counts(in_str, i, 1, argc);
				p.info_lim = true;
				p.min_info = std::stod(argv[i + 1]); // bgen file
				i += 1;
			}

			if(strcmp(in_str, "--range") == 0) {
				static bool check = 0;
				if (check == 1) {
					std::cout << "ERROR: flag '" << in_str << "' can only be provided once." << std::endl;
					exit(EXIT_FAILURE);
				}
				check = 1;
				check_counts(in_str, i, 3, argc);
				p.range = true;
				p.chr = argv[i + 1];
				p.range_start = atoi(argv[i + 2]);
				p.range_end = atoi(argv[i + 3]);
				i += 3;
			}

			if(strcmp(in_str, "--threads") == 0) {
				check_counts(in_str, i, 1, argc);
				p.n_thread = atoi(argv[i + 1]);
				if(p.n_thread < 1) throw std::runtime_error("--threads must be positive.");
				i += 1;
			}

			if(strcmp(in_str, "--min_alpha_diff") == 0) {
				check_counts(in_str, i, 1, argc);
				p.alpha_tol_set_by_user = true;
				p.alpha_tol = atof(argv[i + 1]);
				if(p.alpha_tol < 0) throw std::runtime_error("--min_alpha_diff must be positive.");
				std::cout << "--min_alpha_diff of " << p.alpha_tol << " entered." << std::endl;
				i += 1;
			}

			if(strcmp(in_str, "--min_elbo_diff") == 0) {
				check_counts(in_str, i, 1, argc);
				p.elbo_tol_set_by_user = true;
				p.elbo_tol = atof(argv[i + 1]);
				if(p.elbo_tol < 0) throw std::runtime_error("--min_elbo_diff must be positive.");
				std::cout << "--min_elbo_diff of " << p.elbo_tol << " entered." << std::endl;
				i += 1;
			}

			if(strcmp(in_str, "--incl_sample_ids") == 0) {
				check_counts(in_str, i, 1, argc);
				p.incl_sids_file = argv[i + 1]; // include sample ids file
				check_file_exists(p.incl_sids_file);
				i += 1;
			}

			if(strcmp(in_str, "--incl_rsids") == 0) {
				check_counts(in_str, i, 1, argc);
				p.select_snps = true;
				p.incl_rsids_file = argv[i + 1]; // include variant ids file
				check_file_exists(p.incl_rsids_file);
				i += 1;
			}

			if(strcmp(in_str, "--rsid") == 0) {
				check_counts(in_str, i, 1, argc);
				p.select_rsid = true;
				int jj = i+1;
				while(jj < argc){
					std::string arg_str(argv[jj]);
					if (arg_str.find("--") != std::string::npos) break;
					p.rsid.push_back(argv[jj]);
					jj++;
				}
				i += 1;
			}

			// Other options
			if(strcmp(in_str, "--no_geno_check") == 0) {
				p.geno_check = false;
				i += 0;
			}

			if(strcmp(in_str, "--chunk") == 0) {
				check_counts(in_str, i, 1, argc);
				p.chunk_size = std::stoi(argv[i + 1]); // bgen file
				i += 1;
			}

			if(strcmp(in_str, "--genetic_confounders") == 0) {
				check_counts(in_str, i, 1, argc);
				int jj = i+1;
				while(jj < argc){
					std::string arg_str(argv[jj]);
					if (arg_str.find("--") != std::string::npos) break;
					p.gconf.push_back(argv[jj]);
					jj++;
				}
				p.n_gconf = jj - i - 1;
				i += 1;
			}
		}
	}

	// Sanity checks here
	if(p.range || p.select_snps){
		struct stat buf;
		p.bgi_file = p.bgen_file + ".bgi";
		if(stat(p.bgi_file.c_str(), &buf) != 0){
			std::cout << "If using --range the BGEN index file " << p.bgi_file << " must exist" << std::endl;
			throw std::runtime_error("ERROR: file does not exist");
		}
	}

	// mode_vb specific options
	bool has_bgen = p.bgen_file != "NULL";
	bool has_out = p.out_file != "NULL";
	bool has_pheno = p.pheno_file != "NULL";
	bool has_all = (has_pheno && has_out && has_bgen);
	if(!has_all){
		std::cout << "ERROR: bgen, pheno and out filepaths should all be ";
		std::cout << "provided in conjunction with --mode_vb." << std::endl;
		std::exit(EXIT_FAILURE);
	}
	bool has_hyps = p.hyps_grid_file != "NULL";
	if(!has_hyps){
		std::cout << "ERROR: search grids for hyperparameter values";
		std::cout << "should be provided in conjunction with --mode_vb." << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if(p.env_file == "NULL" && p.env_weights_file != "NULL"){
		std::cout << "WARNING: --environment_weights will be ignored as no --environment provided" << std::endl;
	}
}

#endif
