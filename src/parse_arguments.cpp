//
// Created by kerin on 2019-03-01.
//

#include "parse_arguments.hpp"
#include "parameters.hpp"
#include "my_timer.hpp"

#include "tools/eigen3.3/Dense" // For vectorise profiling

#include <iostream>
#include <iomanip>
#include <set>
#include <cstring>
#include <boost/filesystem.hpp>
#include <regex>
#include <stdexcept>

// For vectorise profiling
void foo(const Eigen::VectorXi& aa1,
         const Eigen::VectorXi& aa2,
         Eigen::VectorXi& aa){
	asm ("#it begins here!");
	aa = aa1 + aa2;
	asm ("#it ends here!");
}

void foo(const Eigen::VectorXf& aa1,
         const Eigen::VectorXf& aa2,
         Eigen::VectorXf& aa){
	asm ("#it begins here!");
	aa = aa1 + aa2;
	asm ("#it ends here!");
}

void foo(const Eigen::VectorXf& aa1,
         const Eigen::VectorXd& aa2,
         Eigen::VectorXf& aa){
	asm ("#it begins here!");
	aa = aa1 + aa2.cast<float>();
	asm ("#it ends here!");
}

void foo(const Eigen::VectorXd& aa1,
         const Eigen::VectorXd& aa2,
         Eigen::VectorXd& aa){
	asm ("#it begins here!");
	aa = aa1 + aa2;
	asm ("#it ends here!");
}

void check_counts(const std::string &in_str, int i, int num, int argc) {
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

void parse_arguments(parameters &p, int argc, char **argv) {
	char *in_str;
	int i;
	std::set<std::string> option_list {
		"--bgen",
		"--pheno",
		"--covar",
		"--recombination_map",
		"--drop_loco",
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
		"--mode_squarem",
		"--mode_constant_hyps",
		"--effects_prior_mog",
		"--mode_spike_slab",
		"--use_vb_on_covars",
		"--mode_regress_out_covars",
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
		"--vb_iter_start",
		"--env_update_repeats",
		"--gamma_updates_thresh",
		"--init_weights_with_snpwise_scan",
		"--suppress_squared_env_removal",
		"--incl_squared_envs",
		"--resume_from_param_dump",
		"--dxteex",
		"--mode_mog_beta",
		"--mode_mog_gamma",
		"--mode_pve_est",
		"--gxe_chunk_size",
		"--main_chunk_size",
		"--param_dump_interval",
		"--random_seed",
		"--mode_debug",
		"--pve_mog_weights",
		"--spike_diff_factor",
		"--beta_spike_diff_factor",
		"--gam_spike_diff_factor",
		"--min_spike_diff_factor",
		"--n_pve_samples"
	};

	std::set<std::string>::iterator set_it;
	// for --version (& no other args), print splash screen and exit.
	if (argc == 2 && strcmp(argv[1], "--version") == 0) {
		std::exit(EXIT_SUCCESS);
	}

	if (argc == 1) {
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
		for (int jj = 0; jj < 500; jj++) {
			foo(aa1, aa2, aa);
		}
		t_testi.stop();
		t_testi.report();

		MyTimer t_testf("500 foo() calls in %ts (float)\n");
		t_testf.resume();
		for (int jj = 0; jj < 500; jj++) {
			foo(bb1, bb2, bb);
		}
		t_testf.stop();
		t_testf.report();

		MyTimer t_testf_cast("500 foo() calls in %ts (float via cast)\n");
		t_testf_cast.resume();
		for (int jj = 0; jj < 500; jj++) {
			foo(bb1, cc2, bb);
		}
		t_testf_cast.stop();
		t_testf_cast.report();


		MyTimer t_testd("500 foo() calls in %ts (double)\n");
		t_testd.resume();
		for (int jj = 0; jj < 500; jj++) {
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

	for(i = 0; i < argc; i++) {
		if(*argv[i] == '-') {
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

			if(strcmp(in_str, "--vb_iter_start") == 0) {
				p.vb_iter_start = std::stol(argv[i + 1]);
				i += 1;
			}

			if(strcmp(in_str, "--n_pve_samples") == 0) {
				p.n_pve_samples = std::stoi(argv[i + 1]);
				i += 1;
			}

			if(strcmp(in_str, "--spike_diff_factor") == 0) {
				p.beta_spike_diff_factor = std::stod(argv[i + 1]);
				p.gam_spike_diff_factor = std::stod(argv[i + 1]);
				std::cout << "Initial slab variance " << p.beta_spike_diff_factor << "x spike variance" << std::endl;
				i += 1;
			}

			if(strcmp(in_str, "--beta_spike_diff_factor") == 0) {
				p.beta_spike_diff_factor = std::stod(argv[i + 1]);
				std::cout << "Initial slab variance " << p.beta_spike_diff_factor << "x spike variance (beta params only)" << std::endl;
				i += 1;
			}

			if(strcmp(in_str, "--gam_spike_diff_factor") == 0) {
				p.gam_spike_diff_factor = std::stod(argv[i + 1]);
				std::cout << "Initial slab variance " << p.beta_spike_diff_factor << "x spike variance (gamma params only)" << std::endl;
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

			if(strcmp(in_str, "--random_seed") == 0) {
				p.random_seed = std::stoul(argv[i + 1]);
				i += 1;
			}

			if(strcmp(in_str, "--effects_prior_mog") == 0) {
				p.mode_mog_prior_beta = true;
				p.mode_mog_prior_gam = true;
				i += 0;
			}

			if(strcmp(in_str, "--mode_spike_slab") == 0) {
				p.mode_mog_prior_beta = false;
				p.mode_mog_prior_gam = false;
				i += 0;
			}

			if(strcmp(in_str, "--mode_mog_beta") == 0) {
				p.mode_mog_prior_beta = true;
				i += 0;
			}

			if(strcmp(in_str, "--drop_loco") == 0) {
				p.drop_loco = true;
			}

			if(strcmp(in_str, "--suppress_squared_env_removal") == 0) {
				p.mode_remove_squared_envs = false;
				i += 0;
			}

			if(strcmp(in_str, "--incl_squared_envs") == 0) {
				p.mode_remove_squared_envs = false;
				p.mode_incl_squared_envs = true;
				i += 0;
			}

			if(strcmp(in_str, "--mode_pve_est") == 0) {
				p.mode_pve_est = true;
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

			if(strcmp(in_str, "--pve_mog_weights") == 0) {
				p.mog_weights_file = argv[i + 1];
				check_file_exists(p.mog_weights_file);
				i += 1;
			}

			if(strcmp(in_str, "--mode_regress_out_covars") == 0) {
				p.use_vb_on_covars = false;
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

			if(strcmp(in_str, "--mode_debug") == 0) {
				p.mode_debug = true;
				p.verbose = true;
				i += 0;
			}

			if(strcmp(in_str, "--mode_empirical_bayes") == 0) {
				p.mode_empirical_bayes = true;
				p.mode_squarem = false;
				i += 0;
			}

			if(strcmp(in_str, "--mode_constant_hyps") == 0) {
				p.mode_empirical_bayes = false;
				p.mode_squarem = false;
				i += 0;
			}

			if(strcmp(in_str, "--mode_squarem") == 0) {
				p.mode_squarem = true;
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
				p.param_dump_interval = 50;
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
				p.bgen_file = argv[i + 1];                                                                                                                                                 // bgen file
				p.bgi_file = p.bgen_file + ".bgi";

				check_file_exists(p.bgen_file);
				check_file_exists(p.bgi_file);
				i += 1;
			}

			if(strcmp(in_str, "--pheno") == 0) {
				check_counts(in_str, i, 1, argc);
				p.pheno_file = argv[i + 1];                                                                                                                                                 // pheno file
				check_file_exists(p.pheno_file);
				i += 1;
			}

			if(strcmp(in_str, "--recombination_map") == 0) {
				check_counts(in_str, i, 1, argc);
				p.recombination_file = argv[i + 1];                                                                                                                                                 // pheno file
				check_file_exists(p.recombination_file);
				i += 1;
			}

			if(strcmp(in_str, "--environment") == 0) {
				check_counts(in_str, i, 1, argc);
				p.interaction_analysis = true;
				p.env_file = argv[i + 1];
				check_file_exists(p.env_file);
				i += 1;
			}

			if(strcmp(in_str, "--environment_weights") == 0) {
				check_counts(in_str, i, 1, argc);
				p.env_coeffs_file = argv[i + 1];
				check_file_exists(p.env_coeffs_file);
				i += 1;
			}

			if(strcmp(in_str, "--param_dump_interval") == 0) {
				check_counts(in_str, i, 1, argc);
				p.param_dump_interval = std::stol(argv[i + 1]);
				i += 1;
			}

			if(strcmp(in_str, "--resume_from_param_dump") == 0) {
				check_counts(in_str, i, 1, argc);
				std::string prefix(argv[i + 1]);
				p.env_coeffs_file = prefix + "_env.out.gz";
				p.covar_coeffs_file = prefix + "_covars.out.gz";
				p.vb_init_file = prefix + "_latent_snps.out.gz";
				p.hyps_grid_file = prefix + "_hyps.out.gz";
				check_file_exists(p.env_coeffs_file);
				check_file_exists(p.covar_coeffs_file);
				check_file_exists(p.vb_init_file);
				check_file_exists(p.hyps_grid_file);
				i += 1;
			}

			if(strcmp(in_str, "--covar") == 0) {
				check_counts(in_str, i, 1, argc);
				p.covar_file = argv[i + 1];
				check_file_exists(p.covar_file);
				i += 1;
			}

			if(strcmp(in_str, "--snpwise_scan") == 0) {
				check_counts(in_str, i, 1, argc);
				p.snpstats_file = argv[i + 1];
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
				p.hyps_grid_file = argv[i + 1];                                                                                                                                                 // covar file
				check_file_exists(p.hyps_grid_file);
				i += 1;
			}

			if(strcmp(in_str, "--r1_hyps_grid") == 0) {
				check_counts(in_str, i, 1, argc);
				p.r1_hyps_grid_file = argv[i + 1];                                                                                                                                                 // covar file
				check_file_exists(p.r1_hyps_grid_file);
				i += 1;
			}

			if(strcmp(in_str, "--r1_probs_grid") == 0) {
				check_counts(in_str, i, 1, argc);
				p.r1_probs_grid_file = argv[i + 1];                                                                                                                                                 // covar file
				check_file_exists(p.r1_probs_grid_file);
				i += 1;
			}

			if(strcmp(in_str, "--hyps_probs") == 0) {
				check_counts(in_str, i, 1, argc);
				p.hyps_probs_file = argv[i + 1];                                                                                                                                                 // covar file
				check_file_exists(p.hyps_probs_file);
				i += 1;
			}

			if(strcmp(in_str, "--vb_init") == 0) {
				check_counts(in_str, i, 1, argc);
				p.vb_init_file = argv[i + 1];                                                                                                                                                 // covar file
				check_file_exists(p.vb_init_file);
				i += 1;
			}

			if(strcmp(in_str, "--dxteex") == 0) {
				check_counts(in_str, i, 1, argc);
				p.dxteex_file = argv[i + 1];                                                                                                                                                 // covar file
				check_file_exists(p.dxteex_file);
				i += 1;
			}

			// Filters
			if(strcmp(in_str, "--maf") == 0) {
				check_counts(in_str, i, 1, argc);
				p.maf_lim = true;
				p.min_maf = std::stod(argv[i + 1]);                                                                                                                                                 // bgen file
				i += 1;
			}

			if(strcmp(in_str, "--info") == 0) {
				check_counts(in_str, i, 1, argc);
				p.info_lim = true;
				p.min_info = std::stod(argv[i + 1]);                                                                                                                                                 // bgen file
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
				p.n_bgen_thread = std::min(4, atoi(argv[i + 1]));
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
				p.incl_sids_file = argv[i + 1];                                                                                                                                                 // include sample ids file
				check_file_exists(p.incl_sids_file);
				i += 1;
			}

			if(strcmp(in_str, "--incl_rsids") == 0) {
				check_counts(in_str, i, 1, argc);
				p.select_snps = true;
				p.incl_rsids_file = argv[i + 1];                                                                                                                                                 // include variant ids file
				check_file_exists(p.incl_rsids_file);
				i += 1;
			}

			if(strcmp(in_str, "--rsid") == 0) {
				check_counts(in_str, i, 1, argc);
				p.select_rsid = true;
				int jj = i+1;
				while(jj < argc) {
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
				p.chunk_size = std::stoi(argv[i + 1]);                                                                                                                                                 // bgen file
				i += 1;
			}

			if(strcmp(in_str, "--genetic_confounders") == 0) {
				check_counts(in_str, i, 1, argc);
				int jj = i+1;
				while(jj < argc) {
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

	// mode_vb specific options
	bool has_bgen = p.bgen_file != "NULL";
	bool has_out = p.out_file != "NULL";
	bool has_pheno = p.pheno_file != "NULL";
	bool has_all = (has_pheno && has_out && has_bgen);
	if(!has_all) {
		std::cout << "ERROR: bgen, pheno and out filepaths should all be ";
		std::cout << "provided in conjunction with --mode_vb." << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if(p.env_file == "NULL" && p.env_coeffs_file != "NULL") {
		std::cout << "WARNING: --environment_weights will be ignored as no --environment provided" << std::endl;
	}
}

void check_file_exists(const std::string &filename) {
	// Throw error if given file does not exist.
	// NB: Doesn't check if file is empty etc.
//	struct stat buf;
//	if(stat(filename.c_str(), &buf) != 0){
//		std::cout << "File " << filename << " does not exist" << std::endl;
//		throw std::runtime_error("ERROR: file does not exist");
//	}
	if(!boost::filesystem::exists( filename )) {
		std::cout << "File " << filename << " does not exist" << std::endl;
		throw std::runtime_error("ERROR: file does not exist");
	}
}
