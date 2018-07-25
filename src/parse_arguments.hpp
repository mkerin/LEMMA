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
#include <regex>
#include <stdexcept>

void check_counts(const std::string& in_str, int i, int num, int argc);
void parse_arguments(parameters &p, int argc, char *argv[]);
void check_file_exists(const std::string& filename);

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
		"--environment",
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
		"--interaction",
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
		"--mode_approximate_residuals",
		"--min_residuals_diff",
		"--mode_sgd",
		"--sgd_delay",
		"--sgd_forgetting_rate",
		"--sgd_minibatch_size"
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
			if(strcmp(in_str, "--mode_sgd") == 0) {
				p.mode_sgd = true;
				i += 0;
			}

			if(strcmp(in_str, "--sgd_delay") == 0) {
				p.sgd_delay = std::stod(argv[i + 1]);
				p.sgd_delay_set = true;
				i += 1;
			}

			if(strcmp(in_str, "--sgd_forgetting_rate") == 0) {
				p.sgd_forgetting_rate = std::stod(argv[i + 1]);
				p.sgd_forgetting_rate_set = true;
				i += 1;
			}

			if(strcmp(in_str, "--sgd_minibatch_size") == 0) {
				p.sgd_minibatch_size = std::stol(argv[i + 1]);
				p.sgd_minibatch_size_set = true;
				i += 1;
			}

			if(strcmp(in_str, "--mode_vb") == 0) {
				p.mode_vb = true;
				i += 0;
			}

			if(strcmp(in_str, "--effects_prior_mog") == 0) {
				p.mode_mog_prior = true;
				i += 0;
			}

			if(strcmp(in_str, "--use_vb_on_covars") == 0) {
				p.use_vb_on_covars = true;
				i += 0;
			}

			if(strcmp(in_str, "--mode_approximate_residuals") == 0) {
				p.mode_approximate_residuals = true;
				i += 0;
			}

			if(strcmp(in_str, "--mode_alternating_updates") == 0) {
				p.mode_alternating_updates = true;
				i += 0;
			}

			if(strcmp(in_str, "--min_residuals_diff") == 0) {
				p.min_residuals_diff = std::stod(argv[i + 1]);
				i += 1;
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

			if(strcmp(in_str, "--raw_phenotypes") == 0) {
				p.scale_pheno = false;
				i += 0;
			}

			// Data inputs
			if(strcmp(in_str, "--bgen") == 0) {
				check_counts(in_str, i, 1, argc);
				p.bgen_file = argv[i + 1]; // bgen file

				std::regex reg_asterisk("\\*");
				if (std::regex_search(p.bgen_file, reg_asterisk)) {
					p.bgen_wildcard = true;
				} else {
					check_file_exists(p.bgen_file);
				}
				i += 1;
			}

			if(strcmp(in_str, "--pheno") == 0) {
				check_counts(in_str, i, 1, argc);
				p.pheno_file = argv[i + 1]; // pheno file
				check_file_exists(p.pheno_file);
				i += 1;
			}

			if(strcmp(in_str, "--environment") == 0) {
				check_counts(in_str, i, 1, argc);
				p.interaction_analysis = true;
				p.env_file = argv[i + 1]; // pheno file
				check_file_exists(p.env_file);
				i += 1;
			}

			if(strcmp(in_str, "--covar") == 0) {
				check_counts(in_str, i, 1, argc);
				p.covar_file = argv[i + 1]; // covar file
				check_file_exists(p.covar_file);
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
				p.start = atoi(argv[i + 2]);
				p.end = atoi(argv[i + 3]);
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

			if(strcmp(in_str, "--interaction") == 0) {
				check_counts(in_str, i, 1, argc);
				p.interaction_analysis = true;
				p.x_param_name = argv[i + 1]; // include sample ids file
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

	if(p.mode_approximate_residuals){
		std::cout << "Vector of residuals only updated if change in beta greater than:";
		std::cout << std::setprecision(10) << std::fixed << p.min_residuals_diff << std::endl;
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

	if(p.interaction_analysis && p.covar_file == "NULL"){
		throw std::runtime_error("ERROR: --covar must be provided if --interaction analysis is being run.");
	}

	if(p.mode_sgd && !(p.sgd_delay_set && p.sgd_forgetting_rate_set && p.sgd_minibatch_size_set)){
		throw std::runtime_error("ERROR: Must set forgetting rate, delay and minibatch size for mode sgd");
	}
	if(p.mode_sgd && p.sgd_minibatch_size <= 0){
		throw std::runtime_error("ERROR: Minibatch size must be greater than zero");
	}
}

#endif
