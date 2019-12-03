//
// Created by kerin on 2019-03-01.
//

#include "parse_arguments.hpp"
#include "parameters.hpp"

#include "tools/eigen3.3/Dense"
#include "tools/cxxopts.hpp"

#include <iostream>
#include <iomanip>
#include <set>
#include <cstring>
#include <boost/filesystem.hpp>
#include <mpi.h>
#include <regex>
#include <stdexcept>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>

namespace boost_io = boost::iostreams;

void print_compilation_details(){
	std::cout << "Compilation details:" << std::endl;
#ifdef EIGEN_VECTORIZE
	std::cout << "- vectorization with SSE is ON" << std::endl;
#else
	std::cout << "- vectorization with SSE is OFF" << std::endl;
#endif

#ifdef DEBUG
	std::cout << "- in DEBUG mode" << std::endl;
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

#ifdef EIGEN_USE_MKL_ALL
	std::cout << "- compiled with Intel MKL backend" << std::endl;
#else
	std::cout << "- compiled with native Eigen backend" << std::endl;
#endif

#ifdef MPI_VERSION
	std::cout << "- compiled with openMPI" << std::endl;
#endif

#ifdef NDEBUG
	std::cout << "- WARNING: NDEBUG is defined which means assert statements won't work!" << std::endl;
#endif

}

void parse_arguments(parameters &p, int argc, char **argv) {

	cxxopts::Options options("LEMMA", "Software package to find GxE interactions with a common environmental profile."
	                         "\n\nLEMMA by default fits the bayesian whole genome regression model:"
	                         "\nY = C alpha + G beta + diag(eta) G gamma + espilon"
	                         "\nwhere eta = E w"
	                         "\n\nLEMMA then estimates the heritability of the G and GxE components from the following model:"
	                         "\nY sim N(C alpha, sigma_g GG^T / M + sigma_gxe Z(eta) Z(eta)^T / M + sigma I)"
	                         "\nwhere Z(eta) = diag(eta) G"
	                         "\nusing randomised HE regressionAfter estimating the posterior of parameters ");

	options.add_options("General")
	    ("verbose", "", cxxopts::value<bool>(p.verbose))
	    ("bgen", "Path to BGEN file. This must be indexed (eg. with BGENIX). By default this is stored in RAM using O(NM) bytes.", cxxopts::value<std::string>(p.bgen_file))
	    ("streamBgen", "Path to BGEN file. This must be indexed (eg. with BGENIX). This can be used with --RHEreg or --singleSnpStats.", cxxopts::value<std::string>())
	    ("mStreamBgen", "Text file containing paths to multiple BGEN files. They must all be indexed (eg. with BGENIX). This can be used with --RHE or --mode_calc_snpstats.", cxxopts::value<std::string>())
	    ("pheno", "Path to phenotype file. Must have a header and have the same number of rows as samples in the BGEN file.", cxxopts::value<std::string>(p.pheno_file))
	    ("covar", "Path to file of covariates. Must have a header and have the same number of rows as samples in the BGEN file.", cxxopts::value<std::string>(p.covar_file))
	    ("environment", "Path to file of environmental variables. Must have a header and have the same number of rows as samples in the BGEN file.", cxxopts::value<std::string>(p.env_file))
	    ("hyps_grid", "Path to initial hyperparameters values", cxxopts::value<std::string>(p.hyps_grid_file))
	    ("environment_weights", "Path to initial environment weights", cxxopts::value<std::string>(p.env_coeffs_file))
	    ("out", "Filepath to output", cxxopts::value<std::string>(p.out_file))
	    ("streamOut", "Output file for tests on imputed variants", cxxopts::value<std::string>(p.streamBgenOutFile))
	    ("suppress_squared_env_removal", "QC: Suppress test for significant squared environmental effects (SQE)")
	    ("incl_squared_envs", "QC: Include significant squared environmental effects (SQE) as covariates")
	;

	options.add_options("VB")
			("VB", "Run VB algorithm.")
			("singleSnpStats", "Compute SNP association tests", cxxopts::value<bool>(p.mode_calc_snpstats))
			("VB-ELBO-thresh", "Convergence threshold for VB convergence.", cxxopts::value<double>(p.elbo_tol))
			("VB-squarem", "Use SQUAREM algorithm for hyperparameter updates (on by default).")
			("VB-varEM", "Maximise ELBO wrt hyperparameters for hyperparameter updates.")
			("VB-constant-hyps", "Keep hyperparameters constant.")
			("VB-iter-max", "Maximum number of VB iterations (default: 10000)", cxxopts::value<long>(p.vb_iter_max))
			("resume_from_param_dump", "For use when resuming VB algorithm from previous run.", cxxopts::value<std::string>(p.resume_prefix))
			("param_dump_interval", "Save VB state to file every N iterations (default: None)", cxxopts::value<long>(p.param_dump_interval))
			("dxteex", "Optional flag to pass precomputed dXtEEX array.", cxxopts::value<std::string>(p.dxteex_file))
			;

	options.add_options("RHE")
			("RHEreg", "Run randomised HE-regression algorithm", cxxopts::value<bool>())
			("RHEreg-fast", "Run randomised HE-regression algorithm without centering and scaling genotype matrix (unnecessary with modified h2 conversion)")
			("RHEreg-groups", "Text file containing group of each SNP for use with multicomponent randomised HE regression", cxxopts::value<std::string>())
			("mRHEreg-groups", "Text file to paths of --RHE-groups files. Each should correspond to the bgen files given in --mStreamBgen", cxxopts::value<std::string>())
			("n-RHEreg-samples", "Number of random vectors used in RHE algorithm", cxxopts::value<long>(p.n_pve_samples))
			("n-RHEreg-jacknife", "Number of jacknife samples used in RHE algorithm", cxxopts::value<long>(p.n_jacknife))
			("random-seed", "Seed used to draw random vectors in RHE algorithm (default: random)",
			 cxxopts::value<unsigned int>(p.random_seed))
			("extra-pve-covar", "Covariables in addition to those provided to --VB (Eg. principle components).", cxxopts::value<std::string>(p.extra_pve_covar_file))
			;

	options.add_options("Filtering")
			("maf", "Exclude all SNPs whose minor allele frequency is below this threshold", cxxopts::value<double>(p.min_maf))
			("info", "Exclude all SNPs whose IMPUTE info is below this threshold", cxxopts::value<double>(p.min_info))
			("incl_rsids", "Exclude all SNPs whose RSID is not in the given file.", cxxopts::value<std::string>(p.incl_rsids_file))
			("incl_sample_ids", "Exclude all samples whose sample ID is not in the BGEN file.", cxxopts::value<std::string>(p.incl_sids_file))
			("range", "Genomic range in format chr:start-end", cxxopts::value<std::string>())
			;

	options.add_options("Internal")
			("spike_diff_factor", "", cxxopts::value<double>())
			("gam_spike_diff_factor", "", cxxopts::value<double>(p.gam_spike_diff_factor))
			("beta_spike_diff_factor", "", cxxopts::value<double>(p.beta_spike_diff_factor))
	    ("covar_init", "Path to initial covariate weights", cxxopts::value<std::string>(p.covar_coeffs_file))
	    ("vb_init", "", cxxopts::value<std::string>(p.vb_init_file))
	    ("xtra_verbose", "")
	    ("snpwise_scan", "", cxxopts::value<std::string>(p.snpstats_file))
	    ("pve_mog_weights", "", cxxopts::value<std::string>(p.mog_weights_file))
	    ("rhe_random_vectors", "Random vectors used to calculate randomised matrix trace estimates. (Optional)", cxxopts::value<std::string>(p.rhe_random_vectors_file))
	    ("use_vb_on_covars", "")
	    ("keep_constant_variants", "", cxxopts::value<bool>(p.keep_constant_variants))
	    ("mode_debug", "")
	    ("raw_phenotypes", "")
	    ("chunk", "", cxxopts::value<long>(p.chunk_size))
	    ("high_mem", "", cxxopts::value<bool>())
	    ("low_mem", "", cxxopts::value<bool>())
	    ("joint_covar_update", "Perform batch update in VB algorithm when updating covariates", cxxopts::value<bool>(p.joint_covar_update))
	    ("min_alpha_diff", "", cxxopts::value<double>(p.alpha_tol))
	    ("vb_iter_start", "", cxxopts::value<long>(p.vb_iter_start))
	    ("effects_prior_mog", "")
	    ("mode_spike_slab", "")
	    ("main_chunk_size", "", cxxopts::value<unsigned int>(p.main_chunk_size))
	    ("gxe_chunk_size", "", cxxopts::value<unsigned int>(p.gxe_chunk_size))
	    ("min_spike_diff_factor", "", cxxopts::value<double>(p.min_spike_diff_factor))
	    ("mode_regress_out_covars", "QC: Regress covariates from phenotype instead of including in VB")
	    ("exclude_ones_from_env_sq", "", cxxopts::value<bool>(p.exclude_ones_from_env_sq))
	    ("mode_alternating_updates", "", cxxopts::value<bool>(p.mode_alternating_updates))
	    ("threads", "", cxxopts::value<unsigned int>(p.n_thread))
	    ("hyps_probs", "", cxxopts::value<std::string>(p.hyps_probs_file))
	    ("maxBytesPerRank", "Maximum number of bytes of RAM available on each partition when using MPI (Default: 16GB)",
	    cxxopts::value<long long>(p.maxBytesPerRank))
	    ("loso_window_size", "", cxxopts::value<long>(p.LOSO_window))
	    ("drop_loco", "", cxxopts::value<bool>(p.drop_loco))
	    ("init_weights_with_snpwise_scan", "", cxxopts::value<bool>(p.init_weights_with_snpwise_scan))
	    ("mode_pve_est", "Depreciated: Run RHE algorithm", cxxopts::value<bool>())
	    ("streamBgen-print-interval", "", cxxopts::value<long>(p.streamBgen_print_interval))
	    ("mode_dump_processed_data", "", cxxopts::value<bool>(p.mode_dump_processed_data))
	    ("RHEreg-NM", "", cxxopts::value<bool>(p.mode_RHEreg_NM))
	    ("RHEreg-LM", "", cxxopts::value<bool>(p.mode_RHEreg_LM))
	    ("NM-max-iter", "", cxxopts::value<long>(p.nelderMead_max_iter))
	    ("LM-max-iter", "", cxxopts::value<long>(p.levenburgMarquardt_max_iter))
	    ("LM-random-starts", "", cxxopts::value<long>(p.n_LM_starts))
	;

	options.add_options("Other")
	    ("h, help", "Print help page")
	;

	try{
		auto opts = options.parse(argc, argv);
		auto args = opts.arguments();

		if (opts.count("help")) {
			std::cout << options.help({"General", "VB", "RHE", "Other"}) << std::endl;
			std::exit(0);
		} else {
			print_compilation_details();
			std::cout << std::endl << "Commandline arguments:" << std::endl;
			std::cout << argv[0] << " \\" << std::endl;
			for (auto keyvalue : args) {
				std::cout << "\t--" << keyvalue.key() << "=" << keyvalue.value() <<" \\" << std::endl;
			}
			std::cout << std::endl;
		}
		if(opts.count("RHEreg-fast")) {
			p.mode_RHE = true;
			p.mode_RHE_fast = true;
		}
		if(opts.count("RHEreg")) {
			p.mode_RHE = true;
		}
		if(opts.count("RHEreg-NM")){
			p.mode_RHE = true;
		}
		if(opts.count("RHEreg-LM")){
			p.mode_RHE = true;
		}
		if(opts.count("mode_pve_est")) {
			p.mode_RHE = true;
		}
		if(opts.count("RHEreg-groups")) {
			p.RHE_multicomponent = true;
			p.RHE_groups_files.push_back(opts["RHEreg-groups"].as<std::string>());
		}
		if(opts.count("mRHEreg-groups")) {
			p.RHE_multicomponent = true;
			boost_io::filtering_istream fg;
			std::string filename = opts["mRHEreg-groups"].as<std::string>();
			fg.push(boost_io::file_source(filename));
			if (!fg) {
				std::cout << "ERROR: " << filename << " not opened." << std::endl;
				std::exit(EXIT_FAILURE);
			}
			std::string line, s;
			while (getline(fg, line)) {
				std::stringstream ss(line);
				while (ss >> s) {
					p.RHE_groups_files.push_back(s);
				}
			}
		}

		if(opts.count("streamBgen")) {
			p.streamBgenFiles.push_back(opts["streamBgen"].as<std::string>());
		}
		if(opts.count("mStreamBgen")) {
			boost_io::filtering_istream fg;
			std::string filename = opts["mStreamBgen"].as<std::string>();
			fg.push(boost_io::file_source(filename));
			if (!fg) {
				std::cout << "ERROR: " << filename << " not opened." << std::endl;
				std::exit(EXIT_FAILURE);
			}
			std::string line, s;
			while (getline(fg, line)) {
				std::stringstream ss(line);
				while (ss >> s) {
					p.streamBgenFiles.push_back(s);
				}
			}
		}
		if(opts.count("loso_window_size")) {
			p.drop_loco = true;
		}
		if (opts.count("high_mem")) {
			p.low_mem = false;
		}
		if (opts.count("high_mem")) {
			p.low_mem = false;
		}
		if(p.pheno_file != "NULL") {
			check_file_exists(p.pheno_file);
		}
		if(p.extra_pve_covar_file != "NULL") {
			check_file_exists(p.extra_pve_covar_file);
		}
		if(p.dxteex_file != "NULL") {
			check_file_exists(p.dxteex_file);
		}
		if(p.vb_init_file != "NULL") {
			check_file_exists(p.vb_init_file);
		}
		if(p.covar_coeffs_file != "NULL") {
			check_file_exists(p.covar_coeffs_file);
		}
		if(p.hyps_grid_file != "NULL") {
			check_file_exists(p.hyps_grid_file);
		}
		if(p.snpstats_file != "NULL") {
			check_file_exists(p.snpstats_file);
		}
		if(p.covar_file != "NULL") {
			check_file_exists(p.covar_file);
		}
		if(p.env_file != "NULL") {
			check_file_exists(p.env_file);
			p.interaction_analysis = true;
		}
		if(p.bgen_file != "NULL") {
			p.bgi_file = p.bgen_file + ".bgi";
			check_file_exists(p.bgen_file);
			check_file_exists(p.bgi_file);
		}
		for (auto streamBgenFile : p.streamBgenFiles) {
			std::string bgi = streamBgenFile + ".bgi";
			check_file_exists(streamBgenFile);
			check_file_exists(bgi);
			p.streamBgiFiles.push_back(bgi);
		}
		if(opts.count("range")) {
			auto ss = opts["range"].as<std::string>();
			p.range = true;
			p.range_chr = ss.substr(0, ss.find(':'));
			p.range_start = std::atoi(ss.substr(ss.find(':')+1, ss.find('-')).c_str());
			p.range_end = std::atoi(ss.substr(ss.find('-')+1, ss.size()).c_str());
		}

		if(opts.count("VB-varEM")) {
			p.mode_empirical_bayes = true;
			p.mode_squarem = false;
		}

		if(opts.count("VB-constant-hyps")) {
			p.mode_empirical_bayes = false;
			p.mode_squarem = false;
		}

		if(opts.count("VB-squarem")) {
			p.mode_squarem = true;
			p.mode_empirical_bayes = true;
		}

		if(opts.count("xtra_verbose")) {
			p.verbose = true;
			p.xtra_verbose = true;
			if(!opts.count("param_dump_interval")) {
				p.param_dump_interval = 50;
			}
		}

		if(opts.count("mode_debug")) {
			p.debug = true;
			p.verbose = true;
			p.xtra_verbose = true;
			if(!opts.count("param_dump_interval")) {
				p.param_dump_interval = 50;
			}
		}

		if(opts.count("mode_spike_slab")) {
			p.mode_mog_prior_beta = false;
			p.mode_mog_prior_gam = false;
		}

		if(opts.count("suppress_squared_env_removal")) {
			p.mode_remove_squared_envs = false;
		}

		if(opts.count("incl_squared_envs")) {
			p.mode_remove_squared_envs = false;
			p.mode_incl_squared_envs = true;
		}

		if(p.mog_weights_file != "NULL") {
			check_file_exists(p.mog_weights_file);
		}


		if(opts.count("spike_diff_factor")) {
			p.beta_spike_diff_factor = opts["spike_diff_factor"].as<double>();
			p.gam_spike_diff_factor = opts["spike_diff_factor"].as<double>();
			std::cout << "Initial slab variance " << p.beta_spike_diff_factor << "x spike variance" << std::endl;
		}

		if(opts.count("beta_spike_diff_factor")) {
			std::cout << "Initial slab variance " << p.beta_spike_diff_factor << "x spike variance (beta params only)" << std::endl;
		}

		if(opts.count("gam_spike_diff_factor")) {
			std::cout << "Initial slab variance " << p.beta_spike_diff_factor << "x spike variance (gamma params only)" << std::endl;
		}

		if(opts.count("min_spike_diff_factor")) {
			p.min_spike_diff_set = true;
			std::cout << "Slab variance constrained to atleast " << p.min_spike_diff_factor << "x spike variance" << std::endl;
		}

		if(opts.count("VB")) {
			p.mode_vb = true;
			p.mode_calc_snpstats = true;
		}

		if(opts.count("effects_prior_mog")) {
			p.mode_mog_prior_beta = true;
			p.mode_mog_prior_gam = true;
		}

		if(opts.count("mode_regress_out_covars")) {
			p.use_vb_on_covars = false;
		}

		if(opts.count("maf")) {
			p.maf_lim = true;
		}

		if(opts.count("info")) {
			p.info_lim = true;
		}

		if(opts.count("threads")) {
			if(p.n_thread < 1) throw std::runtime_error("--threads must be positive.");
		}

		if(opts.count("min_alpha_diff")) {
			p.alpha_tol_set_by_user = true;
			if(p.alpha_tol < 0) throw std::runtime_error("--min_alpha_diff must be positive.");
			std::cout << "--min_alpha_diff of " << p.alpha_tol << " entered." << std::endl;
		}

		if(opts.count("VB-ELBO-thresh")) {
			p.elbo_tol_set_by_user = true;
			if(p.elbo_tol < 0) throw std::runtime_error("--VB-ELBO-thresh must be positive.");
			std::cout << "--VB-ELBO-thresh of " << p.elbo_tol << " entered." << std::endl;
		}

		if(opts.count("incl_sample_ids")) {
			check_file_exists(p.incl_sids_file);
		}

		if(opts.count("incl_rsids")) {
			check_file_exists(p.incl_rsids_file);
		}
	} catch (const cxxopts::OptionException& e) {
		std::cout << "Error parsing options: " << e.what() << std::endl;
		throw std::runtime_error("");
	}

	// char *in_str;
	// int i;
	// std::set<std::string> option_list {
	//  "--bgen",
	//  "--pheno",
	//  "--covar",
	//  "--recombination_map",
	//  "--environment",
	//  "--environment_weights",
	//  "--snpwise_scan",
	//  "--chunk",
	//  "--range",
	//  "--maf",
	//  "--info",
	//  "--out",
	//  "--mode_vb",
	//  "--mode_empirical_bayes",
	//  "--mode_squarem",
	//  "--mode_constant_hyps",
	//  "--effects_prior_mog",
	//  "--mode_spike_slab",
	//  "--use_vb_on_covars",
	//  "--drop_loco",
	//  "--mode_regress_out_covars",
	//  "--threads",
	//  "--low_mem",
	//  "--high_mem",
	//  "--incl_sample_ids",
	//  "--incl_rsids",
	//  "--rsid",
	//  "--no_geno_check",
	//  "--genetic_confounders",
	//  "--r1_hyps_grid",
	//  "--r1_probs_grid",
	//  "--min_elbo_diff",
	//  "--min_alpha_diff",
	//  "--hyps_grid",
	//  "--hyps_probs",
	//  "--vb_init",
	//  "--verbose",
	//  "--xtra_verbose",
	//  "--keep_constant_variants",
	//  "--force_round1",
	//  "--mode_alternating_updates",
	//  "--mode_no_gxe",
	//  "--vb_iter_max",
	//  "--vb_iter_start",
	//  "--env_update_repeats",
	//  "--gamma_updates_thresh",
	//  "--init_weights_with_snpwise_scan",
	//  "--suppress_squared_env_removal",
	//  "--incl_squared_envs",
	//  "--resume_from_param_dump",
	//  "--dxteex",
	//  "--mode_mog_beta",
	//  "--mode_mog_gamma",
	//  "--mode_RHE",
	//  "--gxe_chunk_size",
	//  "--main_chunk_size",
	//  "--param_dump_interval",
	//  "--random_seed",
	//  "--mode_debug",
	//  "--pve_mog_weights",
	//  "--spike_diff_factor",
	//  "--beta_spike_diff_factor",
	//  "--gam_spike_diff_factor",
	//  "--min_spike_diff_factor",
	//  "--n_pve_samples",
	//  "--loso_window_size",
	//  "--mode_calc_snpstats",
	//  "--streamBgen",
	//  "--streamOut"
	// };
	//
	// std::set<std::string>::iterator set_it;
	// // for --version (& no other args), print splash screen and exit.
	// if (argc == 2 && strcmp(argv[1], "--version") == 0) {
	//  std::exit(EXIT_SUCCESS);
	// }
	//
	//
	// // read in and check option flags
	// for (i = 0; i < argc; i++) {
	//  in_str = argv[i];
	//  if (strcmp(in_str, "--version") == 0 || strcmp(in_str, "--help") == 0) {
	//      std::cout << "ERROR: flag '" << in_str << "' cannot be used with any other flags." << std::endl;
	//      std::exit(EXIT_FAILURE);
	//  }
	// }
	//
	// // Ensure some arguments only appear once
	// bool check_out = 0;
	//
	// for(i = 0; i < argc; i++) {
	//  if(*argv[i] == '-') {
	//      in_str = argv[i];
	//      set_it = option_list.find(in_str);
	//
	//      if(set_it == option_list.end()) {
	//          std::cout << "ERROR: flag '" << in_str <<
	//              "' not valid. Please refer to the manual for usage instructions." <<
	//              std::endl;
	//
	//          exit(EXIT_FAILURE);
	//      }
	//
	//      // flags with parameters should eat their arguments
	//      // & also make sure they don't go over argc
	//
	//      // Modes - a variety of different functionalities now included
	//      if(strcmp(in_str, "--gxe_chunk_size") == 0) {
	//          p.gxe_chunk_size = std::stoi(argv[i + 1]);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--main_chunk_size") == 0) {
	//          p.main_chunk_size = std::stoi(argv[i + 1]);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--env_update_repeats") == 0) {
	//          p.env_update_repeats = std::stoi(argv[i + 1]);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--init_weights_with_snpwise_scan") == 0) {
	//          p.init_weights_with_snpwise_scan = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--gamma_updates_thresh") == 0) {
	//          p.restrict_gamma_updates = true;
	//          p.gamma_updates_thresh = std::stod(argv[i + 1]);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--vb_iter_max") == 0) {
	//          p.vb_iter_max = std::stol(argv[i + 1]);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--vb_iter_start") == 0) {
	//          p.vb_iter_start = std::stol(argv[i + 1]);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--n_pve_samples") == 0) {
	//          p.n_pve_samples = std::stoi(argv[i + 1]);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--spike_diff_factor") == 0) {
	//          p.beta_spike_diff_factor = std::stod(argv[i + 1]);
	//          p.gam_spike_diff_factor = std::stod(argv[i + 1]);
	//          std::cout << "Initial slab variance " << p.beta_spike_diff_factor << "x spike variance" << std::endl;
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--beta_spike_diff_factor") == 0) {
	//          p.beta_spike_diff_factor = std::stod(argv[i + 1]);
	//          std::cout << "Initial slab variance " << p.beta_spike_diff_factor << "x spike variance (beta params only)" << std::endl;
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--gam_spike_diff_factor") == 0) {
	//          p.gam_spike_diff_factor = std::stod(argv[i + 1]);
	//          std::cout << "Initial slab variance " << p.beta_spike_diff_factor << "x spike variance (gamma params only)" << std::endl;
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--min_spike_diff_factor") == 0) {
	//          p.min_spike_diff_factor = std::stod(argv[i + 1]);
	//          p.min_spike_diff_set = true;
	//          std::cout << "Slab variance constrained to atleast " << p.min_spike_diff_factor << "x spike variance" << std::endl;
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--mode_vb") == 0) {
	//          p.mode_vb = true;
	//          p.mode_calc_snpstats = true;
	//      }
	//
	//      if(strcmp(in_str, "--mode_calc_snpstats") == 0) {
	//          p.mode_calc_snpstats = true;
	//      }
	//
	//      if(strcmp(in_str, "--random_seed") == 0) {
	//          p.random_seed = std::stoul(argv[i + 1]);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--effects_prior_mog") == 0) {
	//          p.mode_mog_prior_beta = true;
	//          p.mode_mog_prior_gam = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--mode_spike_slab") == 0) {
	//          p.mode_mog_prior_beta = false;
	//          p.mode_mog_prior_gam = false;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--mode_mog_beta") == 0) {
	//          p.mode_mog_prior_beta = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--drop_loco") == 0) {
	//          p.drop_loco = true;
	//      }
	//
	//      if(strcmp(in_str, "--loso_window_size") == 0) {
	//          p.LOSO_window = std::stol(argv[i + 1]);
	//          std::cout << "Leave out segment size: " << p.LOSO_window << std::endl;
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--suppress_squared_env_removal") == 0) {
	//          p.mode_remove_squared_envs = false;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--incl_squared_envs") == 0) {
	//          p.mode_remove_squared_envs = false;
	//          p.mode_incl_squared_envs = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--mode_RHE") == 0) {
	//          p.mode_RHE = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--mode_no_gxe") == 0) {
	//          p.mode_no_gxe = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--mode_mog_gamma") == 0) {
	//          p.mode_mog_prior_gam = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--use_vb_on_covars") == 0) {
	//          p.use_vb_on_covars = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--pve_mog_weights") == 0) {
	//          p.mog_weights_file = argv[i + 1];
	//          check_file_exists(p.mog_weights_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--mode_regress_out_covars") == 0) {
	//          p.use_vb_on_covars = false;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--mode_alternating_updates") == 0) {
	//          p.mode_alternating_updates = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--keep_constant_variants") == 0) {
	//          p.keep_constant_variants = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--force_round1") == 0) {
	//          p.user_requests_round1 = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--mode_debug") == 0) {
	//          p.mode_debug = true;
	//          p.verbose = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--mode_empirical_bayes") == 0) {
	//          p.mode_empirical_bayes = true;
	//          p.mode_squarem = false;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--mode_constant_hyps") == 0) {
	//          p.mode_empirical_bayes = false;
	//          p.mode_squarem = false;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--mode_squarem") == 0) {
	//          p.mode_squarem = true;
	//          p.mode_empirical_bayes = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--verbose") == 0) {
	//          p.verbose = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--xtra_verbose") == 0) {
	//          p.verbose = true;
	//          p.xtra_verbose = true;
	//          p.param_dump_interval = 50;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--low_mem") == 0) {
	//          p.low_mem = true;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--high_mem") == 0) {
	//          p.low_mem = false;
	//          i += 0;
	//      }
	//
	//      // Data inputs
	//      if(strcmp(in_str, "--bgen") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.bgen_file = argv[i + 1];
	//          p.bgi_file = p.bgen_file + ".bgi";
	//
	//          check_file_exists(p.bgen_file);
	//          check_file_exists(p.bgi_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--streamBgen") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.streamBgenFile = argv[i + 1];
	//          p.streamBgiFile = p.streamBgenFile + ".bgi";
	//
	//          check_file_exists(p.streamBgenFile);
	//          check_file_exists(p.streamBgiFile);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--streamOut") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.streamBgenOutFile = argv[i + 1];
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--pheno") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.pheno_file = argv[i + 1];
	//          check_file_exists(p.pheno_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--recombination_map") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.recombination_file = argv[i + 1];
	//          check_file_exists(p.recombination_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--environment") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.interaction_analysis = true;
	//          p.env_file = argv[i + 1];
	//          check_file_exists(p.env_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--environment_weights") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.env_coeffs_file = argv[i + 1];
	//          check_file_exists(p.env_coeffs_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--param_dump_interval") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.param_dump_interval = std::stol(argv[i + 1]);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--resume_from_param_dump") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.resume_prefix = argv[i + 1];
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--covar") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.covar_file = argv[i + 1];
	//          check_file_exists(p.covar_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--snpwise_scan") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.snpstats_file = argv[i + 1];
	//          check_file_exists(p.snpstats_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--out") == 0) {
	//          if (check_out == 1) {
	//              std::cout << "ERROR: flag '" << in_str << "' can only be provided once." << std::endl;
	//              exit(EXIT_FAILURE);
	//          }
	//          check_out = 1;
	//          check_counts(in_str, i, 1, argc);
	//          p.out_file = argv[i + 1];
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--hyps_grid") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.hyps_grid_file = argv[i + 1];
	//          check_file_exists(p.hyps_grid_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--r1_hyps_grid") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.r1_hyps_grid_file = argv[i + 1];
	//          check_file_exists(p.r1_hyps_grid_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--r1_probs_grid") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.r1_probs_grid_file = argv[i + 1];
	//          check_file_exists(p.r1_probs_grid_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--hyps_probs") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.hyps_probs_file = argv[i + 1];
	//          check_file_exists(p.hyps_probs_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--vb_init") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.vb_init_file = argv[i + 1];
	//          check_file_exists(p.vb_init_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--dxteex") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.dxteex_file = argv[i + 1];
	//          check_file_exists(p.dxteex_file);
	//          i += 1;
	//      }
	//
	//      // Filters
	//      if(strcmp(in_str, "--maf") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.maf_lim = true;
	//          p.min_maf = std::stod(argv[i + 1]);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--info") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.info_lim = true;
	//          p.min_info = std::stod(argv[i + 1]);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--range") == 0) {
	//          static bool check = 0;
	//          if (check == 1) {
	//              std::cout << "ERROR: flag '" << in_str << "' can only be provided once." << std::endl;
	//              exit(EXIT_FAILURE);
	//          }
	//          check = 1;
	//          check_counts(in_str, i, 3, argc);
	//          p.range = true;
	//          p.chr = argv[i + 1];
	//          p.range_start = atoi(argv[i + 2]);
	//          p.range_end = atoi(argv[i + 3]);
	//          i += 3;
	//      }
	//
	//      if(strcmp(in_str, "--threads") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.n_thread = atoi(argv[i + 1]);
	//          p.n_bgen_thread = std::min(4, atoi(argv[i + 1]));
	//          if(p.n_thread < 1) throw std::runtime_error("--threads must be positive.");
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--min_alpha_diff") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.alpha_tol_set_by_user = true;
	//          p.alpha_tol = atof(argv[i + 1]);
	//          if(p.alpha_tol < 0) throw std::runtime_error("--min_alpha_diff must be positive.");
	//          std::cout << "--min_alpha_diff of " << p.alpha_tol << " entered." << std::endl;
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--min_elbo_diff") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.elbo_tol_set_by_user = true;
	//          p.elbo_tol = atof(argv[i + 1]);
	//          if(p.elbo_tol < 0) throw std::runtime_error("--min_elbo_diff must be positive.");
	//          std::cout << "--min_elbo_diff of " << p.elbo_tol << " entered." << std::endl;
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--incl_sample_ids") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.incl_sids_file = argv[i + 1];
	//          check_file_exists(p.incl_sids_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--incl_rsids") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.select_snps = true;
	//          p.incl_rsids_file = argv[i + 1];
	//          check_file_exists(p.incl_rsids_file);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--rsid") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.select_rsid = true;
	//          int jj = i+1;
	//          while(jj < argc) {
	//              std::string arg_str(argv[jj]);
	//              if (arg_str.find("--") != std::string::npos) break;
	//              p.rsid.push_back(argv[jj]);
	//              jj++;
	//          }
	//          i += 1;
	//      }
	//
	//      // Other options
	//      if(strcmp(in_str, "--no_geno_check") == 0) {
	//          p.geno_check = false;
	//          i += 0;
	//      }
	//
	//      if(strcmp(in_str, "--chunk") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          p.chunk_size = std::stoi(argv[i + 1]);
	//          i += 1;
	//      }
	//
	//      if(strcmp(in_str, "--genetic_confounders") == 0) {
	//          check_counts(in_str, i, 1, argc);
	//          int jj = i+1;
	//          while(jj < argc) {
	//              std::string arg_str(argv[jj]);
	//              if (arg_str.find("--") != std::string::npos) break;
	//              p.gconf.push_back(argv[jj]);
	//              jj++;
	//          }
	//          p.n_gconf = jj - i - 1;
	//          i += 1;
	//      }
	//  }
	// }

	if(p.resume_prefix != "NULL") {
		assert(p.out_file != "NULL");
		std::string file_ext = p.out_file.substr(p.out_file.find('.'), p.out_file.size());

		p.vb_init_file = p.resume_prefix + "_latent_snps" + file_ext;
		p.hyps_grid_file = p.resume_prefix + "_hyps" + file_ext;
		check_file_exists(p.vb_init_file);
		check_file_exists(p.hyps_grid_file);

		p.env_coeffs_file = p.resume_prefix + "_env" + file_ext;
		if(!boost::filesystem::exists(p.env_coeffs_file)) {
			assert(p.env_file == "NULL");
			p.env_coeffs_file = "NULL";
		}

		p.covar_coeffs_file = p.resume_prefix + "_covars" + file_ext;
		if(!boost::filesystem::exists(p.covar_coeffs_file)) {
			p.covar_coeffs_file = "NULL";
		}

		std::string index = std::regex_replace(
			p.resume_prefix,
			std::regex(".*it([0-9]+).*"),
			std::string("$1"));
		std::cout << "Resuming from previously saved state" << std::endl;
		if (p.vb_iter_start == 0) {
			try {
				p.vb_iter_start = std::stol(index) + 1;
				std::cout << "Incrementing vb counter to " << p.vb_iter_start << std::endl;
			} catch (const std::invalid_argument& ia) {
			}
		}
	}

	// mode_vb specific options
	if(p.mode_vb) {
		bool has_bgen = p.bgen_file != "NULL";
		bool has_out = p.out_file != "NULL";
		bool has_pheno = p.pheno_file != "NULL";
		bool has_all = (has_pheno && has_out && has_bgen);
		if(!has_all) {
			std::cout << "ERROR: bgen, pheno and out filepaths should all be ";
			std::cout << "provided in conjunction with --mode_vb." << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	if(p.mode_calc_snpstats) {
		bool has_bgen = p.bgen_file != "NULL" || p.streamBgenFiles.size() > 0;
		bool has_out = p.out_file != "NULL";
		bool has_pheno = p.pheno_file != "NULL";
		bool has_all = (has_pheno && has_out && has_bgen);
		if(!has_all) {
			std::cout << "ERROR: bgen, pheno and out filepaths should all be ";
			std::cout << "provided in conjunction with --singleSnpStats" << std::endl;
			std::exit(EXIT_FAILURE);
		}
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
	if(!boost::filesystem::exists(filename)) {
		std::cout << "File " << filename << " does not exist" << std::endl;
		throw std::runtime_error("ERROR: file does not exist");
	}
}
