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
#include <boost/filesystem.hpp>

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
	std::cout << "- Data encoded as double" << std::endl;
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
	    ("hyps-grid", "Path to initial hyperparameters values", cxxopts::value<std::string>(p.hyps_grid_file))
	    ("environment-weights", "Path to initial environment weights", cxxopts::value<std::string>(p.env_coeffs_file))
	    ("out", "Filepath to output", cxxopts::value<std::string>(p.out_file))
	    ("streamOut", "Output file for tests on imputed variants", cxxopts::value<std::string>(p.assocOutFile))
	    ("suppress-squared-env-removal", "QC: Suppress test for significant squared environmental effects (SQE)")
	    ("incl-squared-envs", "QC: Include significant squared environmental effects (SQE) as covariates")
	;

	options.add_options("VB")
	    ("VB", "Run VB algorithm.")
	    ("VB-ELBO-thresh", "Convergence threshold for VB convergence.", cxxopts::value<double>(p.elbo_tol))
	    ("VB-squarem", "Use SQUAREM algorithm for hyperparameter updates (on by default).")
	    ("VB-varEM", "Maximise ELBO wrt hyperparameters for hyperparameter updates.")
	    ("VB-constant-hyps", "Keep hyperparameters constant.")
	    ("VB-iter-max", "Maximum number of VB iterations (default: 10000)", cxxopts::value<long>(p.vb_iter_max))
	    ("resume-from-state", "For use when resuming VB algorithm from previous run.", cxxopts::value<std::string>(p.resume_prefix))
	    ("state-dump-interval", "Save VB parameter state to file every N iterations (default: None)", cxxopts::value<long>(p.param_dump_interval))
	    ("dxteex", "Optional flag to pass precomputed dXtEEX array.", cxxopts::value<std::string>(p.dxteex_file))
	;

	options.add_options("Assoc")
		("singleSnpStats", "Compute SNP association tests", cxxopts::value<bool>(p.mode_calc_snpstats))
		("resid-pheno", "Residualised phenotypes to compute association tests on. Inherited from VB algorithm if this is run first", cxxopts::value<std::string>(p.resid_loco_file))
	;

	options.add_options("RHE")
	    ("RHEreg", "Run randomised HE-regression algorithm", cxxopts::value<bool>())
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
	    ("incl-rsids", "Exclude all SNPs whose RSID is not in the given file.", cxxopts::value<std::string>(p.incl_rsids_file))
	    ("incl-sample-ids", "Exclude all samples whose sample ID is not in the BGEN file.", cxxopts::value<std::string>(p.incl_sids_file))
	    ("range", "Genomic range in format chr:start-end", cxxopts::value<std::string>())
	    ("pheno-col-num", "Column number of the phenotype to use (if --pheno contains multiple phenotypes)", cxxopts::value<long>(p.pheno_col_num))
	;

	options.add_options("Internal")
	    ("spike-diff-factor", "", cxxopts::value<double>())
	    ("gam-spike-diff-factor", "", cxxopts::value<double>(p.gam_spike_diff_factor))
	    ("beta-spike-diff-factor", "", cxxopts::value<double>(p.beta_spike_diff_factor))
	    ("force-write-vparams", "", cxxopts::value<bool>(p.force_write_vparams))
	    ("covar-init", "Path to initial covariate weights", cxxopts::value<std::string>(p.covar_coeffs_file))
	    ("vb-init", "", cxxopts::value<std::string>(p.vb_init_file))
	    ("xtra-verbose", "")
	    ("snpwise-scan", "", cxxopts::value<std::string>(p.snpstats_file))
	    ("pve-mog-weights", "", cxxopts::value<std::string>(p.mog_weights_file))
	    ("rhe-random-vectors", "Random vectors used to calculate randomised matrix trace estimates. (Optional)", cxxopts::value<std::string>(p.rhe_random_vectors_file))
	    ("use-vb-on-covars", "")
	    ("keep-constant-variants", "", cxxopts::value<bool>(p.keep_constant_variants))
	    ("mode-debug", "")
	    ("raw-phenotypes", "")
	    ("chunk", "", cxxopts::value<long>(p.chunk_size))
	    ("high-mem", "", cxxopts::value<bool>())
	    ("low-mem", "", cxxopts::value<bool>())
	    ("joint-covar-update", "Perform batch update in VB algorithm when updating covariates", cxxopts::value<bool>(p.joint_covar_update))
	    ("min-alpha-diff", "", cxxopts::value<double>(p.alpha_tol))
	    ("vb-iter-start", "", cxxopts::value<long>(p.vb_iter_start))
	    ("effects-prior-mog", "")
	    ("mode-spike-slab", "")
	    ("main-chunk-size", "", cxxopts::value<unsigned int>(p.main_chunk_size))
	    ("gxe-chunk-size", "", cxxopts::value<unsigned int>(p.gxe_chunk_size))
	    ("min-spike-diff-factor", "", cxxopts::value<double>(p.min_spike_diff_factor))
	    ("mode-regress-out-covars", "QC: Regress covariates from phenotype instead of including in VB")
	    ("exclude-ones-from-env-sq", "", cxxopts::value<bool>(p.exclude_ones_from_env_sq))
	    ("mode-alternating-updates", "", cxxopts::value<bool>(p.mode_alternating_updates))
	    ("threads", "", cxxopts::value<unsigned int>(p.n_thread))
	    ("hyps-probs", "", cxxopts::value<std::string>(p.hyps_probs_file))
	    ("maxBytesPerRank", "Maximum number of bytes of RAM available on each partition when using MPI (Default: 16GB)",
	    cxxopts::value<long long>(p.maxBytesPerRank))
	    ("loso-window-size", "", cxxopts::value<long>(p.LOSO_window))
	    ("drop-loco", "", cxxopts::value<bool>(p.drop_loco))
	    ("init-weights-with-snpwise-scan", "", cxxopts::value<bool>(p.init_weights_with_snpwise_scan))
	    ("mode-pve-est", "Depreciated: Run RHE algorithm", cxxopts::value<bool>())
	    ("streamBgen-print-interval", "", cxxopts::value<long>(p.streamBgen_print_interval))
	    ("mode-dump-processed-data", "", cxxopts::value<bool>(p.mode_dump_processed_data))
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
			std::cout << options.help({"General", "VB", "Assoc", "RHE", "Other"}) << std::endl;
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
		if(opts.count("RHEreg-NM")) {
			p.mode_RHE = true;
		}
		if(opts.count("RHEreg-LM")) {
			p.mode_RHE = true;
		}
		if(opts.count("mode-pve-est")) {
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
			std::string filename = opts["mStreamBgen"].as<std::string>();
			boost_io::filtering_istream fg;
			fg.push(boost_io::file_source(filename));
			if (!boost::filesystem::exists(filename) || !fg) {
				throw std::runtime_error("Error when trying to open file: " + filename);
			}
			std::string line, s;
			while (getline(fg, line)) {
				std::stringstream ss(line);
				while (ss >> s) {
					p.streamBgenFiles.push_back(s);
				}
			}
			if (p.streamBgenFiles.empty()){
				throw std::runtime_error("No valid bgen files found in " + filename);
			}
		}
		if(opts.count("loso-window-size")) {
			p.drop_loco = true;
		}
		if (opts.count("high-mem")) {
			p.low_mem = false;
		}
		if (opts.count("high-mem")) {
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

		if(opts.count("VB")) {
			p.mode_vb = true;
		}
		if(opts.count("VB-varEM")) {
			p.mode_empirical_bayes = true;
			p.mode_squarem = false;
			p.mode_vb = true;
		}
		if(opts.count("VB-constant-hyps")) {
			p.mode_empirical_bayes = false;
			p.mode_squarem = false;
			p.mode_vb = true;
		}
		if(opts.count("VB-squarem")) {
			p.mode_squarem = true;
			p.mode_empirical_bayes = true;
			p.mode_vb = true;
		}
		if(opts.count("xtra-verbose")) {
			p.verbose = true;
			p.xtra_verbose = true;
			if(!opts.count("param_dump_interval")) {
				p.param_dump_interval = 50;
			}
		}

		if(opts.count("mode-debug")) {
			p.debug = true;
			p.verbose = true;
			p.xtra_verbose = true;
			if(!opts.count("param_dump_interval")) {
				p.param_dump_interval = 50;
			}
		}

		if(opts.count("mode-spike-slab")) {
			p.mode_mog_prior_beta = false;
			p.mode_mog_prior_gam = false;
		}

		if(opts.count("suppress-squared-env-removal")) {
			p.mode_remove_squared_envs = false;
		}

		if(opts.count("incl-squared-envs")) {
			p.mode_remove_squared_envs = false;
			p.mode_incl_squared_envs = true;
		}

		if(p.mog_weights_file != "NULL") {
			check_file_exists(p.mog_weights_file);
		}


		if(opts.count("spike-diff-factor")) {
			p.beta_spike_diff_factor = opts["spike-diff-factor"].as<double>();
			p.gam_spike_diff_factor = opts["spike-diff-factor"].as<double>();
			std::cout << "Initial slab variance " << p.beta_spike_diff_factor << "x spike variance" << std::endl;
		}

		if(opts.count("beta-spike-diff-factor")) {
			std::cout << "Initial slab variance " << p.beta_spike_diff_factor << "x spike variance (beta params only)" << std::endl;
		}

		if(opts.count("gam-spike-diff-factor")) {
			std::cout << "Initial slab variance " << p.beta_spike_diff_factor << "x spike variance (gamma params only)" << std::endl;
		}

		if(opts.count("min-spike-diff-factor")) {
			p.min_spike_diff_set = true;
			std::cout << "Slab variance constrained to atleast " << p.min_spike_diff_factor << "x spike variance" << std::endl;
		}

		if(opts.count("effects-prior-mog")) {
			p.mode_mog_prior_beta = true;
			p.mode_mog_prior_gam = true;
		}

		if(opts.count("mode-regress-out-covars")) {
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

		if(opts.count("min-alpha-diff")) {
			p.alpha_tol_set_by_user = true;
			if(p.alpha_tol < 0) throw std::runtime_error("--min-alpha-diff must be positive.");
			std::cout << "--min-alpha-diff of " << p.alpha_tol << " entered." << std::endl;
		}

		if(opts.count("VB-ELBO-thresh")) {
			p.elbo_tol_set_by_user = true;
			if(p.elbo_tol < 0) throw std::runtime_error("--VB-ELBO-thresh must be positive.");
		}

		if(opts.count("incl-sample-ids")) {
			check_file_exists(p.incl_sids_file);
		}

		if(opts.count("incl-rsids")) {
			check_file_exists(p.incl_rsids_file);
		}
	} catch (const cxxopts::OptionException& e) {
		std::cout << "Error parsing options: " << e.what() << std::endl;
		throw std::runtime_error("");
	}

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

		std::string index = std::regex_replace(p.resume_prefix, std::regex(".*it([0-9]+).*"), std::string("$1"));
		std::cout << "NB: Will resume from previous parameter state";
		if (p.vb_iter_start == 0) {
			try {
				p.vb_iter_start = std::stol(index) + 1;
				std::cout << " (incrementing vb counter to " << p.vb_iter_start << ")";
			} catch (const std::invalid_argument& ia) {
			}
		}
		std::cout << std::endl;
	}

	// mode_vb specific options
	if(p.mode_vb) {
		bool has_bgen = p.bgen_file != "NULL" || !p.streamBgenFiles.empty();
		bool has_out = p.out_file != "NULL";
		bool has_pheno = p.pheno_file != "NULL";
		bool has_all = (has_pheno && has_out && has_bgen);
		if(!has_all) {
			std::cout << "ERROR: bgen, pheno and out filepaths should all be ";
			std::cout << "provided in conjunction with --VB." << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	if(p.mode_calc_snpstats) {
		bool has_bgen = p.bgen_file != "NULL" || p.streamBgenFiles.size() > 0;
		bool has_out = p.out_file != "NULL";
		bool has_pheno = (p.pheno_file != "NULL" && p.mode_vb) ||
						 (p.resume_prefix != "NULL" && p.bgen_file != "NULL") ||
						 p.resid_loco_file != "NULL";
		bool has_all = (has_pheno && has_out && has_bgen);
		if(!has_all) {
			std::cout << "ERROR: bgen, pheno and out filepaths should all be ";
			std::cout << "provided in conjunction with --singleSnpStats" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	if(p.env_file == "NULL" && p.env_coeffs_file != "NULL") {
		std::cout << "WARNING: --environment-weights will be ignored as no --environment provided" << std::endl;
	}
}

void check_file_exists(const std::string &filename) {
	// Throw error if given file does not exist.
	// NB: Doesn't check if file is empty etc.
	if(!boost::filesystem::exists(filename)) {
		std::cout << "File " << filename << " does not exist" << std::endl;
		throw std::runtime_error("ERROR: file does not exist");
	}
}
