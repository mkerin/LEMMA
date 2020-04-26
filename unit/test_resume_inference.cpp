// tests-main.cpp
#include "catch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes_x2.hpp"
#include "../src/data.hpp"
#include "../src/hyps.hpp"


// Scenarios
char* case1a[] = { (char*) "prog",
	               (char*) "--mode-regress-out-covars",
	               (char*) "--VB-varEM",
				   (char*) "--random-seed", (char*) "1",
	               (char*) "--spike-diff-factor", (char*) "10000",
	               (char*) "--VB-iter-max", (char*) "10",
	               (char*) "--hyps-grid", (char*) "data/io_test/single_hyps_gxage.txt",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test1a.out"};

char* case1b[] = { (char*) "prog",
	               (char*) "--mode-regress-out-covars",
	               (char*) "--VB-varEM",
				   (char*) "--random-seed", (char*) "1",
	               (char*) "--spike-diff-factor", (char*) "10000",
	               (char*) "--VB-iter-max", (char*) "10",
	               (char*) "--vb-iter-start", (char*) "3",
	               (char*) "--resume-from-state",
	               (char*) "data/io_test/lemma_interim_files/test1a_dump_it2",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test1b.out"};

char* case1c[] = { (char*) "prog",
	               (char*) "--mode-regress-out-covars",
				   (char*) "--random-seed", (char*) "1",
	               (char*) "--singleSnpStats",
	               (char*) "--resume-from-state",
	               (char*) "data/io_test/lemma_interim_files/test1a_dump_it_converged",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test1c.out"};



TEST_CASE("Resume from multi-env + mog + emp_bayes"){
	parameters p;

	SECTION("Run to iter 10"){
		int argc = sizeof(case1a)/sizeof(case1a[0]);
		parse_arguments(p, argc, case1a);

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.calc_dxteex();
		data.set_vb_init();

		VBayesX2 VB(data);

		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
		SECTION("Ex4. Explicitly checking hyps") {
			// Set up for RunInnerLoop
			long n_grid = VB.hyps_inits.size();
			std::vector<Hyps> all_hyps = VB.hyps_inits;
			std::vector<VariationalParameters> all_vp;
			VB.setup_variational_params(all_hyps, all_vp);

			int round_index = 2;
			std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
			std::vector<std::vector< double > > logw_updates(n_grid);
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-92.2292775905));
			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-89.6317892411));
			VB.updateAllParams(2, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-88.4701165481));

			VbTracker tracker(p);
			tracker.init_interim_output(0,2, VB.n_effects, VB.n_covar, VB.n_env, VB.env_names, vp);
			tracker.dump_state("2", VB.n_samples, VB.n_covar, VB.n_var, VB.n_env,
			                   VB.n_effects, vp, hyps, VB.Y, VB.C, VB.X,
			                   VB.covar_names, VB.env_names, VB.sample_is_invalid, VB.sample_location);

			VB.updateAllParams(3, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-87.8793981688));
		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		VB.output_vb_results();
		SECTION("Check converged inference"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.8131749627));
		}

		SECTION("Check converged snp-stats"){
			CHECK(VB.vp_init.ym.array().abs().sum() == Approx(23.2450939216));
			// Compute snp-stats
			Eigen::MatrixXd neglogPvals, testStats;
			VB.compute_LOCO_pvals(VB.vp_init, neglogPvals, testStats);
			CHECK(neglogPvals(0,0) == Approx(0.261435434));
			CHECK(neglogPvals(0,1) == Approx(1.6047721233));
		}

	}

	SECTION("Resume from iter 2"){
		int argc = sizeof(case1b)/sizeof(case1b[0]);
		parse_arguments(p, argc, case1b);

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.calc_dxteex();
		data.set_vb_init();

		VBayesX2 VB(data);

		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
		SECTION("Ex4. Explicitly checking hyps") {
			long n_grid = VB.hyps_inits.size();
			std::vector<Hyps> all_hyps = VB.hyps_inits;
			std::vector<VariationalParameters> all_vp;
			VB.setup_variational_params(all_hyps, all_vp);

			int round_index = 2;
			std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
			std::vector<std::vector< double > > logw_updates(n_grid);
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			VB.updateAllParams(3, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-87.8793981839));
		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.8131749627));
		}
	}

	SECTION("Compute snp-stats from dump of converged params"){
		int argc = sizeof(case1c)/sizeof(case1c[0]);
		parse_arguments(p, argc, case1c);

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.set_vb_init();
		VBayesX2 VB(data);
		CHECK(VB.vp_init.ym.array().abs().sum() == Approx(23.2450939216));

		// Compute snp-stats
		Eigen::MatrixXd neglogPvals, testStats;
		VB.compute_LOCO_pvals(VB.vp_init, neglogPvals, testStats);
		CHECK(neglogPvals(0,0) == Approx(0.261435434));
		CHECK(neglogPvals(0,1) == Approx(1.6047721233));
	}
}


// Scenarios
char* case3a[] = { (char*) "prog",
	               (char*) "--VB-varEM",
	               (char*) "--VB-iter-max", (char*) "10",
				   (char*) "--random-seed", (char*) "1",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test3a.out"};

char* case3c[] = { (char*) "prog",
	               (char*) "--singleSnpStats",
				   (char*) "--random-seed", (char*) "1",
	               (char*) "--resume-from-state",
	               (char*) "data/io_test/lemma_interim_files/test3a_dump_it_converged",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test3c.out"};


TEST_CASE("Resume from multi-env + mog + emp_bayes + incl_covars"){
	parameters p;

	SECTION("Run to iter 10"){
		int argc = sizeof(case3a)/sizeof(case3a[0]);
		parse_arguments(p, argc, case3a);

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.calc_dxteex();
		data.set_vb_init();

		VBayesX2 VB(data);

		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Check converged inference"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.8429524528));
		}

		SECTION("Check converged snp-stats"){
			// Compute snp-stats
			Eigen::MatrixXd neglogPvals, testStats;
			VB.compute_LOCO_pvals(VB.vp_init, neglogPvals, testStats);
			CHECK(neglogPvals(0,0) == Approx(0.2138468423));
			CHECK(neglogPvals(0,1) == Approx(1.761355603));
			CHECK(neglogPvals(0,2) == Approx(3.2604753173));
		}
		VB.output_vb_results();
	}


	SECTION("Compute snp-stats from dump of converged params"){
		int argc = sizeof(case3c)/sizeof(case3c[0]);
		parse_arguments(p, argc, case3c);

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.set_vb_init();

		VBayesX2 VB(data);

		// Compute snp-stats
		Eigen::MatrixXd neglogPvals, testStats;
		VB.compute_LOCO_pvals(VB.vp_init, neglogPvals, testStats);
		CHECK(neglogPvals(0,0) == Approx(0.2138468423));
		CHECK(neglogPvals(0,1) == Approx(1.761355603));
		CHECK(neglogPvals(0,2) == Approx(3.2604753173));
	}
}
