// tests-main.cpp
#include "catch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes.hpp"
#include "../src/data.hpp"
#include "../src/hyps.hpp"
#include "../src/mpi_utils.hpp"


// Scenarios
char* case1a[] = { (char*) "prog",
	               (char*) "--VB-varEM",
	               (char*) "--mode-regress-out-covars",
	               (char*) "--random-seed", (char*) "1",
	               (char*) "--VB-iter-max", (char*) "10",
	               (char*) "--hyps-grid", (char*) "unit/data/single_hyps_gxage.txt",
	               (char*) "--pheno", (char*) "unit/data/pheno.txt",
	               (char*) "--environment", (char*) "unit/data/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "unit/data/n50_p100.bgen",
	               (char*) "--out", (char*) "unit/data/test1a.out.gz"};

char* case1b[] = { (char*) "prog",
	               (char*) "--VB-varEM",
	               (char*) "--mode-regress-out-covars",
	               (char*) "--random-seed", (char*) "1",
	               (char*) "--VB-iter-max", (char*) "10",
	               (char*) "--vb-iter-start", (char*) "3",
	               (char*) "--resume-from-state",
	               (char*) "unit/data/lemma_interim_files/test1a_dump_it2",
	               (char*) "--pheno", (char*) "unit/data/pheno.txt",
	               (char*) "--environment", (char*) "unit/data/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "unit/data/n50_p100.bgen",
	               (char*) "--out", (char*) "unit/data/test1b.out.gz"};

char* case1c[] = { (char*) "prog",
	               (char*) "--singleSnpStats",
	               (char*) "--mode-regress-out-covars",
	               (char*) "--random-seed", (char*) "1",
	               (char*) "--resume-from-state",
	               (char*) "unit/data/lemma_interim_files/test1a_dump_it_converged",
	               (char*) "--pheno", (char*) "unit/data/pheno.txt",
	               (char*) "--environment", (char*) "unit/data/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "unit/data/n50_p100.bgen",
	               (char*) "--out", (char*) "unit/data/test1c.out.gz"};

TEST_CASE("Resume from state: varEM w/ multiE + regress out covars"){
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

		VBayes VB(data);

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
			CHECK(VB.calc_logw(hyps, vp) == Approx(-97.4171842086));
			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-96.9220418937));
			VB.updateAllParams(2, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-96.5785826372));

			VbTracker tracker(p);
			tracker.init_interim_output(0,2, VB.n_effects, VB.n_covar, VB.n_env, VB.env_names, vp);
			tracker.dump_state("2", VB.n_samples, VB.n_covar, VB.n_var, VB.n_env,
			                   VB.n_effects, vp, hyps, VB.Y, VB.C, VB.X,
			                   VB.covar_names, VB.env_names, VB.sample_is_invalid, VB.sample_location);

			VB.updateAllParams(3, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-96.3066786091));
		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		VB.output_vb_results();
		SECTION("Check converged inference"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-95.5990828788));
		}

		SECTION("Check converged snp-stats"){
			CHECK(mpiUtils::mpiReduce_inplace(VB.vp_init.ym.array().abs().sum()) == Approx(9.3519780272));
			// Compute snp-stats
			Eigen::MatrixXd neglogPvals, testStats;
			VB.compute_LOCO_pvals(VB.vp_init, neglogPvals, testStats);
			CHECK(neglogPvals(0,0) == Approx(0.2695461918));
			CHECK(neglogPvals(0,1) == Approx(1.4084944363));
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

		VBayes VB(data);

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
			CHECK(VB.calc_logw(hyps, vp) == Approx(-96.3066786142));
		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-95.5990828779));
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
		VBayes VB(data);
		CHECK(mpiUtils::mpiReduce_inplace(VB.vp_init.ym.array().abs().sum()) == Approx(9.3519781087));

		// Compute snp-stats
		Eigen::MatrixXd neglogPvals, testStats;
		VB.compute_LOCO_pvals(VB.vp_init, neglogPvals, testStats);
		CHECK(neglogPvals(0,0) == Approx(0.2695461941));
		CHECK(neglogPvals(0,1) == Approx(1.4084944606));
	}
}
