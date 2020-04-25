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
	               (char*) "--mode_regress_out_covars",
	               (char*) "--VB",
	               (char*) "--VB-varEM",
	               (char*) "--spike_diff_factor", (char*) "10000",
	               (char*) "--VB-iter-max", (char*) "10",
	               (char*) "--hyps_grid", (char*) "data/io_test/single_hyps_gxage.txt",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test1a.out"};

char* case1b[] = { (char*) "prog",
	               (char*) "--mode_regress_out_covars",
	               (char*) "--VB",
	               (char*) "--VB-varEM",
	               (char*) "--spike_diff_factor", (char*) "10000",
	               (char*) "--VB-iter-max", (char*) "10",
	               (char*) "--vb_iter_start", (char*) "3",
	               (char*) "--resume_from_state",
	               (char*) "data/io_test/lemma_interim_files/test1a_dump_it2",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test1b.out"};

char* case1c[] = { (char*) "prog",
	               (char*) "--mode_regress_out_covars",
	               (char*) "--singleSnpStats",
	               (char*) "--resume_from_state",
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


char* case2a[] = { (char*) "prog",
	               (char*) "--VB",
	               (char*) "--VB-squarem",
	               (char*) "--verbose",
	               (char*) "--spike_diff_factor", (char*) "10000",
	               (char*) "--VB-iter-max", (char*) "10",
	               (char*) "--hyps_grid", (char*) "data/io_test/single_hyps_gxage.txt",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test2a.out.gz"};

char* case2b[] = { (char*) "prog",
	               (char*) "--VB",
	               (char*) "--VB-squarem",
	               (char*) "--spike_diff_factor", (char*) "10000",
	               (char*) "--VB-iter-max", (char*) "10",
	               (char*) "--vb_iter_start", (char*) "3",
	               (char*) "--resume_from_state",
	               (char*) "data/io_test/r2_interim_files/grid_point_0/test2a_dump_it2",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test2b.out.gz"};

TEST_CASE("Resume from multi-env + mog + squarem"){
	parameters p;

	SECTION("Run to iter 10"){
		int argc = sizeof(case2a)/sizeof(case2a[0]);
		parse_arguments(p, argc, case2a);
		p.mode_squarem = true;

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.calc_dxteex();
		data.set_vb_init();

		VBayesX2 VB(data);
		std::cout << "Initialised" << std::endl;
		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
		// SECTION("Ex4. Explicitly checking hyps") {
		//  // Set up for RunInnerLoop
		//  long n_grid = VB.hyps_inits.size();
		//  std::vector<Hyps> all_hyps = VB.hyps_inits;
		//  std::vector<VariationalParameters> all_vp;
		//  VB.setup_variational_params(all_hyps, all_vp);
		//
		//  int round_index = 2;
		//  std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
		//  std::vector<std::vector< double > > logw_updates(n_grid);
		//  VariationalParameters& vp = all_vp[0];
		//  Hyps& hyps = all_hyps[0];
		//
		//  VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);
		//  CHECK(VB.calc_logw(hyps, vp) == Approx(-92.2292775905));
		//  VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);
		//  CHECK(VB.calc_logw(hyps, vp) == Approx(-89.6317892411));
		//  VB.updateAllParams(2, round_index, all_vp, all_hyps, logw_prev);
		//  CHECK(VB.calc_logw(hyps, vp) == Approx(-88.4701165481));
		//
		//  // CHECK(VB.YM.squaredNorm() == Approx(14.5271697557));
		//  // CHECK(VB.YX.squaredNorm() == Approx(0.0004746624));
		//  // CHECK(VB.ETA.squaredNorm() == Approx(0.0740853408));
		//  // CHECK(VB.ETA_SQ.squaredNorm() == Approx(294.4966683916));
		//
		//  VbTracker tracker(p);
		//  tracker.init_interim_output(0,2, VB.n_effects, VB.n_covar, VB.n_env, VB.env_names, vp);
		//  tracker.dump_state("2", VB.n_samples, VB.n_covar, VB.n_var, VB.n_env,
		//                     VB.n_effects, vp, hyps, VB.Y, VB.C, VB.X,
		//                     VB.covar_names, VB.env_names, VB.sample_is_invalid);
		//
		//  VB.updateAllParams(3, round_index, all_vp, all_hyps, logw_prev);
		//  CHECK(VB.calc_logw(hyps, vp) == Approx(-87.8793981688));
		//
		//  // CHECK(vp.ym.squaredNorm() == Approx(15.7275802138));
		//  // CHECK(vp.yx.squaredNorm() == Approx(0.0000884683));
		//  // CHECK(vp.eta.squaredNorm() == Approx(0.0218885153));
		//  // CHECK(VB.ETA_SQ.squaredNorm() == Approx(397.4128947093));
		// }

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		// SECTION("Ex3. Vbayes_X2 inference correct"){
		//  CHECK(trackers[0].count == 10);
		//  CHECK(trackers[0].logw == Approx(-86.6476101917));
		// }
	}

	// SECTION("Resume from iter 2"){
	//  int argc = sizeof(case2b)/sizeof(case2b[0]);
	//  parse_arguments(p, argc, case2b);
	//  p.mode_squarem = true;
	//
	//  Data data(p);
	//  data.read_non_genetic_data();
	//  data.standardise_non_genetic_data();
	//  data.read_full_bgen();
	//  data.calc_dxteex();
	//  data.set_vb_init();
	//
	//  VBayesX2 VB(data);
	//
	//  std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
	//  SECTION("Ex4. Explicitly checking hyps") {
	//      long n_grid = VB.hyps_inits.size();
	//      std::vector<Hyps> all_hyps = VB.hyps_inits;
	//      std::vector<VariationalParameters> all_vp;
	//      VB.setup_variational_params(all_hyps, all_vp);
	//
	//      int round_index = 2;
	//      std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
	//      std::vector<std::vector< double > > logw_updates(n_grid);
	//      VariationalParameters& vp = all_vp[0];
	//      Hyps& hyps = all_hyps[0];
	//
	//      // CHECK(VB.YM.squaredNorm() == Approx(14.5271697722));
	//      // CHECK(VB.YX.squaredNorm() == Approx(0.0004746624));
	//      // CHECK(VB.ETA.squaredNorm() == Approx(0.0740853423));
	//      // CHECK(VB.ETA_SQ.squaredNorm() == Approx(294.4966685937));
	//      CHECK(VB.calc_logw(hyps, vp) == Approx(-88.4701165472));
	//
	//      VB.updateAllParams(3, round_index, all_vp, all_hyps, logw_prev);
	//      CHECK(VB.calc_logw(hyps, vp) == Approx(-87.8793981839));
	//
	//      // CHECK(vp.ym.squaredNorm() == Approx(15.7275801873));
	//      // CHECK(vp.yx.squaredNorm() == Approx(0.0000884683));
	//      // CHECK(vp.eta.squaredNorm() == Approx(0.0218885164));
	//      // CHECK(VB.ETA_SQ.squaredNorm() == Approx(397.4128907661));
	//  }
	//
	//  VB.run_inference(VB.hyps_inits, false, 2, trackers);
	//  SECTION("Ex3. Vbayes_X2 inference correct"){
	//      CHECK(trackers[0].count == 10);
	//      // Slight discrepancy between original run and restart.
	//      // Think this is because we now need the previous two hyps values to
	//      // keep using SQUAREM from the same place
	//      // CHECK(trackers[0].logw == Approx(-86.6456071112));
	//      CHECK(trackers[0].logw == Approx(-86.5367967574));
	//  }
	// }
}


// Scenarios
char* case3a[] = { (char*) "prog",
	               (char*) "--VB",
	               (char*) "--VB-varEM",
	               (char*) "--use_vb_on_covars",
	               (char*) "--VB-iter-max", (char*) "10",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test3a.out"};

char* case3c[] = { (char*) "prog",
	               (char*) "--singleSnpStats", (char*) "--use_vb_on_covars",
	               (char*) "--resume_from_state",
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
			CHECK(trackers[0].logw == Approx(-88.8466020959));
		}

		SECTION("Check converged snp-stats"){
			// Compute snp-stats
			Eigen::MatrixXd neglogPvals, testStats;
			VB.compute_LOCO_pvals(VB.vp_init, neglogPvals, testStats);
			CHECK(neglogPvals(0,0) == Approx(0.2763157508));
			CHECK(neglogPvals(0,1) == Approx(1.5541356878));
			CHECK(neglogPvals(0,2) == Approx(3.0085535567));
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
		CHECK(neglogPvals(0,0) == Approx(0.2763157508));
		CHECK(neglogPvals(0,1) == Approx(1.5541356878));
		CHECK(neglogPvals(0,2) == Approx(3.0085535567));
	}
}
