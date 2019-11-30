//
// Created by kerin on 2019-04-24.
//

// tests-main.cpp
#define EIGEN_USE_MKL_ALL
#include "catch.hpp"
#include "../src/tools/Eigen/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes_x2.hpp"
#include "../src/data.hpp"
#include "../src/hyps.hpp"


// Scenarios
char* case1a[] = { (char*) "--mode_vb",
				   (char*) "--mode_empirical_bayes",
				   (char*) "--env_momentum_coeff", (char*) "0.8",
				   (char*) "--vb_iter_max", (char*) "10",
				   (char*) "--hyps_grid", (char*) "data/io_test/single_hyps_gxage.txt",
				   (char*) "--pheno", (char*) "data/io_test/pheno.txt",
				   (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
				   (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
				   (char*) "--out", (char*) "data/io_test/test1a.out.gz"};


TEST_CASE("Momentum with multi-env + mog + emp_bayes"){
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
			CHECK(VB.calc_logw(vp) == Approx(-92.2141626048));
			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(vp) == Approx(-89.6586854116));


			VB.updateAllParams(2, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(vp) == Approx(-88.4813237554));

			CHECK(vp.ym.squaredNorm() == Approx(14.6480266984));
			CHECK(vp.yx.squaredNorm() == Approx(0.0004676258));
			CHECK(vp.eta.squaredNorm() == Approx(0.0744622661));
			CHECK(vp.eta_sq.squaredNorm() == Approx(297.6156962976));

			VbTracker tracker(p);
			tracker.init_interim_output(0,2, VB.n_effects, VB.n_env, VB.env_names, vp);
			tracker.dump_state(2, VB.n_samples, VB.n_covar, VB.n_var, VB.n_env,
							   VB.n_effects, vp, hyps, VB.Y, VB.C, VB.X,
							   VB.covar_names, VB.env_names);

			VB.updateAllParams(3, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(vp) == Approx(-87.880171198));

			CHECK(vp.ym.squaredNorm() == Approx(15.7914628605));
			CHECK(vp.yx.squaredNorm() == Approx(0.0000880429));
			CHECK(vp.eta.squaredNorm() == Approx(0.0221053892));
		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.8131749627));
		}
	}

}
