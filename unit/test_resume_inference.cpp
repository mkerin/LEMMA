// tests-main.cpp
#define EIGEN_USE_MKL_ALL
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
char* case1[] = { (char*) "--mode_vb",
				 (char*) "--mode_empirical_bayes",
				 (char*) "--vb_iter_max", (char*) "10",
				 (char*) "--hyps_grid", (char*) "data/io_test/single_hyps_gxage.txt",
				 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
				 (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
				 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
				 (char*) "--out", (char*) "data/io_test/test1.out.gz"};

char* case2[] = { (char*) "--mode_vb",
				  (char*) "--mode_empirical_bayes",
				  (char*) "--vb_iter_max", (char*) "7",
				  (char*) "--resume_from_param_dump",
				  (char*) "data/io_test/r2_interim_files/grid_point_0/test1_dump_it2",
				  (char*) "--pheno", (char*) "data/io_test/pheno.txt",
				  (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
				  (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
				  (char*) "--out", (char*) "data/io_test/test2.out"};



TEST_CASE("Resume from multi-env + mog + emp_bayes"){
	parameters p;

	SECTION("Run to iter 10"){
		int argc = sizeof(case1)/sizeof(case1[0]);
		parse_arguments(p, argc, case1);

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
			CHECK(VB.calc_logw(hyps, vp) == Approx(-92.1213458535));
			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-89.5921227491));
			VB.updateAllParams(2, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-88.4669513868));

			VbTracker tracker(p);
			tracker.init_interim_output(0,2, VB.n_effects, VB.n_env, VB.env_names, vp);
			tracker.dump_state(2, VB.n_samples, VB.n_covar, VB.n_var, VB.n_env,
							   VB.n_effects, vp, hyps, VB.Y, VB.C, VB.X,
							   VB.covar_names, VB.env_names);

			// Diagonostics
			CHECK(vp.ym.squaredNorm() == Approx(14.6703611205));
			CHECK(vp.yx.squaredNorm() == Approx(0.0005146067));
			CHECK(vp.eta.squaredNorm() == Approx(0.0813884679));
			CHECK(vp.muw.square().sum() == Approx(0.0016355123));
			CHECK(vp.alpha_beta.square().sum() == Approx(3.4319352731));
			CHECK(vp.alpha_gam.square().sum() == Approx(0.7119338618));
			CHECK(vp.sw_sq.square().sum() == Approx(0.9690735878));
			CHECK(vp.sc_sq.square().sum() == Approx(0.0006112583));

			CHECK(hyps.sigma                == Approx(0.563194192));
			CHECK(hyps.lambda[0]            == Approx(0.2146615463));
			CHECK(hyps.lambda[1]    == Approx(0.1030672137));
			CHECK(hyps.slab_var[0]  == Approx(0.008667515));
			CHECK(hyps.slab_var[1]  == Approx(0.0019321614));
			// CHECK(hyps.spike_var[0] == Approx(0.0000003668));
			// CHECK(hyps.spike_var[1] == Approx(0.0000003667));

			VB.updateAllParams(3, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-87.8790255503));

		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.8131749627));
		}
	}

	SECTION("Resume from iter 2"){
		int argc = sizeof(case2)/sizeof(case2[0]);
		parse_arguments(p, argc, case2);

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.calc_dxteex();
		data.set_vb_init();

		SECTION("Initialise properly from file") {
			CHECK(data.vp_init.muw.square().sum() == Approx(0.0016355123));
			CHECK(data.hyps_inits[0].sigma                == Approx(0.563194192));
			CHECK(data.vp_init.sw_sq.square().sum() == Approx(0.9690735878));
		}

		VBayesX2 VB(data);
		CHECK(VB.hyps_inits[0].sigma                == Approx(0.563194192));
		CHECK(VB.vp_init.sw_sq.square().sum() == Approx(0.9690735878));

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

			CHECK(hyps.sigma                == Approx(0.563194192));
			CHECK(hyps.lambda[0]            == Approx(0.2146615463));
			CHECK(hyps.lambda[1]    == Approx(0.1030672137));
			CHECK(hyps.slab_var[0]  == Approx(0.008667515));
			CHECK(hyps.slab_var[1]  == Approx(0.0019321614));
			// CHECK(hyps.spike_var[0] == Approx(0.0000003668));
			// CHECK(hyps.spike_var[1] == Approx(0.0000003667));

			CHECK(vp.ym.squaredNorm() == Approx(14.6703611205));
			CHECK(vp.yx.squaredNorm() == Approx(0.0005146067));
			CHECK(vp.eta.squaredNorm() == Approx(0.0813884679));
			CHECK(vp.muw.square().sum() == Approx(0.0016355123));
			CHECK(vp.alpha_beta.square().sum() == Approx(3.4319352731));
			CHECK(vp.alpha_gam.square().sum() == Approx(0.7119338618));
			CHECK(vp.sw_sq.square().sum() == Approx(0.9690735878));
			CHECK(vp.sc_sq.square().sum() == Approx(0.0006112583));
			CHECK(VB.calc_logw(hyps, vp) == Approx(-88.4669513868));

			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(hyps, vp) == Approx(-87.8790255503));

		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 7);
			CHECK(trackers[0].logw == Approx(-86.8131749627));
		}
	}
}
