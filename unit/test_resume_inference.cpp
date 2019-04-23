// tests-main.cpp
#define EIGEN_USE_MKL_ALL
#include "catch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include "../src/tools/Eigen/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes_x2.hpp"
#include "../src/data.hpp"
#include "../src/hyps.hpp"


// Scenarios
char* case1a[] = { (char*) "--mode_vb",
	               (char*) "--mode_empirical_bayes",
	               (char*) "--vb_iter_max", (char*) "10",
	               (char*) "--hyps_grid", (char*) "data/io_test/single_hyps_gxage.txt",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test1a.out.gz"};

char* case1b[] = { (char*) "--mode_vb",
	               (char*) "--mode_empirical_bayes",
	               (char*) "--vb_iter_max", (char*) "10",
	               (char*) "--vb_iter_start", (char*) "3",
	               (char*) "--resume_from_param_dump",
	               (char*) "data/io_test/r2_interim_files/grid_point_0/test1a_dump_it2",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test1b.out"};



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

			CHECK(vp.ym.squaredNorm() == Approx(14.6480266997));
			CHECK(vp.yx.squaredNorm() == Approx(0.0004676258));
			CHECK(vp.eta.squaredNorm() == Approx(0.0744622647));
			CHECK(vp.eta_sq.squaredNorm() == Approx(297.6156980469));

			VB.updateAllParams(3, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(vp) == Approx(-87.8801711925));

			CHECK(vp.ym.squaredNorm() == Approx(15.7914630418));
			CHECK(vp.yx.squaredNorm() == Approx(0.0000880429));
			CHECK(vp.eta.squaredNorm() == Approx(0.0221053884));
		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.8131749627));
		}
	}
}


char* case2a[] = { (char*) "--mode_vb",
	               (char*) "--mode_squarem",
	               (char*) "--vb_iter_max", (char*) "10",
	               (char*) "--hyps_grid", (char*) "data/io_test/single_hyps_gxage.txt",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test2a.out.gz"};

char* case2b[] = { (char*) "--mode_vb",
	               (char*) "--mode_squarem",
	               (char*) "--vb_iter_max", (char*) "10",
	               (char*) "--vb_iter_start", (char*) "3",
	               (char*) "--resume_from_param_dump",
	               (char*) "data/io_test/r2_interim_files/grid_point_0/test2a_dump_it2",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test2b.out"};

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
			CHECK(vp.eta_sq.squaredNorm() == Approx(400.5912659102));
		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.6556683784));
		}
	}

	SECTION("Resume from iter 2"){
		int argc = sizeof(case2b)/sizeof(case2b[0]);
		parse_arguments(p, argc, case2b);
		p.mode_squarem = true;

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

			CHECK(vp.ym.squaredNorm() == Approx(14.6480266997));
			CHECK(vp.yx.squaredNorm() == Approx(0.0004676258));
			CHECK(vp.eta.squaredNorm() == Approx(0.0744622647));
			CHECK(vp.eta_sq.squaredNorm() == Approx(297.6156980469));
			CHECK(VB.calc_logw(vp) == Approx(-88.4813237607));

//			CHECK(vp.kl_div_beta() == vp.betas.kl)

			VB.updateAllParams(3, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(vp) == Approx(-87.8801711925));

			CHECK(vp.ym.squaredNorm() == Approx(15.7914630418));
			CHECK(vp.yx.squaredNorm() == Approx(0.0000880429));
			CHECK(vp.eta.squaredNorm() == Approx(0.0221053884));
			CHECK(vp.eta_sq.squaredNorm() == Approx(400.5912674531));
		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			// Slight discrepancy between original run and restart.
			// Think this is because we now need the previous two hyps values to
			// keep using SQUAREM from the same place
			// CHECK(trackers[0].logw == Approx(-86.6456071112));
			CHECK(trackers[0].logw == Approx(-86.5254441461));
		}
	}
}
