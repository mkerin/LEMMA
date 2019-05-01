//
// Created by kerin on 2019-04-30.
//

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

// #define EIGEN_USE_MKL_ALL
// #include "../src/hyps.hpp"
// #include "../src/parse_arguments.hpp"
// #include "../src/vbayes_x2.hpp"
// #include "../src/data.hpp"
//
// #include "catch.hpp"
// #include "../src/tools/Eigen/Dense"
//
// #include <iostream>
// #include <vector>
// #include <string>

// Scenarios
char* accel_case1a[] = { (char*) "--mode_vb",
	                     (char*) "--mode_empirical_bayes",
	                     (char*) "--beta_prior_gaussian",
	                     (char*) "--vb_iter_max", (char*) "10",
	                     (char*) "--hyps_grid", (char*) "data/io_test/single_hyps_main.txt",
	                     (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	                     (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	                     (char*) "--out", (char*) "data/io_test/test1a.out.gz"};

char* accel_case1b[] = { (char*) "--DEBUG_mode_vb_accelerated",
	                     (char*) "--mode_empirical_bayes",
	                     (char*) "--beta_prior_gaussian",
	                     (char*) "--vb_iter_max", (char*) "10",
	                     (char*) "--hyps_grid", (char*) "data/io_test/single_hyps_main.txt",
	                     (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	                     (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	                     (char*) "--out", (char*) "data/io_test/test1a.out.gz"};

char* accel_case1c[] = { (char*) "--mode_vb_accelerated",
	                     (char*) "--mode_empirical_bayes",
	                     (char*) "--beta_prior_gaussian",
	                     (char*) "--vb_iter_max", (char*) "10",
	                     (char*) "--hyps_grid", (char*) "data/io_test/single_hyps_main.txt",
	                     (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	                     (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	                     (char*) "--out", (char*) "data/io_test/test1a.out.gz"};

// Accelerated Coordinate Descent
TEST_CASE("ACD"){
	parameters p;

	SECTION("MainEffectsOnly (null)"){
		int argc = sizeof(accel_case1a)/sizeof(accel_case1a[0]);
		parse_arguments(p, argc, accel_case1a);

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
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(vp) == Approx(-93.0209563673));
		}
		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-61.9139513159));
		}
	}

	SECTION("MainEffectsOnly (accelerated; theta constant)"){
		int argc = sizeof(accel_case1b)/sizeof(accel_case1b[0]);
		parse_arguments(p, argc, accel_case1b);

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
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);
			CHECK(VB.calc_logw(vp) == Approx(-93.0209563673));
		}
		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-61.9139513159));
		}
	}

	SECTION("MainEffectsOnly (accelerated)"){
		int argc = sizeof(accel_case1c)/sizeof(accel_case1c[0]);
		parse_arguments(p, argc, accel_case1c);

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
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);
			// CHECK(VB.calc_logw(vp) == Approx(-93.0209563673));
			CHECK(VB.calc_logw(vp) == Approx(-92.3133440561));
		}
		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			// CHECK(trackers[0].logw == Approx(-61.9139513159));
			// Slight improvement over original estimates is encouraging
			CHECK(trackers[0].logw == Approx(-61.7304605732));
		}
	}
}
