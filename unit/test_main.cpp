// tests-main.cpp
#include "catch.hpp"

#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes_x2.hpp"
#include "../src/data.hpp"
#include "../src/hyps.hpp"
#include "../src/genotype_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <sys/stat.h>

TEST_CASE( "Example 4: multi-env + mog + covars + emp_bayes" ){
	parameters p;

	SECTION("Ex4. No filters applied, high mem mode"){
		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb", (char*) "--low_mem",
			             (char*) "--mode_regress_out_covars", (char*) "--mode_empirical_bayes",
			             (char*) "--spike_diff_factor", (char*) "10000",
			             (char*) "--effects_prior_mog",
			             (char*) "--vb_iter_max", (char*) "10",
			             (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
			             (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
			             (char*) "--out", (char*) "data/io_test/config4.out",
			             (char*) "--pheno", (char*) "data/io_test/pheno.txt",
			             (char*) "--hyps_grid", (char*) "data/io_test/single_hyps_gxage.txt",
			             (char*) "--hyps_probs", (char*) "data/io_test/single_hyps_gxage_probs.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		Data data( p );

		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		SECTION( "Ex4. Non genetic data standardised + covars regressed"){
			CHECK(!data.p.use_vb_on_covars);
			CHECK(data.p.covar_file == "NULL");
		}
		data.read_full_bgen();

		data.calc_dxteex();
		data.calc_snpstats();
		data.set_vb_init();
		VBayesX2 VB(data);
		SECTION("Ex4. Vbayes_X2 initialised correctly"){
			CHECK(VB.Nglobal == 50);
			CHECK(VB.Nglobal == 50.0);
			CHECK(VB.n_env == 4);
			CHECK(VB.n_var == 67);
			CHECK(VB.n_effects == 2);
			CHECK(VB.vp_init.muw(0) == 0.25);
			CHECK(VB.p.init_weights_with_snpwise_scan == false);
			// CHECK(VB.dXtEEX_lowertri(0, dXtEEX_col_ind(1, 0, data.n_env)) == Approx(-2.6239467101));
			// CHECK(VB.dXtEEX_lowertri(1, dXtEEX_col_ind(1, 0, data.n_env)) == Approx(-13.0001255314));
		}

		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
		SECTION("Ex4. Explicitly checking hyps") {
			// Set up for RunInnerLoop
			long n_grid = VB.hyps_inits.size();
			std::vector<Hyps> all_hyps = VB.hyps_inits;

			// Set up for updateAllParams
			std::vector<VariationalParameters> all_vp;
			VB.setup_variational_params(all_hyps, all_vp);

			int round_index = 2;
			std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
			std::vector<std::vector< double > > logw_updates(n_grid);
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			EigenDataVector check_ym;
			Eigen::VectorXd Eq_beta;

			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);

			CHECK(vp.alpha_beta(0)            == Approx(0.1347598474));
			CHECK(vp.alpha_beta(1)            == Approx(0.1406737102));
			CHECK(vp.alpha_beta(63)           == Approx(0.142092349));
			CHECK(vp.mean_gam(0)           == Approx(-0.0008039753));
			CHECK(vp.mean_gam(63)           == Approx(-0.0004864387));
			CHECK(vp.muw(0, 0)              == Approx(0.1064068968));
			CHECK(vp.sw_sq(0, 0)              == Approx(0.4188888912));

			CHECK(hyps.sigma                == Approx(0.6981184547));
			CHECK(hyps.lambda[0]            == Approx(0.1683579572));
			CHECK(hyps.lambda[1]            == Approx(0.1351099647));
			CHECK(hyps.slab_relative_var[0] == Approx(0.0080625769));
			CHECK(hyps.slab_relative_var[1] == Approx(0.0051035881));

			Eq_beta = vp.alpha_beta * vp.mu1_beta;
			if(p.mode_mog_prior_beta) Eq_beta.array() += (1 - vp.alpha_beta) * vp.mu2_beta;
			check_ym  = VB.X * Eq_beta;
			check_ym += VB.C * vp.muc.cast<scalarData>().matrix();
			CHECK(vp.ym(0)            == Approx(check_ym(0)));

			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);

			CHECK(vp.alpha_beta(0)            == Approx(0.1465935244));
			CHECK(vp.muw(0, 0)              == Approx(0.0586700329));
			CHECK(vp.alpha_gam(63)           == Approx(0.1180479917));
			CHECK(vp.mu1_gam(63)              == Approx(0.0018853779));
			CHECK(vp.s1_gam_sq(63)            == Approx(0.002611992));

			Eq_beta = vp.alpha_beta * vp.mu1_beta;
			if(p.mode_mog_prior_beta) Eq_beta.array() += (1 - vp.alpha_beta) * vp.mu2_beta;
			check_ym  = VB.X * Eq_beta;
			check_ym += VB.C * vp.muc.cast<scalarData>().matrix();
			CHECK(vp.ym(0)            == Approx(check_ym(0)));

			VB.updateAllParams(2, round_index, all_vp, all_hyps, logw_prev);

			CHECK(vp.alpha_beta(63)           == Approx(0.1860890959));
			CHECK(vp.muw(0, 0)              == Approx(0.0357150196));
			CHECK(vp.alpha_gam(63)           == Approx(0.1036438402));
			CHECK(vp.mu1_gam(63)              == Approx(0.0000414655));

			CHECK(VB.calc_logw(hyps, vp) == Approx(-88.4701165481));
			VbTracker tracker(p);
			tracker.init_interim_output(0,2, VB.n_effects, VB.n_covar, VB.n_env, VB.env_names, vp);
			tracker.dump_state("2", VB.n_samples, VB.n_covar, VB.n_var, VB.n_env,
			                   VB.n_effects, vp, hyps, VB.Y, VB.C, VB.X,
			                   VB.covar_names, VB.env_names, VB.sample_is_invalid, VB.sample_location);

			// Checking logw
			double int_linear = -1.0 * VB.calcExpLinear(hyps, vp) / 2.0 / hyps.sigma;
			int_linear -= VB.Nglobal * std::log(2.0 * VB.PI * hyps.sigma) / 2.0;
			// CHECK(int_linear  == Approx(-58.5936502834));

			// CHECK(VB.calcExpLinear(hyps, vp)  == Approx(30.4124788103));
			// CHECK(VB.calcKLBeta(hyps, vp)  == Approx(-5.4013615932));
			// CHECK(VB.calcKLGamma(hyps, vp)  == Approx(-0.0053957728));

			// check int_linear

			// Expectation of linear regression log-likelihood
			int_linear  = (VB.Y - vp.ym).squaredNorm();
			int_linear -= 2.0 * (VB.Y - vp.ym).cwiseProduct(vp.eta).cwiseProduct(vp.yx).sum();
			int_linear += vp.yx.cwiseProduct(vp.eta_sq).dot(vp.yx);
			// CHECK(int_linear == Approx(21.6133648827));

			double int_linear2  = (VB.Y - vp.ym - vp.yx.cwiseProduct(vp.eta)).squaredNorm();
			int_linear2 -= vp.yx.cwiseProduct(vp.eta).squaredNorm();
			int_linear2 += vp.yx.cwiseProduct(vp.eta_sq).dot(vp.yx);
			// CHECK(int_linear2 == Approx(21.6133648827));

			double kl_covar = 0.0;
			kl_covar += (double) VB.n_covar * (1.0 - std::log(hyps.sigma * VB.sigma_c)) / 2.0;
			kl_covar += vp.sc_sq.log().sum() / 2.0;
			kl_covar -= vp.sc_sq.sum() / 2.0 / hyps.sigma / VB.sigma_c;
			kl_covar -= vp.muc.square().sum() / 2.0 / hyps.sigma / VB.sigma_c;
			// CHECK(kl_covar == Approx(-24.0588694492));

			// weights
			double kl_weights = 0.0;
			kl_weights += (double) VB.n_env / 2.0;
			kl_weights += vp.sw_sq.log().sum() / 2.0;
			kl_weights -= vp.sw_sq.sum() / 2.0;
			kl_weights -= vp.muw.square().sum() / 2.0;
			// CHECK(kl_weights == Approx(-0.4078325179));


			// variances
			CHECK(vp.EdZtZ.sum() == Approx(6226.9426820702));
			// CHECK(vp.ym.squaredNorm() == Approx(14.5271697557));
			// CHECK(vp.yx.squaredNorm() == Approx(0.0004746624));
		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.8131749627));
		}
	}
}

TEST_CASE( "Example 4: multi-env + mog + covars + emp_bayes + sample subset" ){
	parameters p;

	SECTION("Ex4. No filters applied, high mem mode"){
		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb", (char*) "--low_mem",
			             (char*) "--mode_regress_out_covars", (char*) "--mode_empirical_bayes",
			             (char*) "--spike_diff_factor", (char*) "10000",
			             (char*) "--effects_prior_mog",
			             (char*) "--incl_sample_ids", (char*) "data/io_test/sample_ids_head28.txt",
			             (char*) "--vb_iter_max", (char*) "10",
			             (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
			             (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
			             (char*) "--out", (char*) "data/io_test/config4_subset.out",
			             (char*) "--pheno", (char*) "data/io_test/pheno.txt",
			             (char*) "--hyps_grid", (char*) "data/io_test/single_hyps_gxage.txt",
			             (char*) "--hyps_probs", (char*) "data/io_test/single_hyps_gxage_probs.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		Data data( p );

		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		std::cout << "Standardised" << std::endl;
		data.read_full_bgen();
		std::cout << "finished reading bgen" << std::endl;

		data.calc_dxteex();
		std::cout << "calc dxteex" << std::endl;
		data.calc_snpstats();
		data.set_vb_init();
		VBayesX2 VB(data);
		SECTION("Ex4. Vbayes_X2 initialised correctly"){
			CHECK(VB.Nglobal == 28);
			CHECK(VB.n_var == 54);
		}

		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		VB.output_results("");
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-57.062373213));
		}
	}
}

// Garbage test; testing dxteex should be way lower level
//TEST_CASE("--dxteex") {
//	parameters p;
//	char *argv[] = {(char *) "bin/bgen_prog", (char *) "--mode_vb", (char *) "--low_mem",
//		            (char *) "--use_vb_on_covars", (char *) "--mode_empirical_bayes",
//		            (char *) "--effects_prior_mog",
//		            (char *) "--vb_iter_max", (char *) "10",
//		            (char *) "--environment", (char *) "data/io_test/n50_p100_env.txt",
//		            (char *) "--bgen", (char *) "data/io_test/n50_p100.bgen",
//		            (char *) "--out", (char *) "data/io_test/config4.out",
//		            (char *) "--pheno", (char *) "data/io_test/pheno.txt",
//		            (char *) "--hyps_grid", (char *) "data/io_test/single_hyps_gxage.txt"};
//	int argc = sizeof(argv) / sizeof(argv[0]);
//	parse_arguments(p, argc, argv);
//
//	SECTION("Compute dxteex internally"){
//		Data data(p);
//
//		data.read_non_genetic_data();
//		data.standardise_non_genetic_data();
//		SECTION("Ex4. Non genetic data standardised + covars regressed") {
//			CHECK(data.E(0, 0) == Approx(0.8957059881));
//		}
//		data.read_full_bgen();
//
//		data.calc_dxteex();
//		data.set_vb_init();
//		VBayesX2 VB(data);
//		SECTION("Ex4. Vbayes_X2 initialised correctly") {
//			CHECK(VB.Nglobal == 50);
//			CHECK(VB.N == 50.0);
//			CHECK(VB.n_env == 4);
//			CHECK(VB.n_effects == 2);
//			CHECK(VB.vp_init.muw(0) == 0.25);
//			CHECK(VB.p.init_weights_with_snpwise_scan == false);
//			CHECK(VB.dXtEEX(0, 0) == Approx(38.9610805993));
//			CHECK(VB.dXtEEX(1, 0) == Approx(38.2995451744));
//			CHECK(VB.dXtEEX(2, 0) == Approx(33.7077899144));
//			CHECK(VB.dXtEEX(3, 0) == Approx(35.7391671158));
//
//			CHECK(VB.dXtEEX(0, 4) == Approx(-2.6239467101));
//			CHECK(VB.dXtEEX(1, 4) == Approx(-13.0001255314));
//			CHECK(VB.dXtEEX(2, 4) == Approx(-11.6635557299));
//			CHECK(VB.dXtEEX(3, 4) == Approx(-7.2154836264));
//		}
//
//		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
//		VB.run_inference(VB.hyps_inits, false, 2, trackers);
//		SECTION("Ex3. Vbayes_X2 inference correct"){
//			CHECK(trackers[0].count == 10);
//			CHECK(trackers[0].logw == Approx(-86.8131699164));
//		}
//	}
//
//	SECTION("Compute dxteex external") {
//		p.dxteex_file = "data/io_test/n50_p100_dxteex.txt";
//		Data data(p);
//
//		data.read_non_genetic_data();
//		data.standardise_non_genetic_data();
//		SECTION("Ex4. Non genetic data standardised + covars regressed") {
//			CHECK(data.E(0, 0) == Approx(0.8957059881));
//		}
//		data.read_full_bgen();
//
//		data.read_external_dxteex();
//		data.calc_dxteex();
//		data.set_vb_init();
//		VBayesX2 VB(data);
//		SECTION("Ex4. Vbayes_X2 initialised correctly") {
//			CHECK(VB.Nglobal == 50);
//			CHECK(VB.N == 50.0);
//			CHECK(VB.n_env == 4);
//			CHECK(VB.n_effects == 2);
//			CHECK(VB.vp_init.muw(0) == 0.25);
//			CHECK(VB.p.init_weights_with_snpwise_scan == false);
//			CHECK(VB.dXtEEX(0, 0) == Approx(38.9610805993));
//			CHECK(VB.dXtEEX(1, 0) == Approx(38.3718));
//			CHECK(VB.dXtEEX(2, 0) == Approx(33.81659));
//			CHECK(VB.dXtEEX(3, 0) == Approx(35.8492));
//
//			CHECK(VB.dXtEEX(0, 4) == Approx(-2.6239467101));
//			CHECK(VB.dXtEEX(1, 4) == Approx(-12.96763));
//			CHECK(VB.dXtEEX(2, 4) == Approx(-11.66501));
//			CHECK(VB.dXtEEX(3, 4) == Approx(-7.20105));
//		}
//
//		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
//		VB.run_inference(VB.hyps_inits, false, 2, trackers);
//		SECTION("Ex3. Vbayes_X2 inference correct"){
//			CHECK(trackers[0].count == 10);
//			CHECK(trackers[0].logw == Approx(-86.8131699164));
//		}
//	}
//}
//

// This whole section needs redoing
//TEST_CASE("--dxteex case8") {
//	parameters p;
//	char *argv[] = {(char *) "bin/bgen_prog", (char *) "--mode_vb", (char *) "--low_mem",
//		            (char *) "--mode_empirical_bayes",
//		            (char *) "--effects_prior_mog",
//		            (char *) "--use_vb_on_covars",
//		            (char *) "--vb_iter_max", (char *) "30",
//		            (char *) "--environment", (char *) "data/io_test/case8/env.txt",
//		            (char *) "--bgen", (char *) "data/io_test/n1000_p2000.bgen",
//		            (char *) "--out", (char *) "data/io_test/case8/inference.out",
//		            (char *) "--pheno", (char *) "data/io_test/case8/pheno.txt",
//		            (char *) "--hyps_grid", (char *) "data/io_test/case8/hyperpriors_gxage_v1.txt",
//		            (char *) "--vb_init", (char *) "data/io_test/case8/joint_init2.txt"};
//	int argc = sizeof(argv) / sizeof(argv[0]);
//	parse_arguments(p, argc, argv);
//
//	SECTION("Compute dxteex internally"){
//		Data data(p);
//
//		data.read_non_genetic_data();
//		data.standardise_non_genetic_data();
//		SECTION("Ex4. Non genetic data standardised + covars regressed") {
//			CHECK(data.E(0, 0) == Approx(0));
//		}
//		data.read_full_bgen();
//
//		data.calc_dxteex();
//		data.calc_snpstats();
//		if (p.vb_init_file != "NULL") {
//			data.read_alpha_mu();
//		}
//		SECTION("Ex4. Vbayes_X2 initialised correctly") {
//			CHECK(data.dXtEEX(0, 0) == Approx(0));
//			CHECK(data.dXtEEX(1, 0) == Approx(0));
//			CHECK(data.dXtEEX(2, 0) == Approx(0));
//			CHECK(data.dXtEEX(3, 0) == Approx(0));
//
//			CHECK(data.dXtEEX(0, 7) == Approx(-77.6736297077));
//			CHECK(data.dXtEEX(1, 7) == Approx(-65.7610340352));
//			CHECK(data.dXtEEX(2, 7) == Approx(-106.8630307306));
//			CHECK(data.dXtEEX(3, 7) == Approx(-61.8754581783));
//		}
//
//		data.calc_snpstats();
//		if (p.vb_init_file != "NULL") {
//			data.read_alpha_mu();
//		}
//
//		VBayesX2 VB(data);
//		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
//		VB.run_inference(VB.hyps_inits, false, 2, trackers);
//		SECTION("Ex3. Vbayes_X2 inference correct"){
//			CHECK(trackers[0].count == 30);
//			CHECK(trackers[0].logw == Approx(-1158.9633597738));
//		}
//	}
//
//	SECTION("Compute dxteex external") {
//		p.dxteex_file = "data/io_test/case8/dxteex_low_mem.txt";
//		Data data(p);
//
//		data.read_non_genetic_data();
//		data.standardise_non_genetic_data();
//		SECTION("Ex4. Non genetic data standardised + covars regressed") {
//			CHECK(data.E(0, 0) == Approx(0));
//		}
//		data.read_full_bgen();
//
//		data.read_external_dxteex();
//		data.calc_dxteex();
//		SECTION("Ex4. Vbayes_X2 initialised correctly") {
//			CHECK(data.dXtEEX(0, 0) == Approx(0));
//			CHECK(data.dXtEEX(1, 0) == Approx(0));
//			CHECK(data.dXtEEX(2, 0) == Approx(0));
//			CHECK(data.dXtEEX(3, 0) == Approx(0));
//
//			CHECK(data.dXtEEX(0, 7) == Approx(-77.6736297077));
//			CHECK(data.dXtEEX(1, 7) == Approx(-65.5542323344));
//			CHECK(data.dXtEEX(2, 7) == Approx(-106.8630307306));
//			CHECK(data.dXtEEX(3, 7) == Approx(-61.8862174056));
//		}
//
//		data.calc_snpstats();
//		if (p.vb_init_file != "NULL") {
//			data.read_alpha_mu();
//		}
//		VBayesX2 VB(data);
//		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
//		SECTION("Dump params state") {
//			// Set up for RunInnerLoop
//			long n_grid = VB.hyps_inits.size();
//			std::vector<Hyps> all_hyps = VB.hyps_inits;
//
//			// Set up for updateAllParams
//			std::vector<VariationalParameters> all_vp;
//			VB.setup_variational_params(all_hyps, all_vp);
//
//			int round_index = 2;
//			std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
//			std::vector<std::vector<double > > logw_updates(n_grid);
//			VariationalParameters &vp = all_vp[0];
//			Hyps &hyps = all_hyps[0];
//
//			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);
//			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);
//			VB.updateAllParams(2, round_index, all_vp, all_hyps, logw_prev);
//
//			VbTracker tracker(p);
//			tracker.init_interim_output(0, 2, VB.n_effects, VB.n_env, VB.env_names, vp);
//  tracker.dump_state(2, VB.n_samples, VB.n_covar, VB.n_var, VB.n_env,
//                     VB.n_effects, vp, hyps, VB.Y, VB.C, VB.X,
//                     VB.covar_names, VB.env_names);
// }
//		VB.run_inference(VB.hyps_inits, false, 2, trackers);
//		SECTION("Ex3. Vbayes_X2 inference correct"){
//			CHECK(trackers[0].count == 30);
//			CHECK(trackers[0].logw == Approx(-1158.9630661443));
//		}
//	}
//}

TEST_CASE( "Edge case 1: error in alpha" ){
	parameters p;

	SECTION("Ex1. No filters applied, low mem mode"){
		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb", (char*) "--low_mem",
			             (char*) "--mode_spike_slab",
			             (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
			             (char*) "--out", (char*) "data/io_test/fake_age.out",
			             (char*) "--pheno", (char*) "data/io_test/pheno.txt",
			             (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
			             (char*) "--hyps_probs", (char*) "data/io_test/hyperpriors_gxage_probs.txt",
			             (char*) "--environment", (char*) "data/io_test/age.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		Data data( p );
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		data.calc_dxteex();
		data.set_vb_init();
		VBayesX2 VB(data);

		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
		SECTION("Ex1. Explicitly checking updates"){

			// Set up for RunInnerLoop
			long n_grid = VB.hyps_inits.size();
			long n_samples = VB.n_samples;
			std::vector<Hyps> all_hyps = VB.hyps_inits;

			// Set up for updateAllParams
			std::vector<VariationalParameters> all_vp;
			VB.setup_variational_params(all_hyps, all_vp);
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			int round_index = 2;
			std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
			std::vector<std::vector< double > > logw_updates(n_grid);

			vp.alpha_beta(0) = std::nan("1");

			CHECK_THROWS(VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev));
		}
	}
}
