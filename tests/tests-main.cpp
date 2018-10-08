// tests-main.cpp
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sys/stat.h>
#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes_x2.hpp"
#include "../src/data.hpp"
#include "../src/genotype_matrix.hpp"


TEST_CASE( "Algebra in Eigen3" ) {

	Eigen::MatrixXd X(3, 3), X2;
	Eigen::VectorXd v1(3), v2(3);
	X << 1, 2, 3,
		 4, 5, 6,
		 7, 8, 9;
	v1 << 1, 1, 1;
	v2 << 1, 2, 3;
	X2 = X.rowwise().reverse();

	SECTION("dot product of vector with col vector"){
		CHECK((v1.dot(X.col(0))) == 12.0);
	}

	SECTION("Eigen reverses columns as expected"){
		Eigen::MatrixXd res(3, 3);
		res << 3, 2, 1,
			   6, 5, 4,
			   9, 8, 7;
		CHECK(X2 == res);
	}
	
	SECTION("coefficient-wise product between vectors"){
		Eigen::VectorXd res(3);
		res << 1, 2, 3;
		CHECK((v1.array() * v2.array()).matrix() == res);
		CHECK(v1.cwiseProduct(v2) == res);
	}
	
	SECTION("coefficient-wise subtraction between vectors"){
		Eigen::VectorXd res(3);
		res << 0, 1, 2;
		CHECK((v2 - v1) == res);
	}

	SECTION("Check .sum() function"){
		Eigen::VectorXd res(3);
		res << 1, 2, 3;
		CHECK(res.sum() == 6);
	}
	
	SECTION("Sum of NaN returns NaN"){
		Eigen::VectorXd res(3);
		res << 1, std::numeric_limits<double>::quiet_NaN(), 3;
		CHECK(std::isnan(res.sum()));
	}

	SECTION("Ref of columns working correctly"){
		Eigen::Ref<Eigen::VectorXd> y1 = X.col(0);
		CHECK(y1(0) == 1);
		CHECK(y1(1) == 4);
		CHECK(y1(2) == 7);
		X = X + X;
		CHECK(y1(0) == 2);
		CHECK(y1(1) == 8);
		CHECK(y1(2) == 14);
	}
}
//
//TEST_CASE( "Example 1: single-env" ){
//	parameters p;
//
//	SECTION("Ex1. No filters applied, high mem mode"){
//		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb",
//						 (char*) "--interaction", (char*) "x",
//						 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
//						 (char*) "--out", (char*) "data/io_test/fake_age.out",
//						 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
//						 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
//						 (char*) "--hyps_probs", (char*) "data/io_test/hyperpriors_gxage_probs.txt",
//						 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
//						 (char*) "--covar", (char*) "data/io_test/age.txt"};
//		int argc = sizeof(argv)/sizeof(argv[0]);
//		parse_arguments(p, argc, argv);
//		Data data( p );
//
//		std::cout << "Data initialised" << std::endl;
//		data.read_non_genetic_data();
//		SECTION( "Ex1. Raw non genetic data read in accurately"){
//            CHECK(data.n_covar == 1);
//            CHECK(data.n_env == 1);
//			CHECK(data.n_pheno == 1);
//			CHECK(data.n_samples == 50);
//			CHECK(data.Y(0,0) == Approx(-1.18865038973338));
//			CHECK(data.W(0,0) == Approx(-0.33472645347487201));
//			CHECK(data.E(0, 0) == Approx(-0.33472645347487201));
//			CHECK(data.hyps_grid(0,1) == Approx(0.317067781333932));
//		}
//
//		data.standardise_non_genetic_data();
//		SECTION( "Ex1. Non genetic data standardised + covars regressed"){
//			CHECK(data.params.scale_pheno == true);
//			CHECK(data.params.use_vb_on_covars == false);
//			CHECK(data.params.covar_file != "NULL");
////			CHECK(data.Y(0,0) == Approx(-3.6676363273605137)); Centered
////			CHECK(data.Y(0,0) == Approx(-1.5800573524786081)); Scaled
//			CHECK(data.Y(0,0) == Approx(-1.262491384814441));
//			CHECK(data.Y2(0,0) == Approx(-1.262491384814441));
//			CHECK(data.W(0,0) == Approx(-0.58947939694779772));
//			CHECK(data.E(0,0) == Approx(-0.58947939694779772));
//		}
//
//		data.read_full_bgen();
//		SECTION( "Ex1. bgen read in & standardised correctly"){
//			CHECK(data.G.low_mem == false);
//			CHECK(data.params.low_mem == false);
//            CHECK(data.params.flip_high_maf_variants == true);
//			CHECK(data.G(0, 0) == Approx(1.8604233373));
//		}
//
//		SECTION( "Ex1. Confirm calc_dxteex() reorders properly"){
//		    data.params.dxteex_file = "data/io_test/inputs/dxteex_mixed.txt";
//			data.read_external_dxteex();
//            data.calc_dxteex();
//            CHECK(data.dXtEEX(0, 0) == Approx(87.204591182113916));
//            CHECK(data.n_dxteex_computed == 1);
//		}
//
//		data.calc_dxteex();
//		if(p.vb_init_file != "NULL"){
//			data.read_alpha_mu();
//		}
//		VBayesX2 VB(data);
//		VB.check_inputs();
//		SECTION("Ex1. Vbayes_X2 initialised correctly"){
//			CHECK(VB.n_samples == 50);
//			CHECK(VB.N == 50.0);
//			CHECK(VB.n_env == 1);
//			CHECK(VB.n_covar == 1);
//			CHECK(VB.n_effects == 2);
//			CHECK(VB.vp_init.muw(0) == 1.0);
//			CHECK(VB.p.init_weights_with_snpwise_scan == false);
//			CHECK(VB.dXtEEX(0, 0) == Approx(87.204591182113916));
//			CHECK(VB.Cty(0, 0) == Approx(-5.3290705182007514e-15));
//		}
//
////		SECTION("Checking output"){
////		    VB.run();
////		}
//
//        SECTION("Ex1. Explicitly checking updates"){
//			// Set up for RunInnerLoop
//			int n_effects = 2;
//			int n_env = 1;
//			int n_grid = VB.n_grid;
//			std::vector<Hyps> all_hyps(n_grid);
//			for (int ii = 0; ii < n_grid; ii++) {
//				double sigma = VB.hyps_grid(ii, VB.sigma_ind);
//				double sigma_b = VB.hyps_grid(ii, VB.sigma_b_ind);
//				double sigma_g = VB.hyps_grid(ii, VB.sigma_g_ind);
//				double lam_b = VB.hyps_grid(ii, VB.lam_b_ind);
//				double lam_g = VB.hyps_grid(ii, VB.lam_g_ind);
//				all_hyps[ii].slab_var.resize(n_effects);
//				all_hyps[ii].spike_var.resize(n_effects);
//				all_hyps[ii].slab_relative_var.resize(n_effects);
//				all_hyps[ii].spike_relative_var.resize(n_effects);
//				all_hyps[ii].lambda.resize(n_effects);
//				all_hyps[ii].s_x.resize(2);
//
//				Eigen::ArrayXd muw_sq(n_env * n_env);
//				for (int ll = 0; ll < n_env; ll++) {
//					for (int mm = 0; mm < n_env; mm++) {
//						muw_sq(mm * n_env + ll) = VB.vp_init.muw(mm) * VB.vp_init.muw(ll);
//					}
//				}
//				//
//				all_hyps[ii].sigma = sigma;
//				all_hyps[ii].slab_var << sigma * sigma_b, sigma * sigma_g;
//				all_hyps[ii].spike_var << sigma * sigma_b / VB.spike_diff_factor, sigma * sigma_g / VB.spike_diff_factor;
//				all_hyps[ii].slab_relative_var << sigma_b, sigma_g;
//				all_hyps[ii].spike_relative_var << sigma_b / VB.spike_diff_factor, sigma_g / VB.spike_diff_factor;
//				all_hyps[ii].lambda << lam_b, lam_g;
//				all_hyps[ii].s_x << VB.n_var, (VB.dXtEEX.rowwise() * muw_sq.transpose()).sum() / (VB.N - 1.0);
//			}
//
//			// Set up for updateAllParams
//			std::vector<VariationalParameters> all_vp(n_grid);
//			for (int nn = 0; nn < n_grid; nn++) {
//				all_vp[nn].init_from_lite(VB.vp_init, p);
//				VB.updateSSq(all_hyps[nn], all_vp[nn]);
//				all_vp[nn].calcEdZtZ(VB.dXtEEX, n_env);
//			}
//			int round_index = 2;
//			std::vector<double> logw_prev(n_grid, -1);
//			std::vector<std::vector< double >> logw_updates(n_grid);
//
//			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev, logw_updates);
//
//			CHECK(all_vp[0].alpha_beta(0) == Approx(0.14485896221944508));
//			CHECK(all_vp[0].alpha_beta(1) == Approx(0.15184033622793655));
//			CHECK(all_vp[0].alpha_beta(63) == Approx(0.17836527480696865));
//			CHECK(VB.calc_logw(all_hyps[0], all_vp[0]) == Approx(-60.9810031189));
//
//			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev, logw_updates);
//
//			CHECK(all_vp[0].alpha_beta(0) == Approx(0.1351221581));
//			CHECK(all_vp[0].alpha_beta(1) == Approx(0.1401495216));
//			CHECK(all_vp[0].alpha_beta(63) == Approx(0.1769087833));
//			CHECK(VB.calc_logw(all_hyps[0], all_vp[0]) == Approx(-60.6030355156));
//		}
//
//		std::vector< VbTracker > trackers(p.n_thread);
//		VB.run_inference(VB.hyps_grid, false, 2, trackers);
//		SECTION("Ex1. Vbayes_X2 inference correct"){
//			CHECK(trackers[0].counts_list[0] == 11);
//			CHECK(trackers[0].counts_list[3] == 33);
//			CHECK(trackers[0].logw_list[0] == Approx(-60.5189122149));
//			CHECK(trackers[0].logw_list[1] == Approx(-59.9653759921));
//			CHECK(trackers[0].logw_list[2] == Approx(-60.3020600309));
//			CHECK(trackers[0].logw_list[3] == Approx(-61.0641461747));
//		}
//	}
//}

//TEST_CASE( "Example 2: multi-env" ){
//	parameters p;
//
//	SECTION("Ex2. No filters applied, high mem mode"){
//		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb",
//						 (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
//						 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
//						 (char*) "--out", (char*) "data/io_test/fake_env.out",
//						 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
//						 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
//						 (char*) "--hyps_probs", (char*) "data/io_test/hyperpriors_gxage_probs.txt",
//						 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
//						 (char*) "--covar", (char*) "data/io_test/n50_p100_env.txt"};
//		int argc = sizeof(argv)/sizeof(argv[0]);
//		parse_arguments(p, argc, argv);
//		Data data( p );
//
//		data.read_non_genetic_data();
//		data.standardise_non_genetic_data();
//		data.read_full_bgen();
//
//		data.calc_dxteex();
//        data.calc_snpstats();
//		if(p.vb_init_file != "NULL"){
//			data.read_alpha_mu();
//		}
//		VBayesX2 VB(data);
//		VB.check_inputs();
//		SECTION("Ex2. Vbayes_X2 initialised correctly"){
//			CHECK(VB.n_samples == 50);
//			CHECK(VB.N == 50.0);
//			CHECK(VB.n_env == 4);
//			CHECK(VB.n_covar == 4);
//			CHECK(VB.n_effects == 2);
//			CHECK(VB.vp_init.muw(0) == 0.25);
//			CHECK(VB.p.init_weights_with_snpwise_scan == false);
//			CHECK(VB.dXtEEX(0, 0) == Approx(38.9390135703));
//		}
//
//		SECTION("Ex2. Explicitly checking updates"){
//			// Set up for RunInnerLoop
//			int n_effects = 2;
//			int n_env = 4;
//			int n_grid = VB.n_grid;
//			std::vector<Hyps> all_hyps(n_grid);
//			for (int ii = 0; ii < n_grid; ii++) {
//				double sigma = VB.hyps_grid(ii, VB.sigma_ind);
//				double sigma_b = VB.hyps_grid(ii, VB.sigma_b_ind);
//				double sigma_g = VB.hyps_grid(ii, VB.sigma_g_ind);
//				double lam_b = VB.hyps_grid(ii, VB.lam_b_ind);
//				double lam_g = VB.hyps_grid(ii, VB.lam_g_ind);
//				all_hyps[ii].slab_var.resize(n_effects);
//				all_hyps[ii].spike_var.resize(n_effects);
//				all_hyps[ii].slab_relative_var.resize(n_effects);
//				all_hyps[ii].spike_relative_var.resize(n_effects);
//				all_hyps[ii].lambda.resize(n_effects);
//				all_hyps[ii].s_x.resize(2);
//
//				Eigen::ArrayXd muw_sq(n_env * n_env);
//				for (int ll = 0; ll < n_env; ll++) {
//					for (int mm = 0; mm < n_env; mm++) {
//						muw_sq(mm * n_env + ll) = VB.vp_init.muw(mm) * VB.vp_init.muw(ll);
//					}
//				}
//				//
//				all_hyps[ii].sigma = sigma;
//				all_hyps[ii].slab_var << sigma * sigma_b, sigma * sigma_g;
//				all_hyps[ii].spike_var << sigma * sigma_b / VB.spike_diff_factor, sigma * sigma_g / VB.spike_diff_factor;
//				all_hyps[ii].slab_relative_var << sigma_b, sigma_g;
//				all_hyps[ii].spike_relative_var << sigma_b / VB.spike_diff_factor, sigma_g / VB.spike_diff_factor;
//				all_hyps[ii].lambda << lam_b, lam_g;
//				all_hyps[ii].s_x << VB.n_var, (VB.dXtEEX.rowwise() * muw_sq.transpose()).sum() / (VB.N - 1.0);
//			}
//
//			// Set up for updateAllParams
//			std::vector<VariationalParameters> all_vp(n_grid);
//			for (int nn = 0; nn < n_grid; nn++) {
//				all_vp[nn].init_from_lite(VB.vp_init, p);
//				VB.updateSSq(all_hyps[nn], all_vp[nn]);
//				all_vp[nn].calcEdZtZ(VB.dXtEEX, n_env);
//			}
//			int round_index = 2;
//			std::vector<double> logw_prev(n_grid, -1);
//			std::vector<std::vector< double >> logw_updates(n_grid);
//
//			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev, logw_updates);
//
//			CHECK(all_vp[0].alpha_beta(0) == Approx(0.1339907047));
//			CHECK(all_vp[0].alpha_beta(1) == Approx(0.1393645403 ));
//			CHECK(all_vp[0].alpha_beta(63) == Approx(0.1700976171));
//			CHECK(all_vp[0].alpha_gam(0) == Approx(0.1351102326));
//			CHECK(all_vp[0].alpha_gam(1) == Approx(0.1349464317));
//			CHECK(all_vp[0].alpha_gam(63) == Approx(0.1351214237));
//			CHECK(all_vp[0].muw(0, 0) == Approx(0.1096760209));
//			CHECK(all_vp[0].yx(0) == Approx(-0.02111226));
//			CHECK(all_vp[0].ym(0) == Approx(-0.3874879589));
//			CHECK(VB.calc_logw(all_hyps[0], all_vp[0]) == Approx(-68.2656816517));
//
//			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev, logw_updates);
//
//			CHECK(all_vp[0].alpha_gam(63) == Approx(0.12404782));
//			CHECK(all_vp[0].alpha_gam(1) == Approx(0.1244627819));
//			CHECK(all_vp[0].alpha_gam(0) == Approx(0.1228313573));
//			CHECK(all_vp[0].alpha_beta(63) == Approx(0.1704601589));
//			CHECK(all_vp[0].alpha_beta(1) == Approx(0.1326174323));
//			CHECK(all_vp[0].alpha_beta(0) == Approx(0.1292192489));
//			CHECK(all_vp[0].muw(0, 0) == Approx(0.0455626691));
//			CHECK(all_vp[0].yx(0) == Approx(-0.0071638495));
//			CHECK(all_vp[0].ym(0) == Approx(-0.26284773569));
//			CHECK(VB.calc_logw(all_hyps[0], all_vp[0]) == Approx(-67.6870841008));
//		}
//
//		std::vector< VbTracker > trackers(p.n_thread);
//		VB.run_inference(VB.hyps_grid, false, 2, trackers);
//		SECTION("Ex2. Vbayes_X2 inference correct"){
//			CHECK(trackers[0].counts_list[0] == 11);
//			CHECK(trackers[0].counts_list[3] == 35);
//			CHECK(trackers[0].logw_list[0] == Approx(-67.6055600008));
//			CHECK(trackers[0].logw_list[1] == Approx(-67.3497693394));
//			CHECK(trackers[0].logw_list[2] == Approx(-67.757622793));
//			CHECK(trackers[0].logw_list[3] == Approx(-68.5048150566));
//		}
//	}
//}

TEST_CASE( "Example 3: multi-env w/ covars" ){
	parameters p;

	SECTION("Ex3. No filters applied, high mem mode"){
		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb",
						 (char*) "--use_vb_on_covars",
						 (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
						 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
						 (char*) "--out", (char*) "data/io_test/fake_env.out",
						 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
						 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
						 (char*) "--hyps_probs", (char*) "data/io_test/hyperpriors_gxage_probs.txt",
						 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
						 (char*) "--covar", (char*) "data/io_test/n50_p100_env.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		Data data( p );

		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		SECTION( "Ex3. Non genetic data standardised + covars regressed"){
			CHECK(data.params.scale_pheno == true);
			CHECK(data.params.use_vb_on_covars == true);
			CHECK(data.params.covar_file != "NULL");
//			CHECK(data.Y(0,0) == Approx(-3.6676363273605137)); Centered
			CHECK(data.Y(0,0) == Approx(-1.5800573524786081)); // Scaled
			CHECK(data.Y2(0,0) == Approx(-1.5567970303));
			CHECK(data.W(0,0) == Approx(0.8957059881));
			CHECK(data.E(0,0) == Approx(0.8957059881));
		}
		data.read_full_bgen();

		data.calc_dxteex();
		data.calc_snpstats();
		if(p.vb_init_file != "NULL"){
			data.read_alpha_mu();
		}
		VBayesX2 VB(data);
		VB.check_inputs();
		SECTION("Ex3. Vbayes_X2 initialised correctly"){
			CHECK(VB.n_samples == 50);
			CHECK(VB.N == 50.0);
			CHECK(VB.n_env == 4);
			CHECK(VB.n_covar == 4);
			CHECK(VB.n_effects == 2);
			CHECK(VB.vp_init.muw(0) == 0.25);
			CHECK(VB.p.init_weights_with_snpwise_scan == false);
			CHECK(VB.dXtEEX(0, 0) == Approx(38.9390135703));
		}

		SECTION("Ex3. Explicitly checking updates"){
			// Set up for RunInnerLoop
			int n_effects = 2;
			int n_env = 4;
			int n_grid = VB.n_grid;
			std::vector<Hyps> all_hyps(n_grid);
			for (int ii = 0; ii < n_grid; ii++) {
				double sigma = VB.hyps_grid(ii, VB.sigma_ind);
				double sigma_b = VB.hyps_grid(ii, VB.sigma_b_ind);
				double sigma_g = VB.hyps_grid(ii, VB.sigma_g_ind);
				double lam_b = VB.hyps_grid(ii, VB.lam_b_ind);
				double lam_g = VB.hyps_grid(ii, VB.lam_g_ind);
				all_hyps[ii].slab_var.resize(n_effects);
				all_hyps[ii].spike_var.resize(n_effects);
				all_hyps[ii].slab_relative_var.resize(n_effects);
				all_hyps[ii].spike_relative_var.resize(n_effects);
				all_hyps[ii].lambda.resize(n_effects);
				all_hyps[ii].s_x.resize(2);

				Eigen::ArrayXd muw_sq(n_env * n_env);
				for (int ll = 0; ll < n_env; ll++) {
					for (int mm = 0; mm < n_env; mm++) {
						muw_sq(mm * n_env + ll) = VB.vp_init.muw(mm) * VB.vp_init.muw(ll);
					}
				}
				//
				all_hyps[ii].sigma = sigma;
				all_hyps[ii].slab_var << sigma * sigma_b, sigma * sigma_g;
				all_hyps[ii].spike_var << sigma * sigma_b / VB.spike_diff_factor, sigma * sigma_g / VB.spike_diff_factor;
				all_hyps[ii].slab_relative_var << sigma_b, sigma_g;
				all_hyps[ii].spike_relative_var << sigma_b / VB.spike_diff_factor, sigma_g / VB.spike_diff_factor;
				all_hyps[ii].lambda << lam_b, lam_g;
				all_hyps[ii].s_x << VB.n_var, (VB.dXtEEX.rowwise() * muw_sq.transpose()).sum() / (VB.N - 1.0);
			}

			// Set up for updateAllParams
			std::vector<VariationalParameters> all_vp(n_grid);
			for (int nn = 0; nn < n_grid; nn++) {
				all_vp[nn].init_from_lite(VB.vp_init, p);
				VB.updateSSq(all_hyps[nn], all_vp[nn]);
				all_vp[nn].calcEdZtZ(VB.dXtEEX, n_env);
			}
			int round_index = 2;
			std::vector<double> logw_prev(n_grid, -1);
			std::vector<std::vector< double >> logw_updates(n_grid);

			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev, logw_updates);

			CHECK(all_vp[0].muc(0) == Approx(0.1221946024));
			CHECK(all_vp[0].muc(3) == Approx(-0.1595909887));
			CHECK(all_vp[0].alpha_beta(0) == Approx(0.1339235799));
			CHECK(all_vp[0].alpha_beta(1) == Approx(0.1415361555));
			CHECK(all_vp[0].alpha_beta(63) == Approx(0.1724736345));
			CHECK(all_vp[0].muw(0, 0) == Approx(0.1127445891));
			CHECK(VB.calc_logw(all_hyps[0], all_vp[0]) == Approx(-20076.0449393003));

			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev, logw_updates);

			CHECK(all_vp[0].muc(0) == Approx(0.1463805515));
			CHECK(all_vp[0].muc(3) == Approx(-0.1128544804));
			CHECK(all_vp[0].alpha_beta(0) == Approx(0.1292056073));
			CHECK(all_vp[0].alpha_beta(1) == Approx(0.1338797264));
			CHECK(all_vp[0].alpha_beta(63) == Approx(0.1730150924));
			CHECK(all_vp[0].muw(0, 0) == Approx(0.0460748751));
			CHECK(VB.calc_logw(all_hyps[0], all_vp[0]) == Approx(-20075.3681431899));
		}

		std::vector< VbTracker > trackers(p.n_thread);
		VB.run_inference(VB.hyps_grid, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].counts_list[0] == 10);
			CHECK(trackers[0].counts_list[3] == 33);
			CHECK(trackers[0].logw_list[0] == Approx(-20075.279701215));
			CHECK(trackers[0].logw_list[1] == Approx(-20074.9040629751));
			CHECK(trackers[0].logw_list[2] == Approx(-20075.2341616388));
			CHECK(trackers[0].logw_list[3] == Approx(-20075.9304539824));
		}
	}
}
//
//TEST_CASE( "Example 4: multi-env w/ 2 threads" ){
//	parameters p;
//
//	SECTION("Ex4. No filters applied, high mem mode"){
//		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb",
//						 (char*) "--threads", (char*) "2",
//						 (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
//						 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
//						 (char*) "--out", (char*) "data/io_test/fake_env.out",
//						 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
//						 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
//						 (char*) "--hyps_probs", (char*) "data/io_test/hyperpriors_gxage_probs.txt",
//						 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
//						 (char*) "--covar", (char*) "data/io_test/n50_p100_env.txt"};
//		int argc = sizeof(argv)/sizeof(argv[0]);
//		parse_arguments(p, argc, argv);
//		Data data( p );
//
//		data.read_non_genetic_data();
//		data.standardise_non_genetic_data();
//		data.read_full_bgen();
//
//		data.calc_dxteex();
//		data.calc_snpstats();
//		if(p.vb_init_file != "NULL"){
//			data.read_alpha_mu();
//		}
//		VBayesX2 VB(data);
//		VB.check_inputs();
//		SECTION("Ex4. Vbayes_X2 initialised correctly"){
//			CHECK(VB.n_samples == 50);
//			CHECK(VB.N == 50.0);
//			CHECK(VB.n_env == 4);
//			CHECK(VB.n_covar == 4);
//			CHECK(VB.n_effects == 2);
//			CHECK(VB.vp_init.muw(0) == 0.25);
//			CHECK(VB.p.init_weights_with_snpwise_scan == false);
//			CHECK(VB.dXtEEX(0, 0) == Approx(38.9390135703));
//		}
//
//		SECTION("Ex4. Explicitly checking updates"){
//			// Set up for RunInnerLoop
//			int n_effects = 2;
//			int n_env = 4;
//			int n_grid = VB.n_grid;
//			std::vector<Hyps> all_hyps(n_grid);
//			for (int ii = 0; ii < n_grid; ii++) {
//				double sigma = VB.hyps_grid(ii, VB.sigma_ind);
//				double sigma_b = VB.hyps_grid(ii, VB.sigma_b_ind);
//				double sigma_g = VB.hyps_grid(ii, VB.sigma_g_ind);
//				double lam_b = VB.hyps_grid(ii, VB.lam_b_ind);
//				double lam_g = VB.hyps_grid(ii, VB.lam_g_ind);
//				all_hyps[ii].slab_var.resize(n_effects);
//				all_hyps[ii].spike_var.resize(n_effects);
//				all_hyps[ii].slab_relative_var.resize(n_effects);
//				all_hyps[ii].spike_relative_var.resize(n_effects);
//				all_hyps[ii].lambda.resize(n_effects);
//				all_hyps[ii].s_x.resize(2);
//
//				Eigen::ArrayXd muw_sq(n_env * n_env);
//				for (int ll = 0; ll < n_env; ll++) {
//					for (int mm = 0; mm < n_env; mm++) {
//						muw_sq(mm * n_env + ll) = VB.vp_init.muw(mm) * VB.vp_init.muw(ll);
//					}
//				}
//				//
//				all_hyps[ii].sigma = sigma;
//				all_hyps[ii].slab_var << sigma * sigma_b, sigma * sigma_g;
//				all_hyps[ii].spike_var << sigma * sigma_b / VB.spike_diff_factor, sigma * sigma_g / VB.spike_diff_factor;
//				all_hyps[ii].slab_relative_var << sigma_b, sigma_g;
//				all_hyps[ii].spike_relative_var << sigma_b / VB.spike_diff_factor, sigma_g / VB.spike_diff_factor;
//				all_hyps[ii].lambda << lam_b, lam_g;
//				all_hyps[ii].s_x << VB.n_var, (VB.dXtEEX.rowwise() * muw_sq.transpose()).sum() / (VB.N - 1.0);
//			}
//
//			// Set up for updateAllParams
//			std::vector<VariationalParameters> all_vp(n_grid);
//			for (int nn = 0; nn < n_grid; nn++) {
//				all_vp[nn].init_from_lite(VB.vp_init, p);
//				VB.updateSSq(all_hyps[nn], all_vp[nn]);
//				all_vp[nn].calcEdZtZ(VB.dXtEEX, n_env);
//			}
//			int round_index = 2;
//			std::vector<double> logw_prev(n_grid, -1);
//			std::vector<std::vector< double >> logw_updates(n_grid);
//
//			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev, logw_updates);
//
//			CHECK(all_vp[0].alpha_beta(0) == Approx(0.1339907047));
//			CHECK(all_vp[0].alpha_beta(1) == Approx(0.1393645403 ));
//			CHECK(all_vp[0].alpha_beta(63) == Approx(0.1700976171));
//			CHECK(all_vp[0].alpha_gam(0) == Approx(0.1351102326));
//			CHECK(all_vp[0].alpha_gam(1) == Approx(0.1349464317));
//			CHECK(all_vp[0].alpha_gam(63) == Approx(0.1351214237));
//			CHECK(all_vp[0].muw(0, 0) == Approx(0.1096760209));
//			CHECK(all_vp[0].yx(0) == Approx(-0.02111226));
//			CHECK(all_vp[0].ym(0) == Approx(-0.3874879589));
//			CHECK(VB.calc_logw(all_hyps[0], all_vp[0]) == Approx(-68.2656816517));
//
//			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev, logw_updates);
//
//			CHECK(all_vp[0].alpha_gam(63) == Approx(0.12404782));
//			CHECK(all_vp[0].alpha_gam(1) == Approx(0.1244627819));
//			CHECK(all_vp[0].alpha_gam(0) == Approx(0.1228313573));
//			CHECK(all_vp[0].alpha_beta(63) == Approx(0.1704601589));
//			CHECK(all_vp[0].alpha_beta(1) == Approx(0.1326174323));
//			CHECK(all_vp[0].alpha_beta(0) == Approx(0.1292192489));
//			CHECK(all_vp[0].muw(0, 0) == Approx(0.0455626691));
//			CHECK(all_vp[0].yx(0) == Approx(-0.0071638495));
//			CHECK(all_vp[0].ym(0) == Approx(-0.26284773569));
//			CHECK(VB.calc_logw(all_hyps[0], all_vp[0]) == Approx(-67.6870841008));
//		}
//
//		std::vector< VbTracker > trackers(p.n_thread);
//		VB.run_inference(VB.hyps_grid, false, 2, trackers);
//		SECTION("Ex4. Vbayes_X2 inference correct"){
//			CHECK(trackers[0].counts_list[0] == 11);
//			CHECK(trackers[0].counts_list[3] == 35);
//			CHECK(trackers[0].logw_list[0] == Approx(-67.6055600008));
//			CHECK(trackers[0].logw_list[1] == Approx(-67.3497693394));
//			CHECK(trackers[0].logw_list[2] == Approx(-67.757622793));
//			CHECK(trackers[0].logw_list[3] == Approx(-68.5048150566));
//		}
//	}
//}

