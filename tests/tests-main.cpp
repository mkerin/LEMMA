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

	Eigen::MatrixXd X(3, 3);
	Eigen::VectorXd v1(3), v2(3);
	X << 1, 2, 3,
		 4, 5, 6,
		 7, 8, 9;
	v1 << 1, 1, 1;
	v2 << 1, 2, 3;

	SECTION("dot product of vector with col vector"){
		CHECK((v1.dot(X.col(0))) == 12.0);
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
}

TEST_CASE( "Example 1: single-env" ){
	parameters p;

	SECTION("No filters applied, high mem mode"){
		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb",
						 (char*) "--interaction", (char*) "x",
						 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
						 (char*) "--out", (char*) "data/io_test/fake.out",
						 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
						 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
						 (char*) "--hyps_probs", (char*) "data/io_test/hyperpriors_gxage_probs.txt",
						 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
						 (char*) "--covar", (char*) "data/io_test/age.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		Data data( p );

		std::cout << "Data initialised" << std::endl;
		data.read_non_genetic_data();
		SECTION( "Raw non genetic data read in accurately"){
            CHECK(data.n_covar == 1);
            CHECK(data.n_env == 1);
			CHECK(data.n_pheno == 1);
			CHECK(data.n_samples == 50);
			CHECK(data.Y(0,0) == Approx(-1.18865038973338));
			CHECK(data.W(0,0) == Approx(-0.33472645347487201));
			CHECK(data.E(0, 0) == Approx(-0.33472645347487201));
			CHECK(data.hyps_grid(0,1) == Approx(0.317067781333932));
		}

		data.standardise_non_genetic_data();
		SECTION( "Non genetic data standardised + covars regressed"){
			CHECK(data.params.scale_pheno == true);
			CHECK(data.params.use_vb_on_covars == false);
			CHECK(data.params.covar_file != "NULL");
//			CHECK(data.Y(0,0) == Approx(-3.6676363273605137)); Centered
//			CHECK(data.Y(0,0) == Approx(-1.5800573524786081)); Scaled
			CHECK(data.Y(0,0) == Approx(-1.262491384814441));
			CHECK(data.Y2(0,0) == Approx(-1.262491384814441));
			CHECK(data.W(0,0) == Approx(-0.58947939694779772));
			CHECK(data.E(0,0) == Approx(-0.58947939694779772));
		}

		data.read_full_bgen();
		SECTION( "bgen read in & standardised correctly"){
			CHECK(data.G.low_mem == false);
			CHECK(data.params.low_mem == false);
            CHECK(data.params.flip_high_maf_variants == true);
			CHECK(data.G(0, 0) == Approx(1.8604233373));
		}

		SECTION( "Confirm calc_dxteex() reorders properly"){
		    data.params.dxteex_file = "data/io_test/inputs/dxteex_mixed.txt";
			data.read_external_dxteex();
            data.calc_dxteex();
            CHECK(data.dXtEEX(0, 0) == Approx(87.204591182113916));
            CHECK(data.n_dxteex_computed == 1);
		}

		data.calc_dxteex();
		if(p.vb_init_file != "NULL"){
			data.read_alpha_mu();
		}
		VBayesX2 VB(data);
		VB.check_inputs();
		SECTION("Vbayes_X2 initialised correctly"){
			CHECK(VB.n_samples == 50);
			CHECK(VB.N == 50.0);
			CHECK(VB.n_env == 1);
			CHECK(VB.n_covar == 1);
			CHECK(VB.n_effects == 2);
			CHECK(VB.vp_init.muw(0) == 1.0);
			CHECK(VB.p.init_weights_with_snpwise_scan == false);
			CHECK(VB.dXtEEX(0, 0) == Approx(87.204591182113916));
			CHECK(VB.Cty(0, 0) == Approx(-5.3290705182007514e-15));
		}

//		SECTION("Checking output"){
//		    VB.run();
//		}

		std::vector< VbTracker > trackers(p.n_thread);
		VB.run_inference(VB.hyps_grid, false, 2, trackers);
		SECTION("Vbayes_X2 inference correct"){
			CHECK(trackers[0].counts_list[0] == 11);
			CHECK(trackers[0].counts_list[3] == 33);
			CHECK(trackers[0].logw_list[0] == Approx(-60.461680267318215));
			CHECK(trackers[0].logw_list[1] == Approx(-59.551958432875722));
			CHECK(trackers[0].logw_list[2] == Approx(-59.461267092637925));
			CHECK(trackers[0].logw_list[3] == Approx(-59.761370335497297));
		}
	}
}

//TEST_CASE( "vbayes_x2.hpp", "[VBayesX2]" ) {
//	parameters p;
//	char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb",
//					 (char*) "--interaction", (char*) "x",
//					 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
//					 (char*) "--out", (char*) "data/io_test/fake.out",
//					 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
//					 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
//					 (char*) "--hyps_probs", (char*) "data/io_test/hyperpriors_gxage_probs.txt",
//					 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
//					 (char*) "--covar", (char*) "data/io_test/age.txt"};
//	int argc = sizeof(argv)/sizeof(argv[0]);
//	parse_arguments(p, argc, argv);
//	Data data( p );
//	data.read_non_genetic_data();
//	data.standardise_non_genetic_data();
//	data.read_full_bgen();
//
//	// Pass data to VBayes object
//	VBayesX2 VB(data);
//
//	// SECTION()
//	CHECK(VB.n_grid == 7);
//	VB.check_inputs();
//	CHECK(VB.n_grid == 6);
//	CHECK(VB.hyps_grid.rows() == 6);
//	CHECK(VB.probs_grid.rows() == 6);
//
//	SECTION("Function to validate hyperparameter grid"){
//			int n_var = 50;
//			Eigen::MatrixXd orig(3, 5), attempt, answer(2, 5);
//			std::vector<int> attempt_vec, answer_vec;
//
//			// Filling answers
//			orig << 1, 0.1, 0.1, 0.1, 0.1,
//					1, 0.1, 0.1, 0.001, 0.1,
//					1, 0.1, 0.1, 0.1, 0.1;
//			answer << 1, 0.1, 0.1, 0.1, 0.1,
//					  1, 0.1, 0.1, 0.1, 0.1;
//			answer_vec.push_back(0);
//			answer_vec.push_back(2);
//
//			CHECK(validate_grid(orig, n_var) == answer_vec);
//			CHECK(subset_matrix(orig, answer_vec) == answer);
//	}
//}
//
