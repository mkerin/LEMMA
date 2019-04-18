// tests-main.cpp
#define EIGEN_USE_MKL_ALL
#include "catch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <sys/stat.h>
#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes_x2.hpp"
#include "../src/data.hpp"
#include "../src/hyps.hpp"
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

	SECTION("Conservative Resize"){
		std::vector<int> keep;
		keep.push_back(1);
		for (std::size_t i = 0; i < keep.size(); i++) {
			X.col(i) = X.col(keep[i]);
		}
		X.conservativeResize(X.rows(), keep.size());

		CHECK(X.rows() == 3);
		CHECK(X.cols() == 1);
		CHECK(X(0, 0) == 2);
	}

	SECTION("selfAdjoit views"){
		Eigen::MatrixXd m3(3, 3);
		m3.triangularView<Eigen::StrictlyUpper>() = X.transpose() * X;
		CHECK(m3(0, 1) == 78);
	}

	SECTION("colwise subtraction between vector and matrix"){
		Eigen::MatrixXd res;
		res = -1*(X.colwise() - v1);
		CHECK(res(0, 0) == 0);
		CHECK(res.rows() == 3);
		CHECK(res.cols() == 3);
	}
}

TEST_CASE("Data") {
	parameters p;

	p.env_file = "data/io_test/n50_p100_env.txt";
	p.pheno_file = "data/io_test/pheno.txt";

	SECTION("n50_p100.bgen (low mem) w covars") {
		p.covar_file = "data/io_test/age.txt";
		p.bgen_file = "data/io_test/n50_p100.bgen";
		p.bgi_file = "data/io_test/n50_p100.bgen.bgi";
		p.low_mem = true;
		Data data(p);

		data.read_non_genetic_data();
		CHECK(data.n_env == 4);
		CHECK(data.E(0, 0) == Approx(0.785198212));

		data.standardise_non_genetic_data();
		CHECK(data.params.use_vb_on_covars);
		CHECK(data.E(0, 0) == Approx(0.9959851422));

		data.read_full_bgen();
		SECTION("Ex1. bgen read in & standardised correctly") {
			CHECK(data.G.low_mem);
			CHECK(data.params.low_mem);
			CHECK(!data.params.flip_high_maf_variants);
			CHECK(data.G(0, 0) == Approx(-1.8575040711));
			CHECK(data.G(0, 1) == Approx(-0.7404793547));
			CHECK(data.G(0, 2) == Approx(-0.5845122102));
			CHECK(data.G(0, 3) == Approx(-0.6633007506));
			CHECK(data.n_var == 67);
		}

		SECTION("dXtEEX computed correctly") {
			data.calc_dxteex();
			CHECK(data.dXtEEX(0, 0) == Approx(42.2994405499));
			CHECK(data.dXtEEX(1, 0) == Approx(43.2979303929));
			CHECK(data.dXtEEX(2, 0) == Approx(37.6440444004));
			CHECK(data.dXtEEX(3, 0) == Approx(40.9258647207));

			CHECK(data.dXtEEX(0, 4) == Approx(-4.0453940676));
			CHECK(data.dXtEEX(1, 4) == Approx(-15.6140263169));
			CHECK(data.dXtEEX(2, 4) == Approx(-13.2508795732));
			CHECK(data.dXtEEX(3, 4) == Approx(-9.8081456731));
		}
	}

	SECTION("n50_p100.bgen (low mem), covars, sample subset") {
		p.covar_file = "data/io_test/age.txt";
		p.bgen_file = "data/io_test/n50_p100.bgen";
		p.bgi_file = "data/io_test/n50_p100.bgen.bgi";
		p.incl_sids_file = "data/io_test/sample_ids.txt";
		p.low_mem = true;
		Data data(p);

		data.read_non_genetic_data();
		CHECK(data.n_env == 4);
		CHECK(data.E(0, 0) == Approx(0.785198212));

		data.standardise_non_genetic_data();
		CHECK(data.params.use_vb_on_covars);
		CHECK(data.E(0, 0) == Approx(0.8123860763));

		data.read_full_bgen();

		SECTION("dXtEEX computed correctly") {
			data.calc_dxteex();
			CHECK(data.dXtEEX(0, 0) == Approx(23.2334219303));
			CHECK(data.dXtEEX(1, 0) == Approx(27.9920667408));
			CHECK(data.dXtEEX(2, 0) == Approx(24.7041225993));
			CHECK(data.dXtEEX(3, 0) == Approx(24.2423580715));

			CHECK(data.dXtEEX(0, 4) == Approx(-1.056112897));
			CHECK(data.dXtEEX(1, 4) == Approx(-8.526431457));
			CHECK(data.dXtEEX(2, 4) == Approx(-6.5950206611));
			CHECK(data.dXtEEX(3, 4) == Approx(-3.6842212598));
		}
	}

	SECTION("n50_p100.bgen (low mem) + non genetic data") {
		p.bgen_file = "data/io_test/n50_p100.bgen";
		p.bgi_file = "data/io_test/n50_p100.bgen.bgi";
		p.low_mem = true;
		Data data(p);

		data.read_non_genetic_data();
		SECTION("Ex1. Raw non genetic data read in accurately") {
			CHECK(data.n_env == 4);
			CHECK(data.n_pheno == 1);
			CHECK(data.n_samples == 50);
			CHECK(data.Y(0, 0) == Approx(-1.18865038973338));
			CHECK(data.E(0, 0) == Approx(0.785198212));
		}
//
		data.standardise_non_genetic_data();
		SECTION("Check non genetic data standardised + covars regressed") {
			CHECK(data.params.scale_pheno);
			CHECK(data.params.use_vb_on_covars);
			CHECK(data.params.covar_file == "NULL");
//			CHECK(data.Y(0,0) == Approx(-3.6676363273605137)); Centered
			CHECK(data.Y(0,0) == Approx(-1.5800573524786081));
			CHECK(data.Y2(0, 0) == Approx(-1.5567970303));
			CHECK(data.E(0, 0) == Approx(0.8957059881));
		}

		data.read_full_bgen();
		SECTION("Ex1. bgen read in & standardised correctly") {
			CHECK(data.G.low_mem);
			CHECK(data.params.low_mem);
			CHECK(!data.params.flip_high_maf_variants);
			CHECK(data.G(0, 0) == Approx(-1.8575040711));
			CHECK(data.G(0, 1) == Approx(-0.7404793547));
			CHECK(data.G(0, 2) == Approx(-0.5845122102));
			CHECK(data.G(0, 3) == Approx(-0.6633007506));
			CHECK(data.n_var == 67);
		}

		SECTION("dXtEEX computed correctly") {
			data.calc_dxteex();
			CHECK(data.dXtEEX(0, 0) == Approx(38.9610805993));
			CHECK(data.dXtEEX(1, 0) == Approx(38.2995451744));
			CHECK(data.dXtEEX(2, 0) == Approx(33.7077899144));
			CHECK(data.dXtEEX(3, 0) == Approx(35.7391671158));

			CHECK(data.dXtEEX(0, 4) == Approx(-2.6239467101));
			CHECK(data.dXtEEX(1, 4) == Approx(-13.0001255314));
			CHECK(data.dXtEEX(2, 4) == Approx(-11.6635557299));
			CHECK(data.dXtEEX(3, 4) == Approx(-7.2154836264));
		}

		SECTION("Ex1. Confirm calc_dxteex() reorders properly") {
			data.params.dxteex_file = "data/io_test/n50_p100_dxteex_low_mem.txt";
			data.read_external_dxteex();
			data.calc_dxteex();
			CHECK(data.dXtEEX(0, 0) == Approx(38.9610805993));
			CHECK(data.dXtEEX(1, 0) == Approx(38.2995451744));
			CHECK(data.dXtEEX(2, 0) == Approx(33.7077899144));
			CHECK(data.dXtEEX(3, 0) == Approx(35.7391671158));

			CHECK(data.dXtEEX(0, 4) == Approx(-2.6239467101));
			CHECK(data.dXtEEX(1, 4) == Approx(-13.0001255314));
			CHECK(data.dXtEEX(2, 4) == Approx(-11.6635557299));
			CHECK(data.dXtEEX(3, 4) == Approx(-7.2154836264));
			CHECK(data.n_dxteex_computed == 1);
		}
	}

	SECTION("n50_p100_chr2.bgen") {
		p.bgen_file = "data/io_test/n50_p100_chr2.bgen";
		p.bgi_file = "data/io_test/n50_p100_chr2.bgen.bgi";
		Data data(p);

		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		SECTION("Ex1. bgen read in & standardised correctly") {
			CHECK(data.G.low_mem);
			CHECK(data.params.low_mem);
			CHECK(!data.params.flip_high_maf_variants);
			CHECK(data.G(0, 0) == Approx(0.7105269065));
			CHECK(data.G(0, 1) == Approx(0.6480740698));
			CHECK(data.G(0, 2) == Approx(0.7105195023));
			CHECK(data.G(0, 3) == Approx(-0.586791551));
			CHECK(data.G(0, 60) == Approx(-1.4317770638));
			CHECK(data.G(0, 61) == Approx(1.4862052498));
			CHECK(data.G(0, 62) == Approx(-0.3299831646));
			CHECK(data.G(0, 63) == Approx(-1.0968694989));
			CHECK(data.G.compressed_dosage_means(60) == Approx(1.00203125));
			CHECK(data.G.compressed_dosage_means(61) == Approx(0.9821875));
			CHECK(data.G.compressed_dosage_means(62) == Approx(0.10390625));
			CHECK(data.G.compressed_dosage_means(63) == Approx(0.68328125));
			CHECK(data.n_var == 75);
		}
	}

	SECTION("n50_p100_chr2.bgen w/ 2 chunks") {
		p.bgen_file = "data/io_test/n50_p100_chr2.bgen";
		p.bgi_file = "data/io_test/n50_p100_chr2.bgen.bgi";
		p.chunk_size = 72;
		p.n_bgen_thread = 2;
		Data data(p);

		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		SECTION("Ex1. bgen read in & standardised correctly") {
			CHECK(data.G.low_mem);
			CHECK(data.params.low_mem);
			CHECK(!data.params.flip_high_maf_variants);
			CHECK(data.G(0, 0) == Approx(0.7105269065));
			CHECK(data.G(0, 1) == Approx(0.6480740698));
			CHECK(data.G(0, 2) == Approx(0.7105195023));
			CHECK(data.G(0, 3) == Approx(-0.586791551));
			CHECK(data.G(0, 60) == Approx(-1.4317770638));
			CHECK(data.G(0, 61) == Approx(1.4862052498));
			CHECK(data.G(0, 62) == Approx(-0.3299831646));
			CHECK(data.G(0, 63) == Approx(-1.0968694989));
			CHECK(data.G.compressed_dosage_means(60) == Approx(1.00203125));
			CHECK(data.G.compressed_dosage_means(61) == Approx(0.9821875));
			CHECK(data.G.compressed_dosage_means(62) == Approx(0.10390625));
			CHECK(data.G.compressed_dosage_means(63) == Approx(0.68328125));
			CHECK(data.n_var == 75);
		}
	}

	SECTION("Check mult_vector_by_chr"){
		p.bgen_file = "data/io_test/n50_p100_chr2.bgen";
		p.bgi_file = "data/io_test/n50_p100_chr2.bgen.bgi";
		Data data(p);

		data.read_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd vv = Eigen::VectorXd::Ones(data.G.pp);
		Eigen::VectorXd v1 = data.G.mult_vector_by_chr(1, vv);
		Eigen::VectorXd v2 = data.G.mult_vector_by_chr(22, vv);

		CHECK(v1(0) == Approx(-0.8981400368));
		CHECK(v1(1) == Approx(-4.9936547948));
		CHECK(v1(2) == Approx(-1.7085924856));
		CHECK(v1(3) == Approx(0.8894016653));

		CHECK(v2(0) == Approx(-10.8022318897));
		CHECK(v2(1) == Approx(11.658910645));
		CHECK(v2(2) == Approx(-16.742754449));
		CHECK(v2(3) == Approx(0.9656298668));
	}
}

TEST_CASE( "Example 4: multi-env + mog + covars + emp_bayes" ){
	parameters p;

	SECTION("Ex4. No filters applied, high mem mode"){
		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb", (char*) "--low_mem",
			             (char*) "--use_vb_on_covars", (char*) "--mode_empirical_bayes",
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
			CHECK(data.params.scale_pheno);
			CHECK(data.params.use_vb_on_covars);
			CHECK(data.params.covar_file == "NULL");
//			CHECK(data.Y(0,0) == Approx(-3.6676363273605137)); Centered
			CHECK(data.Y(0,0) == Approx(-1.5800573524786081));
			CHECK(data.Y2(0,0) == Approx(-1.5567970303));
//			CHECK(data.C(0,0) == Approx(0.8957059881));
			CHECK(data.E(0,0) == Approx(0.8957059881));
			CHECK(data.E.row(0).array().sum() == Approx(2.9708148667));
		}
		data.read_full_bgen();

		data.calc_dxteex();
		data.calc_snpstats();
		data.set_vb_init();
		VBayesX2 VB(data);
		SECTION("Ex4. Vbayes_X2 initialised correctly"){
			CHECK(VB.n_samples == 50);
			CHECK(VB.N == 50.0);
			CHECK(VB.n_env == 4);
			CHECK(VB.n_var == 67);
			CHECK(VB.n_effects == 2);
			CHECK(VB.vp_init.mean_weights(0) == 0.25);
			CHECK(VB.p.init_weights_with_snpwise_scan == false);
			CHECK(VB.dXtEEX(0, 0) == Approx(38.9610805993));
			CHECK(VB.dXtEEX(1, 0) == Approx(38.2995451744));

			CHECK(VB.dXtEEX(0, 4) == Approx(-2.6239467101));
			CHECK(VB.dXtEEX(1, 4) == Approx(-13.0001255314));
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

			CHECK(vp.mean_weights(0)              == Approx(0.1053510228));

			CHECK(hyps.sigma                == Approx(0.6979599837));
			CHECK(hyps.lambda[0]            == Approx(0.1683579572));
			CHECK(hyps.lambda[1]            == Approx(0.1347097155));
			CHECK(hyps.slab_relative_var[0] == Approx(0.0080644075));
			CHECK(hyps.slab_relative_var[1] == Approx(0.0050698799));

			CHECK(hyps.lambda[0] == vp.betas->get_opt_hyps()(0));
			CHECK(hyps.slab_var[0] == vp.betas->get_opt_hyps()(1));
			CHECK(hyps.spike_var[0] == vp.betas->get_opt_hyps()(2));

			Eq_beta = vp.mean_beta();
			check_ym  = VB.X * Eq_beta;
			check_ym += VB.C * vp.mean_covar().cast<scalarData>().matrix();
			CHECK(vp.ym(0)            == Approx(check_ym(0)));

			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);

//			CHECK(vp.mean_weights(0)              == Approx(0.0594808543));
//			CHECK(vp.gammas.mix(63)           == Approx(0.1180509779));
//			CHECK(vp.mu1_gam(63)              == Approx(0.0019247043));
//			CHECK(vp.s1_gam_sq(63)            == Approx(0.002611992));

			Eq_beta = vp.mean_beta();
			check_ym  = VB.X * Eq_beta;
			check_ym += VB.C * vp.mean_covar().cast<scalarData>().matrix();
			CHECK(vp.ym(0)            == Approx(check_ym(0)));

			VB.updateAllParams(2, round_index, all_vp, all_hyps, logw_prev);

			CHECK(vp.mean_beta(63)   == vp.betas->mean(63));
			CHECK(vp.mean_weights(0) == vp.weights.mean(0));
			CHECK(vp.mean_weights(0)              == Approx(0.0358282042));
			CHECK(vp.mean_gam(63)              == Approx(0.0000060221));

			CHECK(VB.calc_logw(hyps, vp) == Approx(-88.4813237554));
			VbTracker tracker(p);
			tracker.init_interim_output(0,2, VB.n_effects, VB.n_env, VB.env_names, vp);
			tracker.dump_state(2, VB.n_samples, VB.n_covar, VB.n_var, VB.n_env,
			                   VB.n_effects, vp, hyps, VB.Y, VB.C, VB.X,
			                   VB.covar_names, VB.env_names);

			// Checking logw
			double int_linear = -1.0 * VB.calcExpLinear(hyps, vp) / 2.0 / hyps.sigma;
			int_linear -= VB.N * std::log(2.0 * VB.PI * hyps.sigma) / 2.0;
			// CHECK(int_linear  == Approx(-58.5936502834));

			// CHECK(VB.calcExpLinear(hyps, vp)  == Approx(30.4124788103));
			// CHECK(VB.calcKLBeta(hyps, vp)  == Approx(-5.4013615932));
			// CHECK(VB.calcKLGamma(hyps, vp)  == Approx(-0.0053957728));

			// check int_linear

			// variances
			CHECK(vp.EdZtZ.sum() == Approx(6260.1708501913));
			CHECK(vp.ym.squaredNorm() == Approx(14.6480266984));
			CHECK(vp.yx.squaredNorm() == Approx(0.0004676258));
		}

		VB.run_inference(VB.hyps_inits, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.8131749627));
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
//			CHECK(VB.n_samples == 50);
//			CHECK(VB.N == 50.0);
//			CHECK(VB.n_env == 4);
//			CHECK(VB.n_effects == 2);
//			CHECK(VB.vp_init.mean_weights(0) == 0.25);
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
//			CHECK(VB.n_samples == 50);
//			CHECK(VB.N == 50.0);
//			CHECK(VB.n_env == 4);
//			CHECK(VB.n_effects == 2);
//			CHECK(VB.vp_init.mean_weights(0) == 0.25);
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

//TEST_CASE( "Edge case 1: error in alpha" ){
//	parameters p;
//
//	SECTION("Ex1. No filters applied, low mem mode"){
//		char* argv[] = { (char*) "bin/bgen_prog", (char*) "--mode_vb", (char*) "--low_mem",
//			             (char*) "--mode_spike_slab",
//			             (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
//			             (char*) "--out", (char*) "data/io_test/fake_age.out",
//			             (char*) "--pheno", (char*) "data/io_test/pheno.txt",
//			             (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
//			             (char*) "--hyps_probs", (char*) "data/io_test/hyperpriors_gxage_probs.txt",
//			             (char*) "--environment", (char*) "data/io_test/age.txt"};
//		int argc = sizeof(argv)/sizeof(argv[0]);
//		parse_arguments(p, argc, argv);
//		Data data( p );
//		data.read_non_genetic_data();
//		data.standardise_non_genetic_data();
//		data.read_full_bgen();
//
//		data.calc_dxteex();
//		data.set_vb_init();
//		VBayesX2 VB(data);
//
//		std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
//		SECTION("Ex1. Explicitly checking updates"){
//
//			// Set up for RunInnerLoop
//			long n_grid = VB.hyps_inits.size();
//			long n_samples = VB.n_samples;
//			std::vector<Hyps> all_hyps = VB.hyps_inits;
//
//			// Set up for updateAllParams
//			std::vector<VariationalParameters> all_vp;
//			VB.setup_variational_params(all_hyps, all_vp);
//			VariationalParameters& vp = all_vp[0];
//			Hyps& hyps = all_hyps[0];
//
//			int round_index = 2;
//			std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
//			std::vector<std::vector< double > > logw_updates(n_grid);
//
//
//			CHECK_THROWS(VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev));
//		}
//	}
//}
