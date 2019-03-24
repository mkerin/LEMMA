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
			data.params.dxteex_file = "data/io_test/case8/dxteex_low_mem.txt";
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
			CHECK(data.n_dxteex_computed == 75);
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
			CHECK(data.params.flip_high_maf_variants);
			CHECK(data.G(0, 0) == Approx(-0.7105269065));
			CHECK(data.G(0, 1) == Approx(-0.6480740698));
			CHECK(data.G(0, 2) == Approx(-0.7105104917));
			CHECK(data.G(0, 3) == Approx(-0.586791551));
			CHECK(data.G(0, 60) == Approx(1.4862052498));
			CHECK(data.G(0, 61) == Approx(-0.3299831646));
			CHECK(data.G(0, 62) == Approx(-1.0968694989));
			CHECK(data.G(0, 63) == Approx(-0.5227553607));
			CHECK(data.G.compressed_dosage_means(60) == Approx(0.9821875));
			CHECK(data.G.compressed_dosage_means(61) == Approx(0.10390625));
			CHECK(data.G.compressed_dosage_means(62) == Approx(0.68328125));
			CHECK(data.G.compressed_dosage_means(63) == Approx(0.28359375));
			CHECK(data.n_var == 73);
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

		CHECK(v1(0) == Approx(-9.6711528276));
		CHECK(v1(1) == Approx(-0.4207388213));
		CHECK(v1(2) == Approx(-3.0495872499));
		CHECK(v1(3) == Approx(-9.1478619829));

		CHECK(v2(0) == Approx(-15.6533077013));
		CHECK(v2(1) == Approx(6.8078348334));
		CHECK(v2(2) == Approx(-4.4887853578));
		CHECK(v2(3) == Approx(8.9980192447));
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
			             (char*) "--hyps_probs", (char*) "data/io_test/single_hyps_gxage_probs.txt",
			             (char*) "--vb_init", (char*) "data/io_test/answer_init.txt"};
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
			CHECK(data.Y(0,0) == Approx(-1.5800573524786081));             // Scaled
			CHECK(data.Y2(0,0) == Approx(-1.5567970303));
//			CHECK(data.W(0,0) == Approx(0.8957059881));
			CHECK(data.E(0,0) == Approx(0.8957059881));
			CHECK(data.E.row(0).array().sum() == Approx(2.9708148667));
		}
		data.read_full_bgen();

		data.calc_dxteex();
		data.calc_snpstats();
		if(p.vb_init_file != "NULL") {
			data.read_alpha_mu();
		}
		VBayesX2 VB(data);
		VB.check_inputs();
		SECTION("Ex4. Vbayes_X2 initialised correctly"){
			CHECK(VB.n_samples == 50);
			CHECK(VB.N == 50.0);
			CHECK(VB.n_env == 4);
			CHECK(VB.n_var == 67);
			CHECK(VB.n_effects == 2);
			CHECK(VB.vp_init.muw(0) == 0.25);
			CHECK(VB.p.init_weights_with_snpwise_scan == false);
			CHECK(VB.dXtEEX(0, 0) == Approx(38.9610805993));
			CHECK(VB.dXtEEX(1, 0) == Approx(38.2995451744));

			CHECK(VB.dXtEEX(0, 4) == Approx(-2.6239467101));
			CHECK(VB.dXtEEX(1, 4) == Approx(-13.0001255314));
		}

		std::vector< VbTracker > trackers(VB.hyps_grid.rows(), p);
		SECTION("Ex4. Explicitly checking hyps") {
			// Set up for RunInnerLoop
			long n_grid = VB.hyps_grid.rows();
			std::vector<Hyps> all_hyps;
			VB.unpack_hyps(VB.hyps_grid, all_hyps);

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

			CHECK(vp.alpha_beta(0)            == Approx(0.1331830674));
			CHECK(vp.alpha_beta(1)            == Approx(0.1395213065));
			CHECK(vp.alpha_beta(63)           == Approx(0.1457756043));
			CHECK(vp.muw(0, 0)              == Approx(0.1151822334));

			CHECK(hyps.sigma                == Approx(0.7035012479));
			CHECK(hyps.lambda[0]            == Approx(0.1665905143));
			CHECK(hyps.lambda[1]            == Approx(0.1350873122));
			CHECK(hyps.slab_relative_var[0] == Approx(0.0078057485));
			CHECK(hyps.slab_relative_var[1] == Approx(0.0050625855));

			Eq_beta = vp.alpha_beta * vp.mu1_beta;
			if(p.mode_mog_prior_beta) Eq_beta.array() += (1 - vp.alpha_beta) * vp.mu2_beta;
			check_ym  = VB.X * Eq_beta;
			check_ym += VB.C * vp.muc.cast<scalarData>().matrix();
			CHECK(vp.ym(0)            == Approx(check_ym(0)));

			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);

			CHECK(vp.alpha_beta(0)            == Approx(0.1455431449));
			CHECK(vp.muw(0, 0)              == Approx(0.0675892528));
			CHECK(vp.alpha_gam(63)           == Approx(0.1181281224));
			CHECK(vp.mu1_gam(63)              == Approx(0.0019354328));
			CHECK(vp.s1_gam_sq(63)            == Approx(0.0026158316));

			CHECK(hyps.sigma                == Approx(0.6078333334));
			CHECK(hyps.lambda[0]            == Approx(0.1951515205));
			CHECK(hyps.lambda[1]            == Approx(0.1175686496));
			CHECK(hyps.slab_relative_var[0] == Approx(0.0120417653));
			CHECK(hyps.slab_relative_var[1] == Approx(0.0042687998));
			CHECK(hyps.s_x[0]               == Approx(67.0));
			CHECK(hyps.s_x[1]               == Approx(0.3092740356));
			CHECK(hyps.pve[1]               == Approx(0.0001342237));
			CHECK(hyps.pve_large[1]         == Approx(0.0001340815));

			Eq_beta = vp.alpha_beta * vp.mu1_beta;
			if(p.mode_mog_prior_beta) Eq_beta.array() += (1 - vp.alpha_beta) * vp.mu2_beta;
			check_ym  = VB.X * Eq_beta;
			check_ym += VB.C * vp.muc.cast<scalarData>().matrix();
			CHECK(vp.ym(0)            == Approx(check_ym(0)));

			VB.updateAllParams(2, round_index, all_vp, all_hyps, logw_prev);

			CHECK(vp.alpha_beta(63)           == Approx(0.1858817399));
			CHECK(vp.muw(0, 0)              == Approx(0.0419539867));
			CHECK(vp.alpha_gam(63)           == Approx(0.1036835901));
			CHECK(vp.mu1_gam(63)              == Approx(0.0000724454));
			CHECK(vp.s1_gam_sq(63)            == Approx(0.0019558949));

			CHECK(hyps.sigma                == Approx(0.5652108593));
			CHECK(hyps.lambda[0]            == Approx(0.213751744));
			CHECK(hyps.lambda[1]            == Approx(0.1032266612));
			CHECK(hyps.slab_relative_var[0] == Approx(0.0151665945));
			CHECK(hyps.slab_relative_var[1] == Approx(0.0034302028));
			CHECK(hyps.s_x[0]               == Approx(67.0));
			CHECK(hyps.s_x[1]               == Approx(0.1225701767));
			CHECK(hyps.pve[1]               == Approx(0.0000357123));
			CHECK(hyps.pve_large[1]         == Approx(0.0000356537));

			CHECK(VB.calc_logw(hyps, vp) == Approx(-88.4955694238));
			VbTracker tracker(p);
			tracker.init_interim_output(0,2, VB.n_effects, VB.n_env, VB.env_names, vp);
			tracker.dump_state(2, VB.n_covar, VB.n_var, VB.n_env, VB.n_effects, vp, hyps, VB.X, VB.covar_names, VB.env_names);

			// Checking logw
			double int_linear = -1.0 * VB.calcExpLinear(hyps, vp) / 2.0 / hyps.sigma;
			int_linear -= VB.N * std::log(2.0 * VB.PI * hyps.sigma) / 2.0;
			CHECK(int_linear  == Approx(-58.6830095598));

			CHECK(VB.calcExpLinear(hyps, vp)  == Approx(30.5213788039));
			CHECK(VB.calcKLBeta(hyps, vp)  == Approx(-5.3388177133));
			CHECK(VB.calcKLGamma(hyps, vp)  == Approx(-0.0059150945));

			// check int_linear

			// Expectation of linear regression log-likelihood
			int_linear  = (VB.Y - vp.ym).squaredNorm();
			int_linear -= 2.0 * (VB.Y - vp.ym).cwiseProduct(vp.eta).dot(vp.yx);
			int_linear += vp.yx.cwiseProduct(vp.eta_sq).dot(vp.yx);
			CHECK(int_linear == Approx(21.7708079744));

			double int_linear2  = (VB.Y - vp.ym - vp.yx.cwiseProduct(vp.eta)).squaredNorm();
			int_linear2 -= vp.yx.cwiseProduct(vp.eta).squaredNorm();
			int_linear2 += vp.yx.cwiseProduct(vp.eta_sq).dot(vp.yx);
			CHECK(int_linear2 == Approx(21.7708079744));

			double kl_covar = 0.0;
			kl_covar += (double) VB.n_covar * (1.0 - std::log(hyps.sigma * VB.sigma_c)) / 2.0;
			kl_covar += vp.sc_sq.log().sum() / 2.0;
			kl_covar -= vp.sc_sq.sum() / 2.0 / hyps.sigma / VB.sigma_c;
			kl_covar -= vp.muc.square().sum() / 2.0 / hyps.sigma / VB.sigma_c;
			CHECK(kl_covar == Approx(-24.0588694492));

			// weights
			double kl_weights = 0.0;
			kl_weights += (double) VB.n_env / 2.0;
			kl_weights += vp.sw_sq.log().sum() / 2.0;
			kl_weights -= vp.sw_sq.sum() / 2.0;
			kl_weights -= vp.muw.square().sum() / 2.0;
			CHECK(kl_weights == Approx(-0.4088998601));



			// variances
			CHECK(vp.sc_sq.sum() == Approx(0.0496189464));
			CHECK(vp.var_beta().sum() == Approx(0.1037083336));
			CHECK(vp.var_gam().sum() == Approx(0.01343011));
			CHECK(vp.mean_beta().sum() == Approx(0.1211143488));
			CHECK(vp.mean_gam().sum() == Approx(0.0024228301));
			CHECK(vp.muw.sum() == Approx(0.025847795));
			CHECK(vp.sw_sq.sum() == Approx(1.9640756682));
			CHECK((vp.EdZtZ * vp.var_gam()).sum() == Approx(1.2375451411));

			CHECK(vp.EdZtZ.sum() == Approx(6224.5757519367));
			CHECK(vp.eta_sq.sum() == Approx(96.3429032915));
			CHECK(vp.eta.squaredNorm() == Approx(0.1031955508));
			CHECK(vp.ym.squaredNorm() == Approx(14.3819142131));
			CHECK(vp.yx.squaredNorm() == Approx(0.0006432176));

			double dztz0 = (VB.X.col(0).array().square() * vp.eta_sq.array()).sum();
			CHECK(dztz0 == Approx(92.1333438413));
			CHECK(vp.EdZtZ(0) == Approx(92.1333438413));
		}

		VB.run_inference(VB.hyps_grid, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.8131749627));
		}
	}
}

TEST_CASE("--dxteex") {
	parameters p;
	char *argv[] = {(char *) "bin/bgen_prog", (char *) "--mode_vb", (char *) "--low_mem",
		            (char *) "--use_vb_on_covars", (char *) "--mode_empirical_bayes",
		            (char *) "--effects_prior_mog",
		            (char *) "--vb_iter_max", (char *) "10",
		            (char *) "--environment", (char *) "data/io_test/n50_p100_env.txt",
		            (char *) "--bgen", (char *) "data/io_test/n50_p100.bgen",
		            (char *) "--out", (char *) "data/io_test/config4.out",
		            (char *) "--pheno", (char *) "data/io_test/pheno.txt",
		            (char *) "--hyps_grid", (char *) "data/io_test/single_hyps_gxage.txt",
		            (char *) "--vb_init", (char *) "data/io_test/answer_init.txt"};
	int argc = sizeof(argv) / sizeof(argv[0]);
	parse_arguments(p, argc, argv);

	SECTION("Compute dxteex internally"){
		Data data(p);

		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		SECTION("Ex4. Non genetic data standardised + covars regressed") {
			CHECK(data.E(0, 0) == Approx(0.8957059881));
		}
		data.read_full_bgen();

		data.calc_dxteex();
		data.calc_snpstats();
		if (p.vb_init_file != "NULL") {
			data.read_alpha_mu();
		}
		VBayesX2 VB(data);
		VB.check_inputs();
		SECTION("Ex4. Vbayes_X2 initialised correctly") {
			CHECK(VB.n_samples == 50);
			CHECK(VB.N == 50.0);
			CHECK(VB.n_env == 4);
			CHECK(VB.n_effects == 2);
			CHECK(VB.vp_init.muw(0) == 0.25);
			CHECK(VB.p.init_weights_with_snpwise_scan == false);
			CHECK(VB.dXtEEX(0, 0) == Approx(38.9610805993));
			CHECK(VB.dXtEEX(1, 0) == Approx(38.2995451744));
			CHECK(VB.dXtEEX(2, 0) == Approx(33.7077899144));
			CHECK(VB.dXtEEX(3, 0) == Approx(35.7391671158));

			CHECK(VB.dXtEEX(0, 4) == Approx(-2.6239467101));
			CHECK(VB.dXtEEX(1, 4) == Approx(-13.0001255314));
			CHECK(VB.dXtEEX(2, 4) == Approx(-11.6635557299));
			CHECK(VB.dXtEEX(3, 4) == Approx(-7.2154836264));
		}

		std::vector< VbTracker > trackers(VB.hyps_grid.rows(), p);
		VB.run_inference(VB.hyps_grid, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.8131699164));
		}
	}

	SECTION("Compute dxteex external") {
		p.dxteex_file = "data/io_test/n50_p100_dxteex.txt";
		Data data(p);

		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		SECTION("Ex4. Non genetic data standardised + covars regressed") {
			CHECK(data.E(0, 0) == Approx(0.8957059881));
		}
		data.read_full_bgen();

		data.read_external_dxteex();
		data.calc_dxteex();
		data.calc_snpstats();
		if (p.vb_init_file != "NULL") {
			data.read_alpha_mu();
		}
		VBayesX2 VB(data);
		VB.check_inputs();
		SECTION("Ex4. Vbayes_X2 initialised correctly") {
			CHECK(VB.n_samples == 50);
			CHECK(VB.N == 50.0);
			CHECK(VB.n_env == 4);
			CHECK(VB.n_effects == 2);
			CHECK(VB.vp_init.muw(0) == 0.25);
			CHECK(VB.p.init_weights_with_snpwise_scan == false);
			CHECK(VB.dXtEEX(0, 0) == Approx(38.9610805993));
			CHECK(VB.dXtEEX(1, 0) == Approx(38.3718));
			CHECK(VB.dXtEEX(2, 0) == Approx(33.81659));
			CHECK(VB.dXtEEX(3, 0) == Approx(35.8492));

			CHECK(VB.dXtEEX(0, 4) == Approx(-2.6239467101));
			CHECK(VB.dXtEEX(1, 4) == Approx(-12.96763));
			CHECK(VB.dXtEEX(2, 4) == Approx(-11.66501));
			CHECK(VB.dXtEEX(3, 4) == Approx(-7.20105));
		}

		std::vector< VbTracker > trackers(VB.hyps_grid.rows(), p);
		VB.run_inference(VB.hyps_grid, false, 2, trackers);
		SECTION("Ex3. Vbayes_X2 inference correct"){
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-86.8131699164));
		}
	}
}


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
//		VB.check_inputs();
//		std::vector< VbTracker > trackers(VB.hyps_grid.rows(), p);
//		VB.run_inference(VB.hyps_grid, false, 2, trackers);
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
//		VB.check_inputs();
//		std::vector< VbTracker > trackers(VB.hyps_grid.rows(), p);
//		SECTION("Dump params state") {
//			// Set up for RunInnerLoop
//			long n_grid = VB.hyps_grid.rows();
//			std::vector<Hyps> all_hyps;
//			VB.unpack_hyps(VB.hyps_grid, all_hyps);
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
//			tracker.dump_state(2, VB.n_covar, VB.n_var, VB.n_env, VB.n_effects, vp, hyps, VB.X, VB.covar_names,
//			                   VB.env_names);
//		}
//
//		VB.run_inference(VB.hyps_grid, false, 2, trackers);
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
			             (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
			             (char*) "--environment", (char*) "data/io_test/age.txt"};
		int argc = sizeof(argv)/sizeof(argv[0]);
		parse_arguments(p, argc, argv);
		Data data( p );
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		data.calc_dxteex();
		if(p.vb_init_file != "NULL") {
			data.read_alpha_mu();
		}
		VBayesX2 VB(data);
		VB.check_inputs();

		std::vector< VbTracker > trackers(VB.hyps_grid.rows(), p);
		SECTION("Ex1. Explicitly checking updates"){
			// Initialisation
#ifdef DATA_AS_FLOAT
			CHECK( (double)  VB.vp_init.ym(0) == Approx(0.0003200434));
#else
			CHECK(VB.vp_init.ym(0) == Approx(-0.0506835772));
#endif
			CHECK(VB.vp_init.yx(0) == Approx(0.0732037568));
			CHECK(VB.vp_init.eta(0) == Approx(-0.5894793969));

			// Set up for RunInnerLoop
			long n_grid = VB.hyps_grid.rows();
			long n_samples = VB.n_samples;
			std::vector<Hyps> all_hyps;
			VB.unpack_hyps(VB.hyps_grid, all_hyps);

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
