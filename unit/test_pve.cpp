// tests-main.cpp
#include "catch.hpp"

#include <iostream>
#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/rhe_reg.hpp"
#include "../src/data.hpp"
#include "../src/levenberg_marquardt.hpp"


// Scenarios
char* argv_rhe_nm[] = { (char*) "LEMMA",
						 (char*) "--RHEreg-NM",
						 (char*) "--NM-max-iter", (char*) "5",
						 (char*) "--maf", (char*) "0.01",
					  (char*) "--random-seed", (char*) "1",
					  (char*) "--n-RHEreg-jacknife", (char*) "1",
					  (char*) "--n-RHEreg-samples", (char*) "5",
					  (char*) "--streamBgen", (char*) "data/io_test/n50_p100.bgen",
					  (char*) "--pheno", (char*) "data/io_test/pheno.txt",
					  (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
					  (char*) "--out", (char*) "data/io_test/test_RHEreg_NM.out.gz"};

char* argv_rhe_lm2[] = { (char*) "LEMMA",
						(char*) "--RHEreg-LM",
						(char*) "--LM-max-iter", (char*) "5",
						(char*) "--LM-random-starts", (char*) "1",
						(char*) "--maf", (char*) "0.01",
						(char*) "--random-seed", (char*) "1",
						(char*) "--n-RHEreg-jacknife", (char*) "1",
						(char*) "--n-RHEreg-samples", (char*) "5",
						(char*) "--streamBgen", (char*) "data/io_test/n1000_p2000.bgen",
						(char*) "--pheno", (char*) "data/io_test/case8/pheno.txt",
						(char*) "--environment", (char*) "data/io_test/case8/env.txt",
						(char*) "--out", (char*) "data/io_test/test_RHEreg_LM.out.gz"};

char* argv_multiE[] = { (char*) "--RHEreg", (char*) "--maf", (char*) "0.01",
						(char*) "--random-seed", (char*) "1",
						(char*) "--n-RHEreg-jacknife", (char*) "1",
						(char*) "--n-RHEreg-samples", (char*) "2",
						(char*) "--streamBgen", (char*) "data/io_test/n1000_p2000.bgen",
						(char*) "--pheno", (char*) "data/io_test/case8/pheno.txt",
						(char*) "--environment", (char*) "data/io_test/case8/env.txt",
						(char*) "--out", (char*) "data/io_test/case8/test_pve_est.out.gz"};


TEST_CASE("RHE-LevenburgMarquardt") {
	SECTION("LevenburgMarquardt fit") {
		parameters p;
		int argc = sizeof(argv_rhe_lm2) / sizeof(argv_rhe_lm2[0]);
		parse_arguments(p, argc, argv_rhe_lm2);
		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();
		RHEreg pve(data, Y, C, data.E);

		pve.run();

		CHECK(pve.nls_env_weights(0) == Approx(0.8369129088));
		CHECK(pve.nls_env_weights(1) == Approx(-0.3255176523));
		CHECK(pve.nls_env_weights(2) == Approx(0.2055679022));

		CHECK(pve.sigmas(0) == Approx(0.4055395441));
		CHECK(pve.sigmas(1) == Approx(0.048474652));
		CHECK(pve.sigmas(2) == Approx(0.3365677122));
	}

	SECTION("Detailed check"){
		parameters p;
		int argc = sizeof(argv_rhe_lm2) / sizeof(argv_rhe_lm2[0]);
		parse_arguments(p, argc, argv_rhe_lm2);
		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();
		RHEreg pve(data, Y, C, data.E);
		pve.standardise_non_genetic_data();
		pve.initialise_components();
		pve.compute_RHE_trace_operators();
		LevenbergMarquardt LM(p, pve.components, pve.Y, pve.E, pve.C, pve.CtC_inv, pve.ytEXXtEys, pve.env_names);
		LM.setupLM();
		CHECK(LM.count == 0);
		CHECK(LM.theta(0) == Approx(0.4070208966));
		CHECK(LM.theta(1) == Approx(0.0072421961));
		CHECK(LM.theta(2) == Approx(0.0072421961));
		CHECK(LM.u == Approx(30.588369445));

		CHECK(LM.JtJ(0,0) == Approx(3058.8369445019));
		CHECK(LM.JtJ(0,1) == Approx(7.1707442261));
		CHECK(LM.JtJ(1,1) == Approx(0.9475845954));
		CHECK(LM.JtJ(1,2) == Approx(0.0530732073));
		CHECK(LM.JtJ(6,6) == Approx(996.0307218357));
		CHECK(LM.JtJ(5,6) == Approx(8.7081577861));

		LM.iterLM();
		CHECK(LM.count == 1);
		CHECK(LM.delta(0) == Approx(0.0000362475));
		CHECK(LM.delta(1) == Approx(0.0680535705));
		CHECK(LM.rho == Approx(9.1680027555));
		CHECK(LM.u == Approx(10.1961231483));
		CHECK(LM.v == Approx(2.0));
	}
}



TEST_CASE("RHE-NelderMead") {
	SECTION("NelderMead fit") {
		parameters p;
		int argc = sizeof(argv_rhe_nm) / sizeof(argv_rhe_nm[0]);
		parse_arguments(p, argc, argv_rhe_nm);
		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();
		RHEreg pve(data, Y, C, data.E);
		pve.run();

		CHECK(pve.nls_env_weights(0) == Approx(0.2911621094));
		CHECK(pve.nls_env_weights(1) == Approx(0.2802246094));
		CHECK(pve.nls_env_weights(2) == Approx(0.2530761719));
		CHECK(pve.nls_env_weights(3) == Approx(0.176171875));

		CHECK(pve.sigmas(0) == Approx(0.9156208775));
		CHECK(pve.sigmas(1) == Approx(0.4171651285));
		CHECK(pve.sigmas(2) == Approx(0.0932642134));
	}

	SECTION("RHEreg with env_weights from NLS") {
		// gunzip -c data/io_test/test_RHEreg_NM_NM_env_weights.out.gz > data/io_test/n50_p100_nm_env_weights.txt
		parameters p;
		int argc = sizeof(argv_rhe_nm) / sizeof(argv_rhe_nm[0]);
		parse_arguments(p, argc, argv_rhe_nm);
		p.mode_RHE = true;
		p.mode_RHEreg_NM = false;
		p.env_coeffs_file = "data/io_test/n50_p100_nm_env_weights.txt";
		p.use_raw_env = true;

		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.set_vb_init();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();
		RHEreg pve(data, Y, C, data.vp_init.eta);
		pve.run();

		CHECK(pve.sigmas(0) == Approx(0.9156208775));
		CHECK(pve.sigmas(1) == Approx(0.4171651285));
		CHECK(pve.sigmas(2) == Approx(0.0932642134));
	}
}

TEST_CASE("RHE-multiE") {
	SECTION("GxE with 1st env") {
		parameters p;
		int argc = sizeof(argv_multiE) / sizeof(argv_multiE[0]);
		parse_arguments(p, argc, argv_multiE);
		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();
		RHEreg pve(data, Y, C, data.E.col(0));
		pve.run();

		for (long ii = 0; ii < pve.n_components; ii++) {
			pve.components[ii].rm_jacknife_block = -1;
		}
		Eigen::MatrixXd CC = pve.construct_vc_system(pve.components);

		long bb_ind = pve.n_components;
		CHECK(pve.n_components == 3);
		CHECK(CC(0, 0) == Approx(2642.4710418312));
		CHECK(CC(1, 0) == Approx(1024.5639876077));
		CHECK(CC(1, 1) == Approx(4975.4755110506));
		CHECK(CC(1, bb_ind) == Approx(1051.0989764703));
	}

	SECTION("GxE with 5th env") {
		parameters p;
		int argc = sizeof(argv_multiE) / sizeof(argv_multiE[0]);
		parse_arguments(p, argc, argv_multiE);
		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();
		RHEreg pve(data, Y, C, data.E.col(4));
		pve.run();

		for (long ii = 0; ii < pve.n_components; ii++) {
			pve.components[ii].rm_jacknife_block = -1;
		}
		Eigen::MatrixXd CC = pve.construct_vc_system(pve.components);

		long bb_ind = pve.n_components;
		CHECK(pve.n_components == 3);
		CHECK(CC(0, 0) == Approx(2642.4710418312));
		CHECK(CC(1, 0) == Approx(991.4735635656));
		CHECK(CC(1, 1) == Approx(5071.0797941864));
		CHECK(CC(1, bb_ind) == Approx(677.3482486883));
	}

	SECTION("GxE all envs") {
		parameters p;
		int argc = sizeof(argv_multiE) / sizeof(argv_multiE[0]);
		parse_arguments(p, argc, argv_multiE);
		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();
		RHEreg pve(data, Y, C, data.E);
		pve.run();

		for (long ii = 0; ii < pve.n_components; ii++) {
			pve.components[ii].rm_jacknife_block = -1;
		}
		Eigen::MatrixXd CC = pve.construct_vc_system(pve.components);

		long bb_ind = pve.n_components;
		CHECK(pve.n_components == 7);
		CHECK(CC(0, 0) == Approx(2642.4710418312));

		CHECK(CC(1, 0) == Approx(1024.5639876077));
		CHECK(CC(1, 1) == Approx(4975.4755110506));
		CHECK(CC(1, bb_ind) == Approx(1051.0989764703));

		CHECK(CC(5, 0) == Approx(991.4735635656));
		CHECK(CC(5, 5) == Approx(5071.0797941864));
		CHECK(CC(5, bb_ind) == Approx(677.3482486883));
	}
}
