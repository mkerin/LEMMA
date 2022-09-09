// tests-main.cpp
#include "catch.hpp"

#include "../src/tools/eigen3.3/Dense"
#include "../src/parameters.hpp"
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
						(char*) "--rhe-random-vectors", (char*) "unit/data/test_pve/gaussian_noise_50_x_5.txt",
	                    (char*) "--streamBgen", (char*) "unit/data/n50_p100.bgen",
	                    (char*) "--pheno", (char*) "unit/data/pheno.txt",
	                    (char*) "--environment", (char*) "unit/data/n50_p100_env.txt",
	                    (char*) "--out", (char*) "unit/data/test_pve/RHEreg_NM.out.gz"};

char* argv_rhe_lm2[] = { (char*) "LEMMA",
	                     (char*) "--gplemma",
	                     (char*) "--gplemma-max-iter", (char*) "5",
	                     (char*) "--gplemma-random-starts", (char*) "1",
	                     (char*) "--maf", (char*) "0.01",
	                     (char*) "--random-seed", (char*) "1",
	                     (char*) "--n-RHEreg-jacknife", (char*) "1",
	                     (char*) "--n-RHEreg-samples", (char*) "5",
						 (char*) "--rhe-random-vectors", (char*) "unit/data/test_pve/gaussian_noise_50_x_5.txt",
	                     (char*) "--streamBgen", (char*) "unit/data/n50_p100.bgen",
	                     (char*) "--pheno", (char*) "unit/data/pheno.txt",
	                     (char*) "--environment", (char*) "unit/data/n50_p100_env.txt",
	                     (char*) "--out", (char*) "unit/data/test_pve/RHEreg_LM.out.gz"};

char* argv_multiE[] = { (char*) "--RHEreg", (char*) "--maf", (char*) "0.01",
	                    (char*) "--random-seed", (char*) "1",
	                    (char*) "--n-RHEreg-jacknife", (char*) "1",
	                    (char*) "--n-RHEreg-samples", (char*) "2",
						(char*) "--rhe-random-vectors", (char*) "unit/data/test_pve/gaussian_noise_50_x_2.txt",
	                    (char*) "--streamBgen", (char*) "unit/data/n50_p100.bgen",
	                    (char*) "--pheno", (char*) "unit/data/pheno.txt",
	                    (char*) "--environment", (char*) "unit/data/n50_p100_env.txt",
	                    (char*) "--out", (char*) "unit/data/test_pve/pve_est.out.gz"};


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

		CHECK(pve.nls_env_weights(0) == Approx(0.119065368));
		CHECK(pve.nls_env_weights(1) == Approx(-0.3631866642));
		CHECK(pve.nls_env_weights(2) == Approx(0.0395414866));

		CHECK(pve.sigmas(0) == Approx( 0.0043657095));
		CHECK(pve.sigmas(1) == Approx(-0.1222527085));
		CHECK(pve.sigmas(2) == Approx(1.1372808824));
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
		CHECK(LM.theta(0) == Approx(-0.0245725812));
		CHECK(LM.theta(2) == Approx(-0.1526502676));
		CHECK(LM.u == Approx(2.0490523438));

		CHECK(LM.JtJ(0,0) == Approx(204.9052343808));
		CHECK(LM.JtJ(1,1) == Approx(8.8937951504));
		CHECK(LM.JtJ(1,2) == Approx(-0.0739273743));

		LM.iterLM();
		CHECK(LM.count == 1);
		CHECK(LM.delta(0) == Approx(-0.0005989636));
		CHECK(LM.delta(1) == Approx(0.1987456184));
		CHECK(LM.rho == Approx(0.119016));
		CHECK(LM.u == Approx(2.9555570866));
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

		CHECK(pve.nls_env_weights(0) == Approx(0.2387207031));
		CHECK(pve.nls_env_weights(1) == Approx(0.192578125));
		CHECK(pve.nls_env_weights(2) == Approx(0.2768066406));
		CHECK(pve.nls_env_weights(3) == Approx(0.2814941406));

		CHECK(pve.sigmas(0) == Approx(-0.0089448344));
		CHECK(pve.sigmas(1) == Approx(-0.5956388748));
		CHECK(pve.sigmas(2) == Approx(1.1600104868));
	}

	SECTION("RHEreg with env_weights from NLS") {
		// gunzip -c data/io_test/test_RHEreg_NM_NM_env_weights.out.gz > data/io_test/n50_p100_nm_env_weights.txt
		parameters p;
		int argc = sizeof(argv_rhe_nm) / sizeof(argv_rhe_nm[0]);
		parse_arguments(p, argc, argv_rhe_nm);
		p.mode_RHE = true;
		p.mode_RHEreg_NM = false;
		p.env_coeffs_file = "unit/data/n50_p100_nm_env_weights.txt";
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

		CHECK(pve.sigmas(0) == Approx(-0.0034659874));
		CHECK(pve.sigmas(1) == Approx(-0.3798285751));
		CHECK(pve.sigmas(2) == Approx(1.1029031174));
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
		CHECK(CC(0, 0) == Approx(318.6309897574));
		CHECK(CC(1, 0) == Approx(41.4835207503));
		CHECK(CC(1, 1) == Approx(113.3920545695));
		CHECK(CC(1, bb_ind) == Approx(17.7540237989));
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
		CHECK(pve.n_components == 6);
		CHECK(CC(0, 0) == Approx(318.6309897574));

		CHECK(CC(1, 0) == Approx(41.4835207503));
		CHECK(CC(1, 1) == Approx(113.3920545695));
		CHECK(CC(1, bb_ind) == Approx(17.7540237989));

		CHECK(CC(5, 0) == Approx(71.5983559289));
		CHECK(CC(5, 5) == Approx(45.0));
		CHECK(CC(5, bb_ind) == Approx(46.5105903879));
	}
}
