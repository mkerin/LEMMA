// tests-main.cpp
#define EIGEN_USE_MKL_ALL
#include "catch.hpp"

#include <iostream>
#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/rhe_reg.hpp"
#include "../src/data.hpp"


// Scenarios
char* argv_pve1[] = { (char*) "--RHE",
	                  (char*) "--random_seed", (char*) "1",
	                  (char*) "--n_pve_samples", (char*) "3",
	                  (char*) "--bgen", (char*) "data/io_test/n1000_p2000.bgen",
	                  (char*) "--pheno", (char*) "data/io_test/case8/pheno.txt",
	                  (char*) "--environment", (char*) "data/io_test/case8/age.txt",
	                  (char*) "--out", (char*) "data/io_test/case8/test_pve_est.out.gz"};

char* argv_pve2[] = { (char*) "--RHE",
	                  (char*) "--random_seed", (char*) "1",
	                  (char*) "--n_pve_samples", (char*) "3",
	                  (char*) "--streamBgen-print-interval", (char*) "1",
	                  (char*) "--streamBgen", (char*) "data/io_test/n1000_p2000.bgen",
	                  (char*) "--pheno", (char*) "data/io_test/case8/pheno.txt",
	                  (char*) "--environment", (char*) "data/io_test/case8/age.txt",
	                  (char*) "--out", (char*) "data/io_test/case8/test_pve_est.out.gz"};

char* argv_pve2b[] = { (char*) "--RHE",
	                   (char*) "--random_seed", (char*) "1",
	                   (char*) "--n_pve_samples", (char*) "3",
	                   (char*) "--streamBgen-print-interval", (char*) "1",
	                   (char*) "--mStreamBgen", (char*) "data/io_test/n1000_p2000_bgens.txt",
	                   (char*) "--pheno", (char*) "data/io_test/case8/pheno.txt",
	                   (char*) "--environment", (char*) "data/io_test/case8/age.txt",
	                   (char*) "--out", (char*) "data/io_test/case8/test_pve_est.out.gz"};

char* argv_pve3[] = { (char*) "--RHE", (char*) "--maf", (char*) "0.01",
	                  (char*) "--random_seed", (char*) "1",
	                  (char*) "--n_jacknife", (char*) "2",
	                  (char*) "--n_pve_samples", (char*) "10",
	                  (char*) "--streamBgen", (char*) "data/io_test/n1000_p2000.bgen",
	                  (char*) "--pheno", (char*) "data/io_test/case8/pheno.txt",
	                  (char*) "--environment", (char*) "data/io_test/case8/age.txt",
	                  (char*) "--out", (char*) "data/io_test/case8/test_pve_est.out.gz"};
//
char* argv_main1[] = { (char*) "--RHE",
	                   (char*) "--random_seed", (char*) "1",
	                   (char*) "--n_pve_samples", (char*) "3",
	                   (char*) "--streamBgen", (char*) "data/io_test/n1000_p2000.bgen",
	                   (char*) "--streamBgen-print-interval", (char*) "1",
	                   (char*) "--pheno", (char*) "data/io_test/case8/pheno.txt",
	                   (char*) "--out", (char*) "data/io_test/case8/test_pve_est.out.gz"};

char* argv_main2[] = { (char*) "--RHE",
	                   (char*) "--random_seed", (char*) "1",
	                   (char*) "--n_pve_samples", (char*) "3",
	                   (char*) "--bgen", (char*) "data/io_test/n1000_p2000.bgen",
	                   (char*) "--pheno", (char*) "data/io_test/case8/pheno.txt",
	                   (char*) "--out", (char*) "data/io_test/case8/test_pve_est.out.gz"};


TEST_CASE("RHEreg-GxE") {
	SECTION("GxE effects fit") {
		parameters p;
		int argc = sizeof(argv_pve1) / sizeof(argv_pve1[0]);
		parse_arguments(p, argc, argv_pve1);
		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();
		Eigen::VectorXd eta = data.E.col(0);
		RHEreg pve(data, Y, C, eta);
		pve.run();

		CHECK(pve.sigmas(0) == Approx(0.4768423518));
		CHECK(pve.sigmas(1) == Approx(0.0629302327));
		CHECK(pve.sigmas(2) == Approx(0.2739590655));
		CHECK(pve.h2(0) == Approx(0.5762155215));
		CHECK(pve.h2(1) == Approx(0.0761118418));
		CHECK(pve.h2(2) == Approx(0.3476726367));

		CHECK(pve.h2_se_jack(0) == Approx(0.2596384624));
		CHECK(pve.h2_se_jack(1) == Approx(0.0391032891));
		pve.to_file(p.out_file);
	}

	SECTION("GxE effects fit (stream version)") {
		parameters p;
		int argc = sizeof(argv_pve2) / sizeof(argv_pve2[0]);
		parse_arguments(p, argc, argv_pve2);
		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();
		Eigen::VectorXd eta = data.E.col(0);
		RHEreg pve(data, Y, C, eta);
		pve.run();

		CHECK(pve.sigmas(0) == Approx(0.4764408911));
		CHECK(pve.sigmas(1) == Approx(0.062986786));
		CHECK(pve.sigmas(2) == Approx(0.2742296405));
		CHECK(pve.h2(0) == Approx(0.575807122));
		CHECK(pve.h2(1) == Approx(0.0761768632));

		CHECK(pve.components[0].n_vars_local[0] == 768);
		CHECK(pve.components[0].n_vars_local[1] == 1042);

		CHECK(pve.h2_se_jack(0) == Approx(0.309833704));
		CHECK(pve.h2_se_jack(1) == Approx(0.007952951));
	}

	SECTION("GxE effects fit (streaming from 2 files)") {
		parameters p;
		int argc = sizeof(argv_pve2b) / sizeof(argv_pve2b[0]);
		parse_arguments(p, argc, argv_pve2b);
		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();
		Eigen::VectorXd eta = data.E.col(0);
		RHEreg pve(data, Y, C, eta);
		pve.run();

		CHECK(pve.sigmas(0) == Approx(0.4764408911));
		CHECK(pve.sigmas(1) == Approx(0.062986786));
		CHECK(pve.sigmas(2) == Approx(0.2742296405));
		CHECK(pve.h2(0) == Approx(0.575807122));
		CHECK(pve.h2(1) == Approx(0.0761768632));

		// Jacknife blocks cut off at gap between bgen files
		CHECK(pve.components[0].n_vars_local[0] == 849);
		CHECK(pve.components[0].n_vars_local[1] == 961);

		CHECK(pve.h2_se_jack(0) == Approx(0.3566948701));
		CHECK(pve.h2_se_jack(1) == Approx(0.0154807884));
	}
}

TEST_CASE("RHEreg-multicomp"){
	SECTION("Main (component 1)") {
		parameters p;
		int argc = sizeof(argv_main1)/sizeof(argv_main1[0]);
		parse_arguments(p, argc, argv_main1);
		p.incl_rsids_file = "data/io_test/n1000_p2000_rsids_group1.txt";
		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();

		RHEreg pve(data, Y, C);
		pve.run();

		CHECK(pve.components[0].n_var_local == 898);
		CHECK(pve.components[0] * pve.components[0] == Approx(3239.2568992814));
	}
	SECTION("Main (component 2)") {
		parameters p;
		int argc = sizeof(argv_main1)/sizeof(argv_main1[0]);
		parse_arguments(p, argc, argv_main1);
		p.incl_rsids_file = "data/io_test/n1000_p2000_rsids_group2.txt";
		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();

		RHEreg pve(data, Y, C);
		pve.run();

		CHECK(pve.components[0].n_var_local == 912);
		CHECK(pve.components[0] * pve.components[0] == Approx(3155.790469249));
	}
	SECTION("Main (multi-comp)") {
		parameters p;
		int argc = sizeof(argv_main1)/sizeof(argv_main1[0]);
		parse_arguments(p, argc, argv_main1);
		p.RHE_groups_files.push_back("data/io_test/n1000_p2000_components.txt");
		p.RHE_multicomponent = true;
		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();

		RHEreg pve(data, Y, C);
		pve.run();

		CHECK(pve.components[0].n_var_local == 898);
		CHECK(pve.components[0] * pve.components[0] == Approx(3239.2568992814));
		CHECK(pve.components[1].n_var_local == 912);
		CHECK(pve.components[1] * pve.components[1] == Approx(3155.790469249));
	}
}

TEST_CASE("RHEreg-G") {
	SECTION("Main effects fit (gaussian prior)"){
		parameters p;
		int argc = sizeof(argv_main2)/sizeof(argv_main2[0]);
		parse_arguments(p, argc, argv_main2);
		Data data( p );
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();

		RHEreg pve(data, Y, C);
		pve.run();

		CHECK(pve.sigmas(0)  == Approx(0.5237752503));
		CHECK(pve.sigmas(1)  == Approx(0.49942111));
		CHECK(pve.h2(0)  == Approx(0.50057889));
		CHECK(pve.h2(1)  == Approx(0.49942111));
		std::cout << p.out_file << std::endl;
		pve.to_file(p.out_file);

		CHECK(pve.h2_se_jack(0)  == Approx(0.1855704313));
		CHECK(pve.h2_se_jack(1)  == Approx(0.8261286578));
	}

	SECTION("GxE Jacknife"){
		parameters p;
		int argc = sizeof(argv_pve3)/sizeof(argv_pve3[0]);
		parse_arguments(p, argc, argv_pve3);

		SECTION("Jacknife block 1"){
			Data data(p);
			data.read_non_genetic_data();
			data.standardise_non_genetic_data();
			data.read_full_bgen();
			data.set_vb_init();

			Eigen::VectorXd Y = data.Y.cast<double>();
			RHEreg pve(data, Y, data.C, data.vp_init.eta);
			pve.run();

			// RHE with 1st jacknife block removed
			for (long ii = 0; ii < pve.n_components; ii++) {
				pve.components[ii].rm_jacknife_block = 0;
			}
			CHECK(pve.components[0].get_n_var_local() == Approx(924));
			// CHECK(pve.components[0].getXXtz().squaredNorm() == Approx(39653744616.7472076416));
			// CHECK(pve.components[1].getXXtz().squaredNorm() == Approx(60669447736.4095077515));

			Eigen::MatrixXd CC = pve.construct_vc_system(pve.components);
			Eigen::MatrixXd AA = CC.block(0, 0, pve.n_components, pve.n_components);
			Eigen::VectorXd bb = CC.col(pve.n_components);
			Eigen::VectorXd ss = AA.colPivHouseholderQr().solve(bb);

			CHECK(CC(0, 0) == Approx(4609.7761957246));
			CHECK(CC(1, 1) == Approx(7042.7712505186));
			CHECK(CC(1, 0) == Approx(1128.1258528109));
			CHECK(CC(1, 2) == Approx(1035.3549637537));
			CHECK(pve.components[1] * pve.components[2] == Approx(1035.3549637537));
			CHECK(pve.components[1].get_n_var_local() == 924);
			CHECK(pve.components[2].get_n_var_local() == 1);
			CHECK(CC(0, 2) == Approx(1023.9319930016));
			CHECK(CC(2, 2) == Approx(998.0));

			// CHECK(pve.components[0].zz.sum() == Approx(123.7670672474));
			CHECK(!pve.components[2].is_active);
			CHECK(pve.components[2].get_bb_trace() == Approx(786.4039860883));
			CHECK(bb(0) == Approx(2009.8649365637));
			CHECK(bb(1) == Approx(998.310581355));
			CHECK(bb(2) == Approx(786.4039860883));
		}
		SECTION("Full method excluding 1st jacknife"){
			p.incl_rsids_file = "data/io_test/case8/jack_rsids_1.txt";
			Data data( p );
			data.read_non_genetic_data();
			data.standardise_non_genetic_data();
			data.read_full_bgen();
			data.set_vb_init();

			Eigen::VectorXd Y = data.Y.cast<double>();
			RHEreg pve(data, Y, data.C, data.vp_init.eta);
			pve.run();

			// RHE on full data with 1st block removed
			for (long ii = 0; ii < pve.n_components; ii++) {
				pve.components[ii].rm_jacknife_block = -1;
			}
			CHECK(pve.components[0].get_n_var_local() == Approx(924));

			Eigen::MatrixXd CC = pve.construct_vc_system(pve.components);
			Eigen::MatrixXd AA = CC.block(0, 0, pve.n_components, pve.n_components);
			Eigen::VectorXd bb = CC.col(pve.n_components);
			Eigen::VectorXd ss = AA.colPivHouseholderQr().solve(bb);

			CHECK(CC(0, 0) == Approx(4609.7761957246));
			CHECK(CC(1, 1) == Approx(7042.7712505186));
			CHECK(CC(1, 0) == Approx(1128.1258528109));
			CHECK(CC(1, 2) == Approx(1035.3549637537));
		}
	}
}
