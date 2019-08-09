// tests-main.cpp
#define EIGEN_USE_MKL_ALL
#include "catch.hpp"

#include <iostream>
#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/pve.hpp"
#include "../src/data.hpp"


// Scenarios
char* argv_pve1[] = { (char*) "--mode_pve_est",
	                  (char*) "--random_seed", (char*) "1",
	                  (char*) "--n_pve_samples", (char*) "3",
	                  (char*) "--bgen", (char*) "data/io_test/n1000_p2000.bgen",
	                  (char*) "--pheno", (char*) "data/io_test/case8/pheno.txt",
	                  (char*) "--environment", (char*) "data/io_test/case8/age.txt",
	                  (char*) "--out", (char*) "data/io_test/case8/test_pve_est.out.gz"};

char* argv_pve2[] = { (char*) "--mode_pve_est",
	                  (char*) "--random_seed", (char*) "1",
	                  (char*) "--n_pve_samples", (char*) "3",
	                  (char*) "--streamBgen", (char*) "data/io_test/n1000_p2000.bgen",
	                  (char*) "--pheno", (char*) "data/io_test/case8/pheno.txt",
	                  (char*) "--environment", (char*) "data/io_test/case8/age.txt",
	                  (char*) "--out", (char*) "data/io_test/case8/test_pve_est.out.gz"};

char* argv_pve3[] = { (char*) "--mode_pve_est", (char*) "--maf", (char*) "0.01",
					  (char*) "--random_seed", (char*) "1",
					  (char*) "--n_jacknife", (char*) "2",
					  (char*) "--n_pve_samples", (char*) "10",
					  (char*) "--streamBgen", (char*) "data/io_test/n1000_p2000.bgen",
					  (char*) "--pheno", (char*) "data/io_test/case8/pheno.txt",
					  (char*) "--environment", (char*) "data/io_test/case8/age.txt",
					  (char*) "--out", (char*) "data/io_test/case8/test_pve_est.out.gz"};
//
char* argv_main1[] = { (char*) "--mode_pve_est",
	                   (char*) "--random_seed", (char*) "1",
	                   (char*) "--n_pve_samples", (char*) "3",
	                   (char*) "--bgen", (char*) "data/io_test/n1000_p2000.bgen",
	                   (char*) "--pheno", (char*) "data/io_test/case8/pheno.txt",
	                   (char*) "--out", (char*) "data/io_test/case8/test_pve_est.out.gz"};

char* argv_main2[] = { (char*) "--mode_pve_est",
	                   (char*) "--random_seed", (char*) "1",
	                   (char*) "--n_pve_samples", (char*) "3",
	                   (char*) "--bgen", (char*) "data/io_test/n1000_p2000.bgen",
	                   (char*) "--pheno", (char*) "data/io_test/case8/pheno.txt",
	                   (char*) "--pve_mog_weights", (char*) "data/io_test/case8/test_mog_weights.txt",
	                   (char*) "--out", (char*) "data/io_test/case8/test_pve_est.out.gz"};


TEST_CASE("HE-reg"){
	SECTION("GxE effects fit"){
		parameters p;
		int argc = sizeof(argv_pve1)/sizeof(argv_pve1[0]);
		parse_arguments(p, argc, argv_pve1);
		Data data( p );
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();
		Eigen::VectorXd eta = data.E.col(0);
		PVE pve(data, Y, C, eta);
		pve.run();

		CHECK(pve.sigmas(0)  == Approx(0.4755321728));
		CHECK(pve.sigmas(1)  == Approx(0.0632032343));
		CHECK(pve.sigmas(2)  == Approx(0.274279032));
		CHECK(pve.h2(0)  == Approx(0.5849000337));
		CHECK(pve.h2(1)  == Approx(0.0777393749));
		CHECK(pve.h2(2)  == Approx(0.3373605914));

		CHECK(pve.h2_se_jack(0)  == Approx(0.2647600098));
		CHECK(pve.h2_se_jack(1)  == Approx(0.0452209812));
		CHECK(pve.h2_se_jack(2)  == Approx(0.2481256178));
		pve.to_file(p.out_file);
	}

	SECTION("GxE effects fit (stream version)"){
		parameters p;
		int argc = sizeof(argv_pve2)/sizeof(argv_pve2[0]);
		parse_arguments(p, argc, argv_pve2);
		Data data( p );
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();
		Eigen::VectorXd eta = data.E.col(0);
		PVE pve(data, Y, C, eta);
		pve.run();

		CHECK(pve.sigmas(0)  == Approx(0.4751298802));
		CHECK(pve.sigmas(1)  == Approx(0.0632593406));
		CHECK(pve.sigmas(2)  == Approx(0.2745515453));
		CHECK(pve.h2(0)  == Approx(0.5844581795));
		CHECK(pve.h2(1)  == Approx(0.0778154366));
		CHECK(pve.h2(2)  == Approx(0.337726384));

		CHECK(pve.h2_se_jack(0)  == Approx(0.3175758074));
		CHECK(pve.h2_se_jack(1)  == Approx(0.0125467466));
		CHECK(pve.h2_se_jack(2)  == Approx(0.2169781382));
	}

	SECTION("Main effects fit (gaussian prior)"){
		parameters p;
		int argc = sizeof(argv_main1)/sizeof(argv_main1[0]);
		parse_arguments(p, argc, argv_main1);
		Data data( p );
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();

		PVE pve(data, Y, C);
		pve.run();

		CHECK(pve.sigmas(0)  == Approx(0.5237752503));
		CHECK(pve.sigmas(1)  == Approx(0.49942111));
		CHECK(pve.h2(0)  == Approx(0.51190101));
		CHECK(pve.h2(1)  == Approx(0.48809899));
		std::cout << p.out_file << std::endl;
		pve.to_file(p.out_file);

		CHECK(pve.h2_se_jack(0)  == Approx(0.193659797));
		CHECK(pve.h2_se_jack(1)  == Approx(0.1653581733));
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
			PVE pve(data, Y, data.C, data.vp_init.eta);
			pve.run();

			// RHE with 1st jacknife block removed
			for (long ii = 0; ii < pve.n_components; ii++) {
				pve.components[ii].rm_jacknife_block = 0;
			}
			CHECK(pve.components[0].get_n_var_local() == Approx(924));
			CHECK(pve.components[0].getXXtz().squaredNorm() == Approx(39653744616.7472076416));
			CHECK(pve.components[1].getXXtz().squaredNorm() == Approx(60669447736.4095077515));

			Eigen::MatrixXd CC = pve.construct_vc_system(pve.components);
			Eigen::MatrixXd AA = CC.block(0, 0, pve.n_components, pve.n_components);
			Eigen::VectorXd bb = CC.col(pve.n_components);
			Eigen::VectorXd ss = AA.colPivHouseholderQr().solve(bb);

			CHECK(CC(0, 0) == Approx(4621.41));
			CHECK(CC(1, 1) == Approx(7041.18));
			CHECK(CC(1, 0) == Approx(1128.3));
			CHECK(CC(1, 2) == Approx(1036.74));

			CHECK(pve.components[0].zz.sum() == Approx(123.7670672474));
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
			PVE pve(data, Y, data.C, data.vp_init.eta);
			pve.run();

			// RHE on full data with 1st block removed
			for (long ii = 0; ii < pve.n_components; ii++) {
				pve.components[ii].rm_jacknife_block = -1;
			}
			CHECK(pve.components[0].get_n_var_local() == Approx(924));
			CHECK(pve.components[0].getXXtz().squaredNorm() == Approx(39653744616.7472076416));
			CHECK(pve.components[1].getXXtz().squaredNorm() == Approx(60669447736.4095077515));

			Eigen::MatrixXd CC = pve.construct_vc_system(pve.components);
			Eigen::MatrixXd AA = CC.block(0, 0, pve.n_components, pve.n_components);
			Eigen::VectorXd bb = CC.col(pve.n_components);
			Eigen::VectorXd ss = AA.colPivHouseholderQr().solve(bb);

			CHECK(CC(0, 0) == Approx(4621.41));
			CHECK(CC(1, 1) == Approx(7041.18));
			CHECK(CC(1, 0) == Approx(1128.3));
			CHECK(CC(1, 2) == Approx(1036.74));
		}
	}
}
