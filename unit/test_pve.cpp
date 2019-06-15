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
		pve.run(p.out_file);

		CHECK(pve.sigmas(0)  == Approx(0.4757065487));
		CHECK(pve.sigmas(1)  == Approx(0.0630985531));
		CHECK(pve.sigmas(2)  == Approx(0.273900849));
		CHECK(pve.h2(0)  == Approx(0.5853366131));
		CHECK(pve.h2(1)  == Approx(0.0776400776));
		CHECK(pve.h2(2)  == Approx(0.3370233093));
	}

	SECTION("Main effects fit (single sample; gaussian prior)"){
		parameters p;
		int argc = sizeof(argv_main1)/sizeof(argv_main1[0]);
		parse_arguments(p, argc, argv_main1);
		p.n_pve_samples = 1;
		Data data( p );
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();

		SECTION("Gaussian prior"){
			PVE pve(data, Y, C);
			pve.run(p.out_file);

			CHECK(pve.bb(0)  == Approx(1872.2247892629));
			CHECK(pve.bb(1)  == Approx(999.0));
			CHECK(pve.sigmas(0)  == Approx(0.5296102553));
			CHECK(pve.sigmas(1)  == Approx(0.4895490047));
//			CHECK(pve.h2(0)  == Approx(0.5124108594));
//			CHECK(pve.h2(1)  == Approx(0.4875891406));
		}

	SECTION("Main effects fit (gaussian prior)") {
		parameters p;
		int argc = sizeof(argv_main1) / sizeof(argv_main1[0]);
		parse_arguments(p, argc, argv_main1);
		Data data(p);
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.C.cast<double>();

		SECTION("Gaussian prior") {
			PVE pve(data, Y, C);
			pve.run(p.out_file);

			CHECK(pve.sigmas(0) == Approx(0.5240473249));
			CHECK(pve.sigmas(1) == Approx(0.4986619236));
		}
	}
//
//		SECTION("MoG prior v1"){
//			Eigen::VectorXd alpha_beta = Eigen::VectorXd::Constant(data.n_var, 0.99999999);
//			Eigen::VectorXd alpha_gam = Eigen::VectorXd::Constant(data.n_var, 0.99999999);
//			PVE pve(p, data.G, Y, C);
//			pve.set_mog_weights(alpha_beta, alpha_gam);
//			pve.run(p.out_file);
//
//			CHECK(pve.sigmas(0)  == Approx(0.5240473249));
//			CHECK(pve.sigmas(1)  == Approx(0.0));
//			CHECK(pve.sigmas(2)  == Approx(0.4986619236));
//			CHECK(pve.h2(0)  == Approx(0.5124108594));
//			CHECK(pve.h2(1)  == Approx(0.0));
//			CHECK(pve.h2(2)  == Approx(0.4875891406));
//		}
//
//		SECTION("MoG prior v2"){
//			Eigen::VectorXd alpha_beta = Eigen::VectorXd::Constant(data.n_var, 0.00000001);
//			Eigen::VectorXd alpha_gam = Eigen::VectorXd::Constant(data.n_var, 0.00000001);
//			PVE pve(p, data.G, Y, C);
//			pve.set_mog_weights(alpha_beta, alpha_gam);
//			pve.run(p.out_file);
//
//			CHECK(pve.sigmas(0)  == Approx(0.0));
//			CHECK(pve.sigmas(1)  == Approx(0.5240473249));
//			CHECK(pve.sigmas(2)  == Approx(0.4986619236));
//			CHECK(pve.h2(0)  == Approx(0.0));
//			CHECK(pve.h2(1)  == Approx(0.5124108594));
//			CHECK(pve.h2(2)  == Approx(0.4875891406));
//		}
	}
//
//	SECTION("Main effects fit (MoG prior)") {
//		/* R
//		 * eps = 0.0000001
//		 * aa = data.frame(alpha_beta = c(rep(eps, 905), rep(1-eps, 905)), alpha_gam = c(rep(eps, 905), rep(1-eps, 905)))
//		 * write.table(aa, "data/io_test/case8/test_mog_weights.txt", col.names=T, row.names=F, quote=F)
//		 */
//		parameters p;
//		int argc = sizeof(argv_main2) / sizeof(argv_main2[0]);
//		parse_arguments(p, argc, argv_main2);
//		Data data(p);
//		data.read_non_genetic_data();
//		data.standardise_non_genetic_data();
//		data.read_full_bgen();
//
//		Eigen::VectorXd Y = data.Y.cast<double>();
//		Eigen::MatrixXd C = data.C.cast<double>();
//		PVE pve(p, data.G, Y, C);
//
//		Eigen::VectorXd alpha_beta, alpha_gam;
//		data.read_mog_weights(p.mog_weights_file, alpha_beta, alpha_gam);
//		pve.set_mog_weights(alpha_beta, alpha_gam);
//		pve.run(p.out_file);
//
//		CHECK(pve.sigmas(0) == Approx(0.4321062545));
//		CHECK(pve.sigmas(1) == Approx(0.1218897171));
//		CHECK(pve.sigmas(2) == Approx(0.4714699134));
//		CHECK(pve.h2(0) == Approx(0.4213755531));
//		CHECK(pve.h2(1) == Approx(0.1188627714));
//		CHECK(pve.h2(2) == Approx(0.4597616755));
//	}
}
