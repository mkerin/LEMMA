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
		Eigen::MatrixXd C = data.W.cast<double>();
		Eigen::VectorXd eta = data.E.col(0);
		PVE pve(p, data.G, Y, C, eta);
		pve.run(p.out_file);

		CHECK(pve.sigmas(0)  == Approx(0.519333642));
		CHECK(pve.sigmas(1)  == Approx(0.1156965252));
		CHECK(pve.sigmas(2)  == Approx(0.3929522193));
		CHECK(pve.h2(0)  == Approx(0.5051970236));
		CHECK(pve.h2(1)  == Approx(0.1125471863));
		CHECK(pve.h2(2)  == Approx(0.3822557901));
	}

	SECTION("Main effects fit (gaussian prior)"){
		parameters p;
		int argc = sizeof(argv_pve1)/sizeof(argv_pve1[0]);
		parse_arguments(p, argc, argv_pve1);
		Data data( p );
		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C = data.W.cast<double>();

		SECTION("Gaussian prior"){
			PVE pve(p, data.G, Y, C);
			pve.run(p.out_file);

			CHECK(pve.sigmas(0)  == Approx(0.5240473249));
			CHECK(pve.sigmas(1)  == Approx(0.4986619236));
			CHECK(pve.h2(0)  == Approx(0.5124108594));
			CHECK(pve.h2(1)  == Approx(0.4875891406));
		}

		SECTION("MoG prior v1"){
			Eigen::VectorXd alpha_beta = Eigen::VectorXd::Constant(data.n_var, 0.99999999);
			Eigen::VectorXd alpha_gam = Eigen::VectorXd::Constant(data.n_var, 0.99999999);
			PVE pve(p, data.G, Y, C);
			pve.set_mog_weights(alpha_beta, alpha_gam);
			pve.run(p.out_file);

			CHECK(pve.sigmas(0)  == Approx(0.5240473249));
			CHECK(pve.sigmas(1)  == Approx(0.0));
			CHECK(pve.sigmas(2)  == Approx(0.4986619236));
			CHECK(pve.h2(0)  == Approx(0.5124108594));
			CHECK(pve.h2(1)  == Approx(0.0));
			CHECK(pve.h2(2)  == Approx(0.4875891406));
		}

		SECTION("MoG prior v2"){
			Eigen::VectorXd alpha_beta = Eigen::VectorXd::Constant(data.n_var, 0.00000001);
			Eigen::VectorXd alpha_gam = Eigen::VectorXd::Constant(data.n_var, 0.00000001);
			PVE pve(p, data.G, Y, C);
			pve.set_mog_weights(alpha_beta, alpha_gam);
			pve.run(p.out_file);

			CHECK(pve.sigmas(0)  == Approx(0.0));
			CHECK(pve.sigmas(1)  == Approx(0.5240473249));
			CHECK(pve.sigmas(2)  == Approx(0.4986619236));
			CHECK(pve.h2(0)  == Approx(0.0));
			CHECK(pve.h2(1)  == Approx(0.5124108594));
			CHECK(pve.h2(2)  == Approx(0.4875891406));
		}

		SECTION("MoG prior v3"){
			long pp = data.n_var;
			Eigen::VectorXd zeros = Eigen::VectorXd::Zero((pp + pp % 2) / 2);
			Eigen::VectorXd ones = Eigen::VectorXd::Constant(pp/2, 1.0);
			Eigen::VectorXd alpha_beta(pp), alpha_gam(pp);
			alpha_beta << zeros, ones;
			alpha_gam << zeros, ones;
			PVE pve(p, data.G, Y, C);
			pve.set_mog_weights(alpha_beta, alpha_gam);
			pve.run(p.out_file);

			CHECK(pve.sigmas(0)  == Approx(0.4321062545));
			CHECK(pve.sigmas(1)  == Approx(0.1218897171));
			CHECK(pve.sigmas(2)  == Approx(0.4714699134));
			CHECK(pve.h2(0)  == Approx(0.4213755531));
			CHECK(pve.h2(1)  == Approx(0.1188627714));
			CHECK(pve.h2(2)  == Approx(0.4597616755));
		}
	}
}