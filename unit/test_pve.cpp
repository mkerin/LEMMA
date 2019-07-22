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
		pve.run();

		CHECK(pve.sigmas(0)  == Approx(0.4757065487));
		CHECK(pve.sigmas(1)  == Approx(0.0630985531));
		CHECK(pve.sigmas(2)  == Approx(0.273900849));
		CHECK(pve.h2(0)  == Approx(0.5853366131));
		CHECK(pve.h2(1)  == Approx(0.0776400776));
		CHECK(pve.h2(2)  == Approx(0.3370233093));
		pve.to_file(p.out_file);
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

		SECTION("Gaussian prior"){
			PVE pve(data, Y, C);
			pve.run();

			CHECK(pve.sigmas(0)  == Approx(0.5240473249));
			CHECK(pve.sigmas(1)  == Approx(0.4986619236));
			CHECK(pve.h2(0)  == Approx(0.5124108594));
			CHECK(pve.h2(1)  == Approx(0.4875891406));
			std::cout << p.out_file << std::endl;
			pve.to_file(p.out_file);
		}
	}
}
