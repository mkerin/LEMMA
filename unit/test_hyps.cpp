//
// Created by kerin on 2019-03-02.
//
#include "catch.hpp"

#include <vector>
#include "../src/tools/Eigen/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes.hpp"
#include "../src/data.hpp"
#include "../src/hyps.hpp"

TEST_CASE("Hyps"){
	parameters p;
	p.random_seed = 1;

	SECTION("Copy constructor in vectors"){
		Hyps hyps(p);
		hyps.sigma = 1;
		hyps.pve.resize(1);
		hyps.pve(0) = 1;

		std::vector<Hyps> all_hyps(2, hyps);
		std::vector<Hyps> theta0 = all_hyps;

		theta0[0].sigma = 2;
		theta0[0].pve(0) = 2;

		CHECK(all_hyps[0].sigma == 1);
		CHECK(all_hyps[0].pve(0) == 1);
	}

	SECTION("Default initialisation"){
		Hyps hyps(p);
		Eigen::VectorXd soln;

		soln = hyps.get_sigmas(0.1, 0.01, 0.05, 1000);
		CHECK(soln[0] == Approx(0.0005555556));
		CHECK(soln[1] == Approx(0.0001066218));

		soln = hyps.get_sigmas(0.1, 0, 0.01, 0.01, 0.05, 0.05, 1000);
		CHECK(soln[0] == Approx(0.0005555556));
		CHECK(soln[1] == Approx(0.0001066218));
		CHECK(soln[2] == Approx(0));
		CHECK(soln[3] == Approx(0));

		soln = hyps.get_sigmas(0, 0.1, 0.01, 0.01, 0.05, 0.05, 1000);
		CHECK(soln[0] == Approx(0));
		CHECK(soln[1] == Approx(0));
		CHECK(soln[2] == Approx(0.0005555556));
		CHECK(soln[3] == Approx(0.0001066218));

		hyps.random_init(2, 1000);
		CHECK(hyps.slab_relative_var[0] == Approx(0.0075781524));
		CHECK(hyps.spike_relative_var[0] == Approx(0.0000075782));
		CHECK(hyps.slab_relative_var[1] == Approx(0.0122428337));
		CHECK(hyps.spike_relative_var[1] == Approx(0.0000122428));
	}

	SECTION("Random initialization"){
		SECTION("G case") {
			Hyps hyps(p);
			hyps.random_init(2, 1000);
			CHECK(hyps.spike_var(0) == Approx(0.0000075782));
			CHECK(hyps.slab_var(0) == Approx(0.0075781524));
			CHECK(hyps.spike_var(1) == Approx(0.0000122428));
			CHECK(hyps.slab_var(1) == Approx(0.0122428337));
		}

		SECTION("GxE case") {
			Hyps hyps(p);
			hyps.random_init(1, 1000);
			CHECK(hyps.spike_var(0) == Approx(0.0000068725));
			CHECK(hyps.slab_var(0) == Approx(0.0068724729));
		}
	}
}
