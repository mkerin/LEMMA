//
// Created by kerin on 2019-03-02.
//
#define EIGEN_USE_MKL_ALL
#include "catch.hpp"

#include <vector>
#include "../src/tools/Eigen/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes_x2.hpp"
#include "../src/data.hpp"
#include "../src/hyps.hpp"

TEST_CASE("Hyps"){
	parameters p;

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

		hyps.use_default_init(2, 1000);
		CHECK(hyps.slab_relative_var[0] == Approx(0.0005882353));
		CHECK(hyps.spike_relative_var[0] == Approx(0.0001128936));
		CHECK(hyps.slab_relative_var[1] == Approx(0.0002941176));
		CHECK(hyps.spike_relative_var[1] == Approx(0.0000564468));
	}

	SECTION("derived initial hyps"){
		p.bgen_file = "data/io_test/n50_p100.bgen";
		p.bgi_file = "data/io_test/n50_p100.bgen.bgi";
		p.pheno_file = "data/io_test/pheno.txt";


		SECTION("Joint effects, MixOfGaussian prior"){
			p.env_file = "data/io_test/n50_p100_env.txt";
			p.hyps_grid_file = "data/io_test/hyps_caseD.txt";

			Data data(p);
			data.read_non_genetic_data();
			data.standardise_non_genetic_data();
			data.read_full_bgen();
			data.calc_dxteex();
			data.set_vb_init();

			VBayesX2 VB(data);
			std::vector<Hyps> all_hyps = VB.hyps_inits;
			std::vector<VariationalParameters> all_vp;
			VB.setup_variational_params(all_hyps, all_vp);

			CHECK(all_vp.size() > 0);
			CHECK(VB.n_var == 67);
			CHECK(all_hyps[0].spike_var(0) == Approx(0.0029850746));
			CHECK(all_hyps[0].slab_var(0) == Approx(0.0029850746));
			CHECK(all_hyps[0].spike_var(1) == Approx(0.0014925373));
			CHECK(all_hyps[0].slab_var(1) == Approx(0.0014925373));
		}
	}
}
