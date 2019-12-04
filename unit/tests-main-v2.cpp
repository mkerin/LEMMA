// tests-main.cpp
#define CATCH_CONFIG_MAIN
// #define EIGEN_USE_MKL_ALL
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
// #include "../src/hyps.hpp"
#include "../src/genotype_matrix.hpp"

// Scenarios
char* argv_single_env[] = { (char*) "--VB", (char*) "--low_mem",
				 (char*) "--effects_prior_mog", (char*) "--VB-iter-max", (char*) "20",
				 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
				 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
				 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
				 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
				 (char*) "--environment", (char*) "data/io_test/age.txt",
				 (char*) "--out", (char*) "data/io_test/fake_age.out"};
//
char* argv_multi_env[] = { (char*) "--VB", (char*) "--low_mem",
				 (char*) "--effects_prior_mog", (char*) "--VB-iter-max", (char*) "20",
				 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
				 (char*) "--hyps_grid", (char*) "data/io_test/hyperpriors_gxage.txt",
				 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
				 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
				 (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
				 (char*) "--out", (char*) "data/io_test/fake_env.out"};
//
char* argv_main[] = { (char*) "--VB", (char*) "--mode_no_gxe", (char*) "--low_mem",
				 (char*) "--effects_prior_mog", (char*) "--VB-iter-max", (char*) "20",
				 (char*) "--vb_init", (char*) "data/io_test/answer_init.txt",
				 (char*) "--hyps_grid", (char*) "data/io_test/single_hyps_gxage.txt",
				 (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
				 (char*) "--pheno", (char*) "data/io_test/pheno.txt",
				 (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
				 (char*) "--out", (char*) "data/io_test/fake_env.out"};

// Options
// use_vb_on_covars, empirical_bayes, effects_prior_mog
// impose iter max on all

// Check errors;
// Insert Nan

void unpack_hyps()


TEST_CASE( "Check VB: multi env" ){
	parameters p;
	int argc = sizeof(argv_multi_env)/sizeof(argv_multi_env[0]);
	parse_arguments(p, argc, argv_multi_env);

	SECTION("Default"){
		Data data( p );

		data.read_non_genetic_data();
		data.standardise_non_genetic_data();
		data.read_full_bgen();
		data.calc_dxteex();
		data.read_alpha_mu();
		VBayesX2 VB(data);
		VB.check_inputs();
		std::vector< VbTracker > trackers(VB.hyps_grid.rows());

		SECTION("Initial checks"){
			CHECK(data.n_env == 1);
			CHECK(data.n_pheno == 1);
			CHECK(data.n_samples == 50);
			CHECK(data.hyps_grid(0,1) == Approx(0.317067781333932));
			if(p.use_vb_on_covars){
				CHECK(data.Y(0,0)  == Approx(-1.5800573524786081));
				CHECK(data.Y2(0,0) == Approx(-1.5567970303));
				CHECK(data.E(0,0)  == Approx(0.8957059881));
			} else {
				CHECK(data.Y(0,0)  == Approx(-1.5567970303));
				CHECK(data.Y2(0,0) == Approx(-1.5567970303));
			}
			CHECK(data.params.flip_high_maf_variants);
			if(p.low_mem){
				CHECK(data.G(0, 0) == Approx(1.8570984229));
			} else {
				CHECK(data.G(0, 0) == Approx(1.8604233373));
			}

			// VB
			CHECK(VB.n_env == 5);
			CHECK(VB.n_effects == 2);
			CHECK(VB.vp_init.yx(0) == Approx(0.0081544079));
			CHECK(VB.vp_init.eta(0) == Approx(-0.5894793969));
#ifdef DATA_AS_FLOAT
			CHECK((double) VB.vp_init.ym(0) == Approx(0.0003200434));
#else
			CHECK(VB.vp_init.ym(0) == Approx(0.0003200476));
#endif
		}

        SECTION("Ex1. Explicitly checking updates"){
			long n_grid = VB.hyps_grid.rows();
			int round_index = 2;
			std::vector<Hyps> all_hyps;
			std::vector<VariationalParameters> all_vp;
			std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
			std::vector<std::vector< double >> logw_updates(n_grid);

			// Set up for RunInnerLoop
			VB.unpack_hyps(VB.hyps_grid, all_hyps);
			VB.setup_variational_params(all_hyps, all_vp);
			VariationalParameters& vp = all_vp[0];
			Hyps& hyps = all_hyps[0];

			// Ground zero as expected
			CHECK(vp.alpha_beta(0) * vp.mu1_beta(0) == Approx(-0.00015854116408000002));

			VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev, trackers, logw_updates);
			updateAllParams(count, round_index, vp, hyps, logw_prev, logw_updates);

			CHECK(VB.X.col(0)(0) == Approx(1.8570984229));
			CHECK(vp.s1_beta_sq(0) == Approx(0.0031087381));
			CHECK(vp.mu1_beta(0) == Approx(-0.0303900712));
			CHECK(vp.alpha_beta(0) == Approx(0.1447783263));
			CHECK(vp.alpha_beta(1) == Approx(0.1517251004));
			CHECK(vp.mu1_beta(1) == Approx(-0.0355760798));
			CHECK(vp.alpha_beta(63) == Approx(0.1784518373));
			CHECK(VB.calc_logw(hyps, vp) == Approx(-60.983398393));

			VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev, trackers, logw_updates);

			CHECK(vp.alpha_beta(0) == Approx(0.1350711123));
			CHECK(vp.mu1_beta(0) == Approx(-0.0205395866));
			CHECK(vp.alpha_beta(1) == Approx(0.1400764528));
			CHECK(vp.alpha_beta(63) == Approx(0.1769882239));
			CHECK(VB.calc_logw(hyps, vp) == Approx(-60.606081598));
		}

		VB.run_inference(VB.hyps_grid, false, 2, trackers);
		SECTION("Ex1. Vbayes_X2 inference correct") {
			CHECK(trackers[0].count == 33);
			CHECK(trackers[3].count == 33);
			CHECK(trackers[0].logw == Approx(-60.522210486));
			CHECK(trackers[1].logw == Approx(-59.9696083263));
			CHECK(trackers[2].logw == Approx(-60.30658117));
			CHECK(trackers[3].logw == Approx(-61.0687573393));
		}
	}
}
