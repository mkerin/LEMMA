// tests-main.cpp
#include "catch.hpp"

#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes.hpp"
#include "../src/data.hpp"
#include "../src/hyps.hpp"
#include "../src/genotype_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <sys/stat.h>

char* case_study_args[] = { (char*) "bin/bgen_prog", (char*) "--VB-varEM",
	                        (char*) "--VB-iter-max", (char*) "10",
	                        (char*) "--environment", (char*) "unit/data/n50_p100_env.txt",
	                        (char*) "--bgen", (char*) "unit/data/n50_p100.bgen",
	                        (char*) "--out", (char*) "unit/data/test_main.out.gz",
	                        (char*) "--pheno", (char*) "unit/data/pheno.txt",
	                        (char*) "--hyps-grid", (char*) "unit/data/single_hyps_gxage.txt",
	                        (char*) "--hyps-probs", (char*) "unit/data/single_hyps_gxage_probs.txt"};

TEST_CASE( "Case study: varEM w/ multi-env + MoG + covars" ){
	parameters p;
	int argc = sizeof(case_study_args)/sizeof(case_study_args[0]);
	parse_arguments(p, argc, case_study_args);
	Data data( p );

	data.read_non_genetic_data();
	data.standardise_non_genetic_data();
	data.read_full_bgen();

	data.calc_dxteex();
	data.calc_snpstats();
	data.set_vb_init();
	VBayes VB(data);
	SECTION("Initialised correctly"){
		CHECK(VB.Nglobal == 50);
		CHECK(VB.Nglobal == 50.0);
		CHECK(VB.n_env == 4);
		CHECK(VB.n_var == 69);
		CHECK(VB.n_effects == 2);
		CHECK(VB.vp_init.muw(0) == 0.25);
	}

	std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
	SECTION("Explicitly checking vparams after updates") {
		// Set up for RunInnerLoop
		long n_grid = VB.hyps_inits.size();
		std::vector<Hyps> all_hyps = VB.hyps_inits;

		// Set up for updateAllParams
		std::vector<VariationalParameters> all_vp;
		VB.setup_variational_params(all_hyps, all_vp);

		int round_index = 2;
		std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
		std::vector<std::vector< double > > logw_updates(n_grid);
		VariationalParameters& vp = all_vp[0];
		Hyps& hyps = all_hyps[0];

		EigenDataVector check_ym;
		Eigen::VectorXd Eq_beta;

		VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev);

		CHECK(vp.alpha_beta(0)            == Approx(0.1304866836));
		CHECK(vp.alpha_beta(63)           == Approx(0.1287316327));
		CHECK(vp.mean_gam(0)           == Approx(-0.0008928842));
		CHECK(vp.mean_gam(63)           == Approx(0.0006366846));
		CHECK(vp.muw(0, 0)              == Approx(0.0881921519));
		CHECK(vp.sw_sq(0, 0)              == Approx(0.3894154753));

		CHECK(hyps.sigma                == Approx(0.9526519885));
		CHECK(hyps.lambda[0]            == Approx(0.1363528774));
		CHECK(hyps.lambda[1]            == Approx(0.1354853332));
		CHECK(hyps.slab_relative_var[0] == Approx(0.0038085952));
		CHECK(hyps.slab_relative_var[1] == Approx(0.0037640112));

		Eq_beta = vp.alpha_beta * vp.mu1_beta;
		if(p.mode_mog_prior_beta) Eq_beta.array() += (1 - vp.alpha_beta) * vp.mu2_beta;
		check_ym  = VB.X * Eq_beta;
		check_ym += VB.C * vp.muc.cast<scalarData>().matrix();
		CHECK(vp.ym(0)            == Approx(check_ym(0)));

		VB.updateAllParams(1, round_index, all_vp, all_hyps, logw_prev);

		CHECK(vp.alpha_beta(0)            == Approx(0.1299179906));
		CHECK(vp.muw(0, 0)              == Approx(0.0252971947));
		CHECK(vp.alpha_gam(63)           == Approx(0.1206189086));
		CHECK(vp.mu1_gam(63)              == Approx(0.0017733056));
		CHECK(vp.s1_gam_sq(63)            == Approx(0.0027427621));

		Eq_beta = vp.alpha_beta * vp.mu1_beta;
		if(p.mode_mog_prior_beta) Eq_beta.array() += (1 - vp.alpha_beta) * vp.mu2_beta;
		check_ym  = VB.X * Eq_beta;
		check_ym += VB.C * vp.muc.cast<scalarData>().matrix();
		CHECK(vp.ym(0)            == Approx(check_ym(0)));

		VB.updateAllParams(2, round_index, all_vp, all_hyps, logw_prev);

		CHECK(vp.alpha_beta(63)           == Approx(0.12755139));
		CHECK(vp.muw(0, 0)              == Approx(0.00704975));
		CHECK(vp.alpha_gam(63)           == Approx(0.1089248778));
		CHECK(vp.mu1_gam(63)              == Approx(0.0004345884));

		CHECK(VB.calc_logw(hyps, vp) == Approx(-96.5785826372));
		VbTracker tracker(p);
		tracker.init_interim_output(0,2, VB.n_effects, VB.n_covar, VB.n_env, VB.env_names, vp);
		tracker.dump_state("2", VB.n_samples, VB.n_covar, VB.n_var, VB.n_env,
		                   VB.n_effects, vp, hyps, VB.Y, VB.C, VB.X,
		                   VB.covar_names, VB.env_names, VB.sample_is_invalid, VB.sample_location);

		// variances
		CHECK(vp.EdZtZ.sum() == Approx(7153.6186444063));
	}

	VB.run_inference(VB.hyps_inits, false, 2, trackers);
	SECTION("ELBO as expected"){
		CHECK(trackers[0].count == 10);
		CHECK(trackers[0].logw == Approx(-95.5990828788));
	}
}

TEST_CASE("Case Study: var-EM w/ sample subsetting"){
	parameters p;
	int argc = sizeof(case_study_args)/sizeof(case_study_args[0]);
	parse_arguments(p, argc, case_study_args);
	p.incl_sids_file = "unit/data/n25_sample_ids.txt";
	Data data( p );

	data.read_non_genetic_data();
	data.standardise_non_genetic_data();
	std::cout << "Standardised" << std::endl;
	data.read_full_bgen();
	std::cout << "finished reading bgen" << std::endl;

	data.calc_dxteex();
	std::cout << "calc dxteex" << std::endl;
	data.calc_snpstats();
	data.set_vb_init();
	VBayes VB(data);
	SECTION("Initialised correctly"){
		CHECK(VB.Nglobal == 25);
		CHECK(VB.n_var == 65);
	}

	std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);
	VB.run_inference(VB.hyps_inits, false, 2, trackers);
	SECTION("Inference as expected"){
		CHECK(trackers[0].count == 10);
		CHECK(trackers[0].logw == Approx(-56.6794307838));
	}
}

TEST_CASE("NaN vparam update throws exception" ){
	parameters p;
	int argc = sizeof(case_study_args)/sizeof(case_study_args[0]);
	parse_arguments(p, argc, case_study_args);
	Data data( p );
	data.read_non_genetic_data();
	data.standardise_non_genetic_data();
	data.read_full_bgen();

	data.calc_dxteex();
	data.set_vb_init();
	VBayes VB(data);

	std::vector< VbTracker > trackers(VB.hyps_inits.size(), p);

	// Set up for RunInnerLoop
	long n_grid = VB.hyps_inits.size();
	long n_samples = VB.n_samples;
	std::vector<Hyps> all_hyps = VB.hyps_inits;

	// Set up for updateAllParams
	std::vector<VariationalParameters> all_vp;
	VB.setup_variational_params(all_hyps, all_vp);
	VariationalParameters& vp = all_vp[0];
	Hyps& hyps = all_hyps[0];

	int round_index = 2;
	std::vector<double> logw_prev(n_grid, -std::numeric_limits<double>::max());
	std::vector<std::vector< double > > logw_updates(n_grid);

	vp.alpha_beta(0) = std::nan("1");

	CHECK_THROWS(VB.updateAllParams(0, round_index, all_vp, all_hyps, logw_prev));
}
