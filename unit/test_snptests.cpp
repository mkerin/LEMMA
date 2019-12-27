//
// Created by kerin on 2019-12-26.
//
#include "catch.hpp"

#include "../src/tools/eigen3.3/Dense"
#include "../src/parse_arguments.hpp"
#include "../src/vbayes_x2.hpp"
#include "../src/data.hpp"

#include <string>

char* case5a[] = { (char*) "prog",
	               (char*) "--VB",
	               (char*) "--VB-varEM",
	               (char*) "--use_vb_on_covars",
	               (char*) "--VB-iter-max", (char*) "10",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--out", (char*) "data/io_test/test3a.out"};

char* case5b[] = { (char*) "prog",
				   (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
	               (char*) "--singleSnpStats", (char*) "--use_vb_on_covars",
	               (char*) "--resume_from_state",
	               (char*) "data/io_test/r2_interim_files/grid_point_0/test3a_dump_it10",
	               (char*) "--pheno", (char*) "data/io_test/pheno.txt",
	               (char*) "--environment", (char*) "data/io_test/n50_p100_env.txt",
	               (char*) "--out", (char*) "data/io_test/test3b.out"};

char* case5c[] = { (char*) "prog",
				   (char*) "--bgen", (char*) "data/io_test/n50_p100.bgen",
				   (char*) "--singleSnpStats", (char*) "--use_vb_on_covars",
				   (char*) "--pheno", (char*) "data/io_test/test3a_converged_resid_pheno_chr10.out",
				   (char*) "--environment", (char*) "data/io_test/test3a_converged_eta.out",
				   (char*) "--out", (char*) "data/io_test/test3b.out"};

void apply_streamBgen_mode(parameters& p){
	p.streamBgenFiles.push_back(p.bgen_file);
	p.streamBgiFiles.push_back(p.bgen_file + ".bgi");
	p.bgen_file = "NULL";
}

void my_test(std::string mode){
	parameters p;
	bool stream_bgen = false;
	if (mode == "after_lemma") {
		int argc = sizeof(case5a)/sizeof(case5a[0]);
		parse_arguments(p, argc, case5a);
	} else if (mode == "from_state") {
		int argc = sizeof(case5b)/sizeof(case5b[0]);
		parse_arguments(p, argc, case5b);
	} else if(mode == "from_file") {
		int argc = sizeof(case5c)/sizeof(case5c[0]);
		parse_arguments(p, argc, case5c);
		apply_streamBgen_mode(p);
		stream_bgen = true;
	} else {
		throw std::runtime_error("Unexpected test mode");
	}

	Data data(p);
	data.read_non_genetic_data();
	data.standardise_non_genetic_data();
	data.read_full_bgen();
	data.calc_dxteex();
	data.set_vb_init();

	Eigen::MatrixXd neglogPvals, testStats;
	if (stream_bgen) {
		long n_var_parsed = 0;
		GenotypeMatrix Xstream(p, false);
		bool bgen_pass = true;
		long ii=0;
		EigenDataMatrix D;
		fileUtils::read_bgen_chunk(data.streamBgenViews[ii], Xstream, data.sample_is_invalid,
		                           data.n_samples, 128, p, bgen_pass, n_var_parsed);
		Xstream.calc_scaled_values();
		compute_LOCO_pvals(data.Y.col(0), Xstream, data.vp_init, neglogPvals, testStats);
	} else {
		VBayesX2 VB(data);
		long n_chrs = VB.n_chrs;
		std::vector<Eigen::VectorXd> pred_main(n_chrs), pred_int(n_chrs), resid_loco(n_chrs);

		if (mode == "after_lemma") {
			std::vector<VbTracker> trackers(VB.hyps_inits.size(), p);
			VB.run_inference(VB.hyps_inits, false, 2, trackers);
			CHECK(trackers[0].count == 10);
			CHECK(trackers[0].logw == Approx(-88.8466020959));
			data.vp_init = VB.vp_init;
		}

		VB.calcPredEffects(data.vp_init);
		VB.compute_residuals_per_chr(data.vp_init, pred_main, pred_int, resid_loco);
		VB.my_LOCO_pvals(data.vp_init, resid_loco, neglogPvals, testStats);
	}
	CHECK(neglogPvals(0, 0) == Approx(0.2858580487));
	CHECK(neglogPvals(0, 1) == Approx(1.6241936617));
	CHECK(neglogPvals(0, 2) == Approx(2.9866331893));
}


TEST_CASE("singleSnpStats combinations"){
	auto mode = GENERATE(as<std::string>{}, "after_lemma", "from_file");
	DYNAMIC_SECTION("Mode: " << mode << ", streamBgen: " << (mode == "after_lemma" ? "true" : "false")) {
		my_test(mode);
	}
}
