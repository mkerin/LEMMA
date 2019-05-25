// Adapted from example/bgen_to_vcf.cpp by Gavin Band
// From project at http://bitbucket.org/gavinband/bgen/get/master.tar.gz

#define EIGEN_USE_MKL_ALL

#include "parse_arguments.hpp"
#include "parameters.hpp"
#include "data.hpp"
#include "vbayes_x2.hpp"
#include "pve.hpp"
#include "genotype_matrix.hpp"
#include "mpi_utils.hpp"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <memory>


int main( int argc, char** argv ) {
	parameters p;
	MPI_Init(NULL, NULL);
	mpiUtils::sanitise_cout();

	std::cout << "============="<< std::endl;
	std::cout << "LEMMA v" << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << std::endl;
	std::cout << "=============" << std::endl << std::endl;

	auto start = std::chrono::system_clock::now();

	try {
		auto data_start = std::chrono::system_clock::now();
		parse_arguments(p, argc, argv);

		std::time_t start_time = std::chrono::system_clock::to_time_t(start);
		std::cout << "Starting analysis at " << std::ctime(&start_time) << std::endl;

		Data data( p );
		data.apply_filters();
		data.read_non_genetic_data();

		// Also regresses out covariables if necessary
		data.standardise_non_genetic_data();
		data.read_full_bgen();

		// For the HE-reg method
		Eigen::VectorXd eta;
		if (data.n_env > 0) {
			eta = data.E.col(0);
		}

		if(p.mode_vb || p.mode_calc_snpstats) {
			data.set_vb_init();
		}

		VBayesX2 VB(data);
		if(p.mode_vb) {
			if (data.n_effects > 1) {
				data.calc_dxteex();
			}
			if (p.env_coeffs_file == "NULL" && p.init_weights_with_snpwise_scan) {
				data.calc_snpstats();
			}

			auto data_end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_reading_data = data_end - data_start;

			// Run inference
			auto vb_start = std::chrono::system_clock::now();
			VB.run();
			auto vb_end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_vb = vb_end - vb_start;
			eta = VB.vp_init.eta.cast<double>();

			std::cout << std::endl << "Time expenditure:" << std::endl;
			std::cout << "Reading data: " << elapsed_reading_data.count() << " secs" << std::endl;
			std::cout << "VB inference: " << elapsed_vb.count() << " secs" << std::endl;
			std::cout << "runInnerLoop: " << VB.elapsed_innerLoop.count() << " secs" << std::endl;

			// Write time log to file
			boost_io::filtering_ostream outf_time;
			std::string ofile_map = VB.fstream_init(outf_time, "", "_time_elapsed");
			outf_time << "function time" << std::endl;
			outf_time << "read_data " << elapsed_reading_data.count() << std::endl;
			outf_time << "full_inference " << elapsed_vb.count() << std::endl;
			outf_time << "vb_outer_loop " << VB.elapsed_innerLoop.count() << std::endl;
		}
		// ONLY IMPLEMENT MPI FOR VARIATIONAL BAYES
		MPI_Finalize();
		return 0;

		if(p.mode_calc_snpstats) {
			// VB.write_map_stats_to_file("");
		}

//		if(p.streamBgenFile != "NULL") {
//			GenotypeMatrix Xstream(false);
//			bool bgen_pass = true;
//			long n_var_parsed = 0;
//
//			genfile::bgen::IndexQuery::UniquePtr query = genfile::bgen::IndexQuery::create(p.bgi_file);
//			genfile::bgen::View::UniquePtr bgenView;
//			query->initialise();
//
//			bgenView->set_query(query);
//			bgenView->summarise(std::cout);
//
//			Eigen::VectorXd neglogp_beta(VB.n_var);
//			Eigen::VectorXd neglogp_rgam(VB.n_var);
//			Eigen::VectorXd neglogp_gam;
//			Eigen::VectorXd neglogp_joint(VB.n_var);
//			Eigen::VectorXd test_stat_beta(VB.n_var);
//			Eigen::VectorXd test_stat_rgam(VB.n_var);
//			Eigen::VectorXd test_stat_gam;
//			Eigen::VectorXd test_stat_joint(VB.n_var);
//			bool append = false;
//
//			boost_io::filtering_ostream outf;
//			fileUtils::fstream_init(outf, p.streamBgenOutFile);
//
//			while (fileUtils::read_bgen_chunk(bgenView, Xstream, data.incomplete_cases,
//			                                  data.n_samples, 128, p, bgen_pass, n_var_parsed)) {
//				VB.LOCO_pvals_v2(Xstream,
//				                 VB.vp_init,
//				                 p.LOSO_window, neglogp_beta, neglogp_rgam,
//				                 neglogp_joint,
//				                 test_stat_beta,
//				                 test_stat_rgam,
//				                 test_stat_joint);
//
//				fileUtils::write_snp_stats_to_file(outf, VB.n_effects, Xstream, append, neglogp_beta, neglogp_gam,
//				                                   neglogp_rgam, neglogp_joint, test_stat_beta, test_stat_gam,
//				                                   test_stat_rgam, test_stat_joint);
//				append = true;
//			}
//		}

		if(p.mode_pve_est) {
			if(p.random_seed == -1) {
				std::random_device rd;
				p.random_seed = rd();
			}
			std::cout << "Initialising random sample generator with seed " << p.random_seed << std::endl;

			// Eigen::VectorXd eta = data.E.col(0).cast<double>();
			Eigen::VectorXd Y = data.Y.cast<double>();
			Eigen::MatrixXd C = data.C.cast<double>();
			if(data.n_env > 0) {
				// If multi env; use VB to collapse to single
				assert(data.n_env == 1 || p.mode_vb);
				PVE pve(p, data.G, Y, C, eta);
				pve.run(p.out_file);
				pve.to_file(p.out_file);
			} else if(p.mog_weights_file != "NULL") {
				Eigen::VectorXd alpha_beta, alpha_gam;
				data.read_mog_weights(p.mog_weights_file, alpha_beta, alpha_gam);
				PVE pve(p, data.G, Y, C);
				pve.set_mog_weights(alpha_beta, alpha_gam);
				pve.run(p.out_file);
				pve.to_file(p.out_file);
			} else {
				PVE pve(p, data.G, Y, C);
				pve.run(p.out_file);
				pve.to_file(p.out_file);
			}
		}
	}
	catch( genfile::bgen::BGenError const& e ) {
		std::cerr << "!! Uh-oh, error parsing bgen file.\n";
		return -1;
	}

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << "Analysis finished at " << std::ctime(&end_time);
	std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
	return 0;
}
