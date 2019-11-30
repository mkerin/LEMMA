//
// Created by kerin on 2018-01-26.
//

#include "parse_arguments.hpp"
#include "parameters.hpp"
#include "data.hpp"
#include "vbayes_x2.hpp"
#include "rhe_reg.hpp"
#include "genotype_matrix.hpp"
#include "mpi_utils.hpp"

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

	// Sanitise std::cout
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	std::ofstream sink("/dev/null");
	std::streambuf *coutbuf = std::cout.rdbuf();
	if (world_rank != 0) {
		std::cout.rdbuf(sink.rdbuf());
	}

	std::cout << "============="<< std::endl;
	std::cout << "LEMMA v" << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << std::endl;
	std::cout << "=============" << std::endl << std::endl;

	auto start = std::chrono::system_clock::now();
	auto data_start = std::chrono::system_clock::now();
	parse_arguments(p, argc, argv);

	std::time_t start_time = std::chrono::system_clock::to_time_t(start);
	std::cout << "Starting analysis at " << std::ctime(&start_time) << std::endl;

	Data data(p);
	data.apply_filters();
	data.read_non_genetic_data();
	data.standardise_non_genetic_data();
	if(p.mode_dump_processed_data) {
		data.dump_processed_data();
	}
	data.read_full_bgen();
	data.set_vb_init();

	if (p.mode_vb || (p.mode_calc_snpstats && p.resume_prefix != "NULL")) {
		VBayesX2 VB(data);
		if (p.mode_vb) {
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
			data.vp_init = VB.vp_init;

//				std::cout << std::endl << "Time expenditure:" << std::endl;
//				std::cout << "Reading data: " << elapsed_reading_data.count() << " secs" << std::endl;
//				std::cout << "VB inference: " << elapsed_vb.count() << " secs" << std::endl;
//				std::cout << "runInnerLoop: " << VB.elapsed_innerLoop.count() << " secs" << std::endl;

			// Write time log to file
			boost_io::filtering_ostream outf_time;
			std::string ofile_map = VB.fstream_init(outf_time, "", "_time_elapsed");
			outf_time << "function time" << std::endl;
			outf_time << "read_data " << elapsed_reading_data.count() << std::endl;
			outf_time << "VB_prepro " << elapsed_reading_data.count() << std::endl;
			outf_time << "VB_inference " << elapsed_vb.count() << std::endl;
			outf_time << "SNP_testing " << VB.elapsed_innerLoop.count() << std::endl;
		}

		if (p.mode_calc_snpstats) {
			VB.output_results("");
		}
	}

	if (!p.streamBgenFiles.empty() && p.mode_calc_snpstats) {
		GenotypeMatrix Xstream(p, false);
		bool bgen_pass = true;
		bool append = false;

		std::cout << "Computing single-snp hypothesis tests" << std::endl;
		boost_io::filtering_ostream outf;
		if(world_rank == 0) {
			if (p.streamBgenOutFile != "NULL"){
				fileUtils::fstream_init(outf, p.streamBgenOutFile);
				std::cout << "Writing single SNP hypothesis tests to file: " << p.streamBgenOutFile << std::endl;
			} else {
				fileUtils::fstream_init(outf, p.out_file);
				std::cout << "Writing single SNP hypothesis tests to file: " << p.out_file << std::endl;
			}
		}

		long n_var_parsed = 0, nChunk = 0;
		long print_interval = p.streamBgen_print_interval;
		if (p.debug) print_interval = 1;
		long long n_vars_tot = 0;
		for (int ii = 0; ii < p.streamBgenFiles.size(); ii++) {
			n_vars_tot += data.streamBgenViews[ii]->number_of_variants();
		}

		for (int ii = 0; ii < p.streamBgenFiles.size(); ii++) {
			std::cout << "Streaming genotypes from " << p.streamBgenFiles[ii] << std::endl;
			while (fileUtils::read_bgen_chunk(data.streamBgenViews[ii], Xstream, data.sample_is_invalid,
			                                  data.n_samples, 128, p, bgen_pass, n_var_parsed)) {
				if (nChunk % print_interval == 0 && nChunk > 0) {
					std::cout << "Chunk " << nChunk << " read (size " << 128;
					std::cout << ", " << n_var_parsed - 1 << "/" << n_vars_tot;
					std::cout << " variants parsed)" << std::endl;
				}

				Eigen::MatrixXd nlogp(Xstream.cols(), 3), chiSq(Xstream.cols(), 3);
				if (data.n_env > 0) {
					computeSingleSnpTests(Xstream.G, nlogp, chiSq, data.Y, data.vp_init.eta);

					if (world_rank == 0) {
						fileUtils::write_snp_stats_to_file(outf, data.n_effects, Xstream, append,
						                                   nlogp.col(0), nlogp.col(1), nlogp.col(2),
						                                   chiSq.col(0), chiSq.col(1), chiSq.col(2));
					}
				} else {
					assert(false);
				}
				append = true;
				nChunk++;
			}
		}
	}

	if(p.mode_RHE) {
		if(p.random_seed == -1) {
			if(world_rank == 0) {
				std::random_device rd;
				p.random_seed = rd();
			}
			MPI_Bcast(&p.random_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
		}
		std::cout << "Initialising random sample generator with seed " << p.random_seed << std::endl;

		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C;
		if (p.extra_pve_covar_file != "NULL") {
			C.resize(data.n_samples, data.C.cols() + data.C_extra_pve.cols());
			C << data.C.cast<double>(), data.C_extra_pve.cast<double>();
		} else {
			C.resize(data.n_samples, data.C.cols());
			C << data.C.cast<double>();
		}

		std::string out_file = p.out_file;
		out_file = out_file.substr(0, out_file.find(".gz"));
		if(data.n_env > 0) {
			// If multi env; use VB to collapse to single
			if(p.mode_vb || p.env_coeffs_file != "NULL") {
				RHEreg pve(data, Y, C, data.vp_init.eta);
				pve.run();
				pve.to_file(p.out_file);

				// Write time log to file
				boost_io::filtering_ostream outf_time;
				std::string ofile_map = fileUtils::fstream_init(outf_time, p.out_file, "", "_time_elapsed");
				outf_time << "function time" << std::endl;
				outf_time << "streamBgen " << pve.elapsed_streamBgen.count() << std::endl;
				outf_time << "solveRHE " << pve.elapsed_solveRHE.count() << std::endl;
				outf_time << "solveJacknife " << pve.elapsed_solveJacknife.count() << std::endl;
				outf_time << "LevMar " << pve.elapsed_LM << std::endl;
				outf_time << "LevMar_copyData " << pve.elapsed_LM_copyData << std::endl;
			} else {
				RHEreg pve(data, Y, C, data.E);
				pve.run();
				pve.to_file(p.out_file);

				// Write time log to file
				boost_io::filtering_ostream outf_time;
				std::string ofile_map = fileUtils::fstream_init(outf_time, p.out_file, "", "_time_elapsed");
				outf_time << "function time" << std::endl;
				outf_time << "streamBgen " << pve.elapsed_streamBgen.count() << std::endl;
				outf_time << "solveRHE " << pve.elapsed_solveRHE.count() << std::endl;
				outf_time << "solveJacknife " << pve.elapsed_solveJacknife.count() << std::endl;
				outf_time << "LevMar " << pve.elapsed_LM << std::endl;
				outf_time << "LevMar_copyData " << pve.elapsed_LM_copyData << std::endl;
			}
		} else {
			RHEreg pve(data, Y, C);
			pve.run();
			pve.to_file(p.out_file);

			// Write time log to file
			boost_io::filtering_ostream outf_time;
			std::string ofile_map = fileUtils::fstream_init(outf_time, p.out_file, "", "_time_elapsed");
			outf_time << "function time" << std::endl;
			outf_time << "streamBgen " << pve.elapsed_streamBgen.count() << std::endl;
			outf_time << "solveRHE " << pve.elapsed_solveRHE.count() << std::endl;
			outf_time << "solveJacknife " << pve.elapsed_solveJacknife.count() << std::endl;
			outf_time << "LevMar " << pve.elapsed_LM << std::endl;
			outf_time << "LevMar_copyData " << pve.elapsed_LM_copyData << std::endl;
		}
	}

	auto end = std::chrono::system_clock::now();
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	std::cout << std::endl << "Analysis finished at " << std::ctime(&end_time);
	if (world_rank != 0) {
		std::cout.rdbuf(coutbuf);
	}
	MPI_Finalize();
	return 0;
}
