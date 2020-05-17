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

	if (argc == 1){
		return 0;
	}
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
			data.loco_chrs = VB.chrs_present;
			data.resid_loco.resize(data.n_samples, data.loco_chrs.size());
			for (long cc = 0; cc < data.loco_chrs.size(); cc++){
				data.resid_loco.col(cc) = VB.loco_phenos[cc];
			}

			// Write time log to file
			boost_io::filtering_ostream outf_time;
			std::string ofile_map = fileUtils::fstream_init(outf_time, p.out_file, "", "_time_elapsed");
			outf_time << "function time" << std::endl;
			outf_time << "read_data " << elapsed_reading_data.count() << std::endl;
			outf_time << "VB_prepro " << elapsed_reading_data.count() << std::endl;
			outf_time << "VB_inference " << elapsed_vb.count() << std::endl;
			outf_time << "SNP_testing " << VB.elapsed_innerLoop.count() << std::endl;
		}

		if (p.mode_vb || p.force_write_vparams) {
			VB.output_vb_results();
		}

		if (p.streamBgenFiles.empty() && p.mode_calc_snpstats) {
			VB.compute_pvals(data.vp_init);
		}
	}

	if (!p.streamBgenFiles.empty() && p.mode_calc_snpstats) {
		std::cout << "Computing single-snp hypothesis tests" << std::endl;
		boost_io::filtering_ostream outf;
		if(world_rank == 0) {
			if (p.assocOutFile != "NULL") {
				fileUtils::fstream_init(outf, p.assocOutFile);
				std::cout << "Writing single SNP hypothesis tests to file: " << p.assocOutFile << std::endl;
			} else {
				std::string assoc_filepath = p.out_file;
				if(p.mode_vb) {
					assoc_filepath = fileUtils::filepath_format(assoc_filepath, "", "_loco_pvals");
				}
				fileUtils::fstream_init(outf, assoc_filepath);
				std::cout << "Writing single SNP hypothesis tests to file: " << assoc_filepath << std::endl;
			}
		}

		long long n_vars_tot = 0;
		for (int ii = 0; ii < p.streamBgenFiles.size(); ii++) {
			n_vars_tot += data.streamBgenViews[ii]->number_of_variants();
		}

		long ixChr, maxChunkSize = 256;
		long n_var_parsed_tot = 0, nChunk = 0, print_interval = (p.debug ? 1 : p.streamBgen_print_interval);
		bool bgen_pass = true, append = false;
		Eigen::MatrixXd neglogPvals, testStats;
		GenotypeMatrix Xstream(p, false);
		for (int ii = 0; ii < p.streamBgenFiles.size(); ii++) {
			std::cout << "Streaming genotypes from " << p.streamBgenFiles[ii] << std::endl;

			std::vector<int> chrsInFile;
			fileUtils::read_bgen_metadata(p.streamBgenFiles[ii],chrsInFile);
			std::map<long,long> chr_changes;
			long chr = chrsInFile[0];
			for (long jj = 0; jj < chrsInFile.size(); jj++) {
				if (chr != chrsInFile[jj]) {
					chr_changes[chr] = jj;
					chr = chrsInFile[jj];
				}
			}
			chr_changes[chr] = chrsInFile.size();

			long n_var_parsed = 0;
			long chunkSize = std::min(maxChunkSize, chr_changes[chr]-n_var_parsed);
			while (fileUtils::read_bgen_chunk(data.streamBgenViews[ii], Xstream, data.sample_is_invalid,
			                                  data.n_samples, chunkSize, p, bgen_pass, n_var_parsed)) {
				if (nChunk % print_interval == 0 && nChunk > 0) {
					std::cout << "Chunk " << nChunk << " read (size " << chunkSize;
					std::cout << ", " << n_var_parsed_tot+n_var_parsed - 1 << "/" << n_vars_tot;
					std::cout << " variants parsed)" << std::endl;
				}

				int firstChr = Xstream.chromosome[0];
				auto cnt = std::count(Xstream.chromosome.begin(),Xstream.chromosome.end(),firstChr);
				if (cnt != Xstream.cols()){
					throw std::logic_error("Expected only one chromosome in this chunk of data");
				}

				auto it = std::find(data.loco_chrs.begin(),data.loco_chrs.end(),firstChr);
				if (it == data.loco_chrs.end()){
					throw std::runtime_error("Could not locate residualised LOCO phenotype for chromosome "+std::to_string(firstChr));
				} else {
					ixChr = it - data.loco_chrs.begin();
				}

				Xstream.calc_scaled_values();
				if (data.n_env > 0) {
					compute_LOCO_pvals(data.resid_loco.col(ixChr), Xstream, neglogPvals, testStats, data.vp_init.eta);
				} else {
					compute_LOCO_pvals(data.resid_loco.col(ixChr), Xstream, neglogPvals, testStats);
				}

				if (world_rank == 0) {
					fileUtils::write_snp_stats_to_file(outf, data.n_effects, Xstream, append, neglogPvals, testStats);
				}

				append = true;
				nChunk++;
				chunkSize = std::min(maxChunkSize,chr_changes[chr]-n_var_parsed);
			}
			n_var_parsed_tot += n_var_parsed;
		}
		boost_io::close(outf);
	}

	if(p.mode_RHE) {
		Eigen::VectorXd Y = data.Y.cast<double>();
		Eigen::MatrixXd C;
		if (p.extra_pve_covar_file != "NULL") {
			C.resize(data.n_samples, data.C.cols() + data.C_extra_pve.cols());
			C << data.C.cast<double>(), data.C_extra_pve.cast<double>();
		} else {
			C.resize(data.n_samples, data.C.cols());
			C << data.C.cast<double>();
		}

		std::string rhe_filepath = p.out_file;
		if(p.mode_vb || p.mode_calc_snpstats) {
			rhe_filepath = fileUtils::filepath_format(rhe_filepath, "", "_pve");
		}
		if(data.n_env > 0) {
			// If multi env; use VB to collapse to single environment
			if(p.mode_vb || p.env_coeffs_file != "NULL") {
				RHEreg pve(data, Y, C, data.vp_init.eta);
				pve.run();
				pve.to_file(rhe_filepath);

				// Write time log to file
				boost_io::filtering_ostream outf_time;
				std::string ofile_map = fileUtils::fstream_init(outf_time, rhe_filepath, "", "_time_elapsed");
				outf_time << "function time" << std::endl;
				outf_time << "streamBgen " << pve.elapsed_streamBgen.count() << std::endl;
				outf_time << "solveRHE " << pve.elapsed_solveRHE.count() << std::endl;
				outf_time << "solveJacknife " << pve.elapsed_solveJacknife.count() << std::endl;
				outf_time << "LevMar " << pve.elapsed_LM << std::endl;
				outf_time << "LevMar_copyData " << pve.elapsed_LM_copyData << std::endl;
			} else {
				RHEreg pve(data, Y, C, data.E);
				pve.run();
				pve.to_file(rhe_filepath);

				// Write time log to file
				boost_io::filtering_ostream outf_time;
				std::string ofile_map = fileUtils::fstream_init(outf_time, rhe_filepath, "", "_time_elapsed");
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
			pve.to_file(rhe_filepath);

			// Write time log to file
			boost_io::filtering_ostream outf_time;
			std::string ofile_map = fileUtils::fstream_init(outf_time, rhe_filepath, "", "_time_elapsed");
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
