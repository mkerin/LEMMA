//
// Created by kerin on 2019-02-26.
//
#include "pve.hpp"
#include "genotype_matrix.hpp"
#include "typedefs.hpp"
#include "file_utils.hpp"
#include "parameters.hpp"
#include "mpi_utils.hpp"

#include "tools/eigen3.3/Dense"

#include <random>

void PVE::run() {
	// Add intercept to covariates
	Eigen::MatrixXd ones = Eigen::MatrixXd::Constant(n_samples, 1, 1.0);
	if(n_covar > 0) {
		Eigen::MatrixXd C1(n_samples, n_covar + 1);
		C1 << C, ones;
		C = C1;
	} else {
		C = ones;
	}
	n_covar = C.cols();

	// Center and scale eta
	if(n_env > 0) {
		std::vector<std::string> placeholder = {"eta"};
		EigenUtils::center_matrix(eta);
		EigenUtils::scale_matrix_and_remove_constant_cols(eta, n_env, placeholder);
	}

	// Compute variance components
	initialise_components();
	if(n_env > 0) {
		std::cout << "G+GxE effects model (gaussian prior)" << std::endl;
		calc_RHE();
	} else {
		std::cout << "Main effects model (gaussian prior)" << std::endl;
		calc_RHE();
	}

	std::cout << "Variance components estimates" << std::endl;
	std::cout << sigmas << std::endl;

	process_jacknife_samples();
	std::cout << "PVE estimates" << std::endl;
	std::cout << h2 << std::endl;
}

void PVE::fill_gaussian_noise(unsigned int seed, Eigen::Ref<Eigen::MatrixXd> zz, long nn, long n_draws) {
	assert(zz.rows() == nn);
	assert(zz.cols() == n_draws);

	std::mt19937 generator{seed};
	std::normal_distribution<scalarData> noise_normal(0.0, 1);

	// for (int bb = 0; bb < pp; bb++) {
	//  for (std::size_t ii = 0; ii < nn; ii++) {
	//      zz(ii, bb) = noise_normal(generator);
	//  }
	// }

	// fill gaussian noise
	if(world_rank == 0) {
		std::vector<long> all_n_samples(world_size);
		for (const auto &kv : sample_location) {
			if (kv.second != -1) {
				all_n_samples[kv.second]++;
			}
		}

		std::vector<Eigen::MatrixXd> allzz(world_size);
		for (int ii = 0; ii < world_size; ii++) {
			allzz[ii].resize(all_n_samples[ii], n_draws);
		}

		for (int bb = 0; bb < n_draws; bb++) {
			std::vector<long> allii(world_size, 0);
			for (const auto &kv : sample_location) {
				if (kv.second != -1) {
					int local_rank = kv.second;
					long local_ii = allii[local_rank];
					allzz[local_rank](local_ii, bb) = noise_normal(generator);
					allii[local_rank]++;
				}
			}
		}

		for (int ii = 1; ii < world_size; ii++) {
//			allzz[ii].resize(all_n_samples[ii], n_draws);
			MPI_Send(allzz[ii].data(), allzz[ii].size(), MPI_DOUBLE, ii, 0, MPI_COMM_WORLD);
		}
		zz = allzz[0];
	} else {
		MPI_Recv(zz.data(), n_samples, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}

void PVE::calc_RHE() {
	// Compute randomised traces
	if(p.bgen_file != "NULL") {
		n_var = data.n_var;
		long n_main_segs;
		n_main_segs = (data.n_var + p.main_chunk_size - 1) / p.main_chunk_size;
		std::vector< std::vector <long> > main_fwd_pass_chunks(n_main_segs);
		for(long kk = 0; kk < data.n_var; kk++) {
			long main_ch_index = kk / p.main_chunk_size;
			main_fwd_pass_chunks[main_ch_index].push_back(kk);
		}

		EigenDataMatrix D;
		long jknf_block_size = (X.cumulative_pos[data.n_var - 1] + p.n_jacknife - 1) / p.n_jacknife;
		for (auto& iter_chunk : main_fwd_pass_chunks) {
			if(D.cols() != iter_chunk.size()) {
				D.resize(n_samples, iter_chunk.size());
			}
			X.col_block3(iter_chunk, D);

			// Get jacknife block (just use block assignment of 1st snp)
			long jacknife_index = X.cumulative_pos[iter_chunk[0]] / jknf_block_size;

			for (auto& comp : components) {
				comp.add_to_trace_estimator(D, jacknife_index);
			}
		}
	} else if (p.streamBgenFile != "NULL") {
		n_var = 0;
		Eigen::MatrixXd D;
		bool bgen_pass = true;
		long n_var_parsed = 0;
		long ch = 0;
		long print_interval = 100;
		if(p.mode_debug) print_interval = 1;
		long jack_block_size = (data.streamBgenView->number_of_variants() + p.n_jacknife) / p.n_jacknife;
		while (fileUtils::read_bgen_chunk(data.streamBgenView, D, sample_is_invalid,
		                                  n_samples, 128, p, bgen_pass, n_var_parsed)) {
			n_var += D.cols();
			if (ch % print_interval == 0 && ch > 0) {
				std::cout << "Chunk " << ch << " read (size " << 128;
				std::cout << ", " << n_var_parsed - 1 << "/" << data.streamBgenView->number_of_variants();
				std::cout << " variants parsed)" << std::endl;
			}

			// Get jacknife block (just use block assignment of 1st snp)
			long jacknife_index = n_var_parsed / jack_block_size;

			long n_chunk = D.cols();
			std::vector<std::string> placeholder(n_chunk, "col");
			EigenUtils::center_matrix(D);
			EigenUtils::scale_matrix_and_remove_constant_cols(D, n_chunk, placeholder);

			for (auto &comp : components) {
				comp.add_to_trace_estimator(D, jacknife_index);
			}
			ch++;
		}
		if(p.verbose) std::cout << n_var << " variants pass QC filters" << std::endl;
	}
	for (long ii = 0; ii < n_components; ii++) {
		components[ii].finalise();
	}

	// Solve system to estimate sigmas
	long n_components = components.size();
	for (long ii = 0; ii < n_components; ii++) {
		components[ii].rm_jacknife_block = -1;
	}
	Eigen::MatrixXd CC = construct_vc_system(components);
	Eigen::MatrixXd A = CC.block(0, 0, n_components, n_components);
	Eigen::VectorXd bb = CC.col(n_components);

	std::cout << "A: " << std::endl << A << std::endl;
	std::cout << "b: " << std::endl << bb << std::endl;
	sigmas = A.colPivHouseholderQr().solve(bb);
	h2 = calc_h2(A, bb, false);
	h2b = calc_h2(A, bb, true);

	boost_io::filtering_ostream outf;
	if(p.mode_debug) {
		auto filename = fileUtils::fstream_init(outf, p.out_file, "", "_rhe_debug");
		std::cout << "Writing RHE debugging info to " << filename << std::endl;
		Eigen::VectorXd tmp(Eigen::Map<Eigen::VectorXd>(CC.data(),CC.cols()*CC.rows()));
		outf << -1 << " " << tmp.transpose() << std::endl;
	}

	// jacknife estimates
	std::cout << "Computing standard errors using " << p.n_jacknife << " jacknife blocks" << std::endl;
	sigmas_jack.resize(p.n_jacknife, n_components);
	h2_jack.resize(p.n_jacknife, n_components);
	h2b_jack.resize(p.n_jacknife, n_components);
	n_var_jack.resize(p.n_jacknife);

	for (long jj = 0; jj < p.n_jacknife; jj++) {
		for (long ii = 0; ii < n_components; ii++) {
			components[ii].rm_jacknife_block = jj;
		}
		n_var_jack[jj] = components[0].get_n_var_local();

		Eigen::MatrixXd CC = construct_vc_system(components);
		Eigen::MatrixXd AA = CC.block(0, 0, n_components, n_components);
		Eigen::VectorXd bb = CC.col(n_components);
		Eigen::VectorXd ss = AA.colPivHouseholderQr().solve(bb);
		sigmas_jack.row(jj) = ss;
		h2b_jack.row(jj) = calc_h2(AA, bb, true);
		h2_jack.row(jj) = calc_h2(AA, bb, false);

		if(p.mode_debug) {
			Eigen::VectorXd tmp(Eigen::Map<Eigen::VectorXd>(CC.data(),CC.cols()*CC.rows()));
			outf << jj << " " << tmp.transpose() << std::endl;
		}
	}
	for (long ii = 0; ii < n_components; ii++) {
		components[ii].rm_jacknife_block = -1;
	}
	if(p.mode_debug) {
		boost_io::close(outf);
	}

	if(n_env > 0) {
		// Main effects model
		Eigen::MatrixXd A1(2, 2);
		Eigen::VectorXd bb1(2);
		A1(0, 0) = A(ind.main, ind.main);
		A1(0, 1) = A(ind.main, ind.noise);
		A1(1, 0) = A(ind.noise, ind.main);
		A1(1, 1) = A(ind.noise, ind.noise);

		bb1 << bb(ind.main), bb(ind.noise);
		Eigen::VectorXd sigmas1 = A1.colPivHouseholderQr().solve(bb1);
		Eigen::VectorXd h2_1 = sigmas1 / sigmas1.sum();
		std::cout << "h2-G = " << h2_1(0, 0) << " (main effects model only)" << std::endl;
	}
}

Eigen::MatrixXd PVE::construct_vc_system(const std::vector<PVE_Component> &components) {
	Eigen::MatrixXd res(n_components, n_components + 1);
	for (long ii = 0; ii < n_components; ii++) {
		res(ii, n_components) = components[ii].get_bb_trace();
		for (long jj = 0; jj <= ii; jj++) {
			if(ii == jj && components[ii].label == "noise") {
				res(ii, jj) = Nglobal - n_covar;
			} else {
				res(ii, jj) = components[ii] * components[jj];
				res(ii, jj) = mpiUtils::mpiReduce_inplace(res(ii, jj));
				res(jj, ii) = res(ii, jj);
			}
		}
	}
	return res;
}

Eigen::ArrayXd PVE::calc_h2(Eigen::Ref<Eigen::MatrixXd> AA, Eigen::Ref<Eigen::VectorXd> bb, const bool &reweight_sigmas) {
	Eigen::ArrayXd ss = AA.colPivHouseholderQr().solve(bb);
	if(reweight_sigmas) {
		ss *= (AA.row(AA.rows()-1)).array() / n_samples;
	}
	return ss / ss.sum();
}

void PVE::process_jacknife_samples() {
	// Rescale h2 to avoid bias
	for (long ii = 0; ii < n_components - 1; ii++) {
		h2_jack.col(ii) *= n_var / n_var_jack;
		h2b_jack.col(ii) *= n_var / n_var_jack;
	}

	// SE of h2
	h2_se_jack.resize(n_components);
	h2b_se_jack.resize(n_components);
	for (long ii = 0; ii < n_components; ii++) {
		h2_se_jack[ii] = std::sqrt(get_jacknife_var(h2_jack.col(ii)));
		h2b_se_jack[ii] = std::sqrt(get_jacknife_var(h2b_jack.col(ii)));
	}

	// bias correction
	h2_bias_corrected.resize(n_components);
	h2b_bias_corrected.resize(n_components);
	for (long ii = 0; ii < n_components; ii++) {
		h2_bias_corrected[ii] = get_jacknife_bias_correct(h2_jack.col(ii), h2(ii));
		h2b_bias_corrected[ii] = get_jacknife_bias_correct(h2b_jack.col(ii), h2b(ii));
	}
}

Eigen::MatrixXd PVE::project_out_covars(Eigen::Ref<Eigen::MatrixXd> rhs) {
	if(n_covar > 0) {
		if (CtC_inv.rows() != n_covar) {
			if(p.mode_debug) std::cout << "Starting compute of CtC_inv" << std::endl;
			Eigen::MatrixXd CtC = C.transpose() * C;
			CtC = mpiUtils::mpiReduce_inplace(CtC);
			CtC_inv = CtC.inverse();
			if(p.mode_debug) std::cout << "Ending compute of CtC_inv" << std::endl;
		}
		return EigenUtils::project_out_covars(rhs, C, CtC_inv, p.mode_debug);
	} else {
		return rhs;
	}
}

void PVE::to_file(const std::string &file) {
	boost_io::filtering_ostream outf;
	std::string suffix = "";
	if(p.mode_vb || p.mode_calc_snpstats) {
		suffix = "_pve";
	}
	auto filename = fileUtils::fstream_init(outf, file, "", suffix);

	std::cout << "Writing PVE results to " << filename << std::endl;
	outf << "component sigmas h2 h2_se h2_bias_corrected" << std::endl;

	for (int ii = 0; ii < n_components; ii++) {
		outf << components[ii].label << " ";
		outf << sigmas[ii] << " ";
		outf << h2[ii] << " ";
		outf << h2_se_jack[ii] << " ";
		outf << h2_bias_corrected[ii] << std::endl;
	}

	for (int ii = 0; ii < n_components; ii++) {
		outf << components[ii].label << "_v2 ";
		outf << sigmas[ii] << " ";
		outf << h2b[ii] << " ";
		outf << h2b_se_jack[ii] << " ";
		outf << h2b_bias_corrected[ii] << std::endl;
	}
	boost_io::close(outf);

	if(p.xtra_verbose) {
		auto filename = fileUtils::fstream_init(outf, file, "", suffix + "_jacknife");
		std::cout << "Writing jacknife estimates to " << filename << std::endl;

		outf << "n_jack";
		for (long ii = 0; ii < n_components; ii++) {
			outf << " " << components[ii].label;
		}
		outf << std::endl;
		for (long jj = 0; jj < p.n_jacknife; jj++) {
			outf << components[0].n_vars_local[jj];
			for (long ii = 0; ii < n_components; ii++) {
				outf << " " << h2_jack(jj, ii);
			}
			outf << std::endl;
		}
		boost_io::close(outf);
	}

	if(p.xtra_verbose) {
		auto filename = fileUtils::fstream_init(outf, file, "", suffix + "_jacknife_scaled");
		std::cout << "Writing rescaled jacknife estimates to " << filename << std::endl;

		outf << "n_jack";
		for (long ii = 0; ii < n_components; ii++) {
			outf << " " << components[ii].label;
		}
		outf << std::endl;
		for (long jj = 0; jj < p.n_jacknife; jj++) {
			outf << components[0].n_vars_local[jj];
			for (long ii = 0; ii < n_components; ii++) {
				outf << " " << h2b_jack(jj, ii);
			}
			outf << std::endl;
		}
		boost_io::close(outf);
	}
}

void PVE::initialise_components() {
	zz.resize(n_samples, n_draws);
	if(p.rhe_random_vectors_file != "NULL") {
		EigenUtils::read_matrix(p.rhe_random_vectors_file, zz);
	} else {
		fill_gaussian_noise(p.random_seed, zz, n_samples, n_draws);
	}

	std::cout << "Initialising HE-regression components with:" << std::endl;
	std::cout << " - N-jacknife = " << p.n_jacknife << std::endl;
	std::cout << " - N-draws = " << p.n_pve_samples << std::endl;
	std::cout << " - N-samples = " << (long) Nglobal << std::endl;
	std::cout << " - N-covars = " << n_covar << std::endl;

	if(n_covar > 0) {
		Wzz = project_out_covars(zz);
		Y = project_out_covars(Y);
	} else {
		Wzz = zz;
	}

	// Set variance components
	if(true) {
		PVE_Component comp(p, Y, zz, Wzz, C, CtC_inv, p.n_jacknife);
		comp.label = "G";
		components.push_back(comp);
	}

	if(n_env == 1) {
		PVE_Component comp(p, Y, zz, Wzz, C, CtC_inv, p.n_jacknife);
		comp.label = "GxE";
		comp.set_eta(eta);
		components.push_back(comp);

		ind.main = 0;
		ind.gxe = 1;
		ind.noise = 2;
	} else {
		ind.main = 0;
		ind.noise = 1;
	}

	if(true) {
		PVE_Component comp(p, Y, zz, Wzz, C, CtC_inv, p.n_jacknife);
		comp.set_inactive();
		comp.label = "noise";
		components.push_back(comp);
	}
	n_components = components.size();

	std::cout << " - N-components = " << n_components - 1 << std::endl;
#ifndef OSX
	std::cout << "Initialised with ";
	std::cout << (double) fileUtils::getValueRAM() / 1000 / 1000;
	std::cout << "GB of RAM" << std::endl;
#endif
}

double PVE::get_jacknife_var(Eigen::ArrayXd jack_estimates) {
	double jack_var = (jack_estimates - jack_estimates.mean()).square().sum();
	jack_var *= (p.n_jacknife - 1.0) / p.n_jacknife;
	return jack_var;
}

double PVE::get_jacknife_bias_correct(Eigen::ArrayXd jack_estimates, double full_data_est) {
	double res = p.n_jacknife * full_data_est - (p.n_jacknife - 1.0) * jack_estimates.mean();
	return res;
}
