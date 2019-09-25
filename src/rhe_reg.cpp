//
// Created by kerin on 2019-02-26.
//
#include "rhe_reg.hpp"
#include "genotype_matrix.hpp"
#include "typedefs.hpp"
#include "file_utils.hpp"
#include "parameters.hpp"
#include "mpi_utils.hpp"
#include "nelder_mead.hpp"
#include "levenberg_marquardt.hpp"

#include "tools/eigen3.3/Dense"

#include <random>
#include <cmath>
#include <functional>
#include <limits>

void RHEreg::run() {
	standardise_non_genetic_data();

	// Parse RHE groups files once to get total number of groups
	if(p.RHE_multicomponent) {
		std::set<std::string> tmp_set;
		for (auto ff : p.RHE_groups_files) {
			read_RHE_groups(ff);
			tmp_set.insert(SNPGROUPS_group.begin(), SNPGROUPS_group.end());
		}
		std::vector<std::string> tmp_vec(tmp_set.begin(), tmp_set.end());
		all_SNPGROUPS = tmp_vec;
		if(p.RHE_groups_files.size() > 1) {
			SNPGROUPS_snpid.clear();
			SNPGROUPS_group.clear();
		}
	}

	// Compute variance components
	initialise_components();
	auto start = std::chrono::system_clock::now();
	compute_RHE_trace_operators();
	auto end = std::chrono::system_clock::now();
	elapsed_streamBgen = end - start;

	start = std::chrono::system_clock::now();
	if(p.mode_RHEreg_NM) {
		nls_env_weights = run_RHE_nelderMead();
	} else if(p.mode_RHEreg_LM) {
		nls_env_weights = run_RHE_levenburgMarquardt();
	} else if(n_env > 0) {
		std::cout << "G+GxE effects model (gaussian prior)" << std::endl;
		solve_RHE(components);
	} else {
		std::cout << "Main effects model (gaussian prior)" << std::endl;
		solve_RHE(components);
	}
	end = std::chrono::system_clock::now();
	elapsed_solveRHE = end - start;

	start = std::chrono::system_clock::now();
	process_jacknife_samples();
	end = std::chrono::system_clock::now();
	elapsed_solveJacknife = end - start;

	std::cout << "PVE estimates" << std::endl;
	std::cout << h2 << std::endl;
}

void RHEreg::standardise_non_genetic_data(){
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
	if(n_env > 0 && !p.use_raw_env) {
		EigenUtils::center_matrix(E);
		EigenUtils::scale_matrix_and_remove_constant_cols(E, n_env, env_names);
	}
}

void RHEreg::initialise_components() {

	zz.resize(n_samples, n_draws);
	if(p.rhe_random_vectors_file != "NULL") {
		EigenUtils::read_matrix(p.rhe_random_vectors_file, zz);
	} else {
		fill_gaussian_noise(p.random_seed, zz, n_samples, n_draws);
	}

	if(p.mode_debug) {
		std::string ram = mpiUtils::currentUsageRAM();
		std::cout << "Before initialising " << ram << std::endl;
	}

	std::cout << "Initialising HE-regression components with:" << std::endl;
	std::cout << " - N-jacknife = " << p.n_jacknife << std::endl;
	std::cout << " - N-draws = " << p.n_pve_samples << std::endl;
	std::cout << " - N-samples = " << (long) Nglobal << std::endl;
	std::cout << " - N-covars = " << n_covar << std::endl;
	std::cout << " - N-env = " << n_env << std::endl;
	if(p.RHE_multicomponent) {
		std::cout << " - N-annotations = " << all_SNPGROUPS.size() << std::endl;
	}

	if(n_covar > 0) {
		zz = project_out_covars(zz);
		Y = project_out_covars(Y);
	}

	// Set variance components
	components.reserve(1 + all_SNPGROUPS.size() * (n_env + 1));
	if(true) {
		if(p.RHE_multicomponent) {
			for (auto group : all_SNPGROUPS) {
				RHEreg_Component comp(p, Y, zz, C, CtC_inv, p.n_jacknife);
				comp.label = group + "_G";
				comp.effect_type = "G";
				comp.group = group;
				components.push_back(std::move(comp));

				if(p.mode_debug) {
					std::string ram = mpiUtils::currentUsageRAM();
					std::cout << "(" << ram << ")" << std::endl;
				}
			}
		} else {
			RHEreg_Component comp(p, Y, zz, C, CtC_inv, p.n_jacknife);
			comp.label = "G";
			comp.effect_type = "G";

			components.push_back(std::move(comp));
		}
	}

	if(p.xtra_verbose) {
		std::string ram = mpiUtils::currentUsageRAM();
		std::cout << "Initialised main effects RHEreg components (" << ram << ")" << std::endl;
	}

	for (long ee = 0; ee < n_env; ee++) {
		if(p.RHE_multicomponent) {
			assert(n_env == 1);
			for (auto group : all_SNPGROUPS) {
				RHEreg_Component comp(p, Y, zz, C, CtC_inv, p.n_jacknife);
				comp.label = group + "_GxE";
				comp.effect_type = "GxE";
				comp.group = group;
				comp.env_var_index = ee;
				comp.set_env_var(E.col(ee));
				components.push_back(std::move(comp));

				if(p.mode_debug) {
					std::string ram = mpiUtils::currentUsageRAM();
					std::cout << "(" << ram << ")" << std::endl;
				}
			}
		} else {
			RHEreg_Component comp(p, Y, zz, C, CtC_inv, p.n_jacknife);
			if(n_env == 1){
				comp.label = "GxE";
			} else {
				comp.label = "GxE_" + env_names[ee];
			}
			comp.effect_type = "GxE";
			comp.env_var_index = ee;
			comp.set_env_var(E.col(ee));
			components.push_back(std::move(comp));
		}
	}

	if(true) {
		RHEreg_Component comp(p, Y, zz, C, CtC_inv, p.n_jacknife);
		comp.set_inactive();
		comp.label = "noise";
		comp.effect_type = "noise";
		components.push_back(std::move(comp));
	}
	n_components = components.size();

	std::cout << " - N-RHEreg-components = " << n_components - 1 << std::endl;
	std::string ram = mpiUtils::currentUsageRAM();
	std::cout << "Initialised RHEreg (" << ram << ")" << std::endl;
}

void RHEreg::compute_RHE_trace_operators() {
	if(p.mode_RHEreg_NM || p.mode_RHEreg_LM) {
		ytEXXtEys.resize(p.n_jacknife);
		for (long jj = 0; jj < p.n_jacknife; jj++) {
			ytEXXtEys[jj] = Eigen::MatrixXd::Zero(n_env, n_env);
		}
	}

	// Compute randomised traces
	if (p.bgen_file != "NULL") {
		// p.bgen_file already read into RAM
		n_var = data.n_var;
		long n_main_segs;
		n_main_segs = (data.n_var + p.main_chunk_size - 1) / p.main_chunk_size;
		std::vector<std::vector<long> > main_fwd_pass_chunks(n_main_segs);
		for (long kk = 0; kk < data.n_var; kk++) {
			long main_ch_index = kk / p.main_chunk_size;
			main_fwd_pass_chunks[main_ch_index].push_back(kk);
		}

		EigenDataMatrix D;
		long jknf_block_size = (X.cumulative_pos[data.n_var - 1] + p.n_jacknife - 1) / p.n_jacknife;
		for (auto &iter_chunk : main_fwd_pass_chunks) {
			if (D.cols() != iter_chunk.size()) {
				D.resize(n_samples, iter_chunk.size());
			}
			X.col_block3(iter_chunk, D);

			// Get jacknife block (just use block assignment of 1st snp)
			long jacknife_index = X.cumulative_pos[iter_chunk[0]] / jknf_block_size;

			for (auto &comp : components) {
				comp.add_to_trace_estimator(D, jacknife_index);
			}

			if(p.mode_RHEreg_NM || p.mode_RHEreg_LM) {
				Eigen::MatrixXd XtEy(D.cols(), n_env);
				for (long ll = 0; ll < n_env; ll++) {
					XtEy.col(ll) = D.transpose() * E.col(ll).asDiagonal() * Y;
				}
				XtEy = mpiUtils::mpiReduce_inplace(XtEy);

				for (long ll = 0; ll < n_env; ll++) {
					for (long mm = 0; mm <= ll; mm++) {
						ytEXXtEys[jacknife_index](mm, ll) += XtEy.col(ll).dot(XtEy.col(mm));
						ytEXXtEys[jacknife_index](ll, mm) = ytEXXtEys[jacknife_index](mm, ll);
					}
				}
			}
		}
	} else if (!p.streamBgenFiles.empty()) {
		n_var = 0;
		long long snp_group_index = 0;
		long n_var_parsed = 0;
		long ch = 0;
		long print_interval = p.streamBgen_print_interval;;
		if (p.mode_debug) print_interval = 1;
		long long n_find_operations = 0;
		std::vector<std::string> SNPIDS;
		long long n_vars_tot = 0;
		for (int ii = 0; ii < p.streamBgenFiles.size(); ii++) {
			n_vars_tot += data.streamBgenViews[ii]->number_of_variants();
		}
		long jack_block_size = (n_vars_tot + p.n_jacknife) / p.n_jacknife;
		std::cout << "jacknife block size = " << jack_block_size << std::endl;
		for (int ii = 0; ii < p.streamBgenFiles.size(); ii++) {
			std::cout << std::endl << "Streaming genotypes from " << p.streamBgenFiles[ii] << std::endl;

			if (p.RHE_groups_files.size() > 1) {
				snp_group_index = 0;
				read_RHE_groups(p.RHE_groups_files[ii]);
			}

			Eigen::MatrixXd D;
			bool bgen_pass = true;
			while (fileUtils::read_bgen_chunk(data.streamBgenViews[ii], D, sample_is_invalid,
			                                  n_samples, 256, p, bgen_pass, n_var_parsed, SNPIDS)) {
				n_var += D.cols();
				if (ch % print_interval == 0 && ch > 0) {
					std::cout << "Chunk " << ch << " read (size " << 256;
					std::cout << ", " << n_var_parsed - 1 << "/" << n_vars_tot;
					std::cout << " variants parsed)" << std::endl;
				}

				// Get jacknife block (just use block assignment of 1st snp)
				long jacknife_index = n_var_parsed / jack_block_size;

				long n_chunk = D.cols();
				std::vector<std::string> placeholder(n_chunk, "col");
				if (!p.mode_RHE_fast) {
					EigenUtils::center_matrix(D);
					EigenUtils::scale_matrix_and_remove_constant_cols(D, n_chunk, placeholder);
				}

				// parse which snp belongs to which group
				std::vector<std::vector<int> > block_membership(n_components);
				if (p.RHE_multicomponent) {
					for (long jj = 0; jj < D.cols(); jj++) {
						if (SNPGROUPS_snpid[snp_group_index] == SNPIDS[jj]) {
							for (int kk = 0; kk < n_components; kk++) {
								if (components[kk].group == SNPGROUPS_group[snp_group_index]) {
									block_membership[kk].push_back(jj);
								}
							}
						} else {
							// find
							auto it = std::find(SNPGROUPS_snpid.begin(), SNPGROUPS_snpid.end(), SNPIDS[jj]);
							if (it == SNPGROUPS_snpid.end()) {
								// skip
								n_var -= 1;
							} else {
								snp_group_index = it - SNPGROUPS_snpid.begin();
								for (int kk = 0; kk < n_components; kk++) {
									if (components[kk].group == SNPGROUPS_group[snp_group_index]) {
										block_membership[kk].push_back(jj);
									}
								}
							}
							n_find_operations++;
						}
						snp_group_index++;
					}
					for (int cc = 0; cc < n_components; cc++) {
						Eigen::MatrixXd D1(D.rows(), block_membership[cc].size());
						if (D1.cols() > 0) {
							for (int jjj = 0; jjj < block_membership[cc].size(); jjj++) {
								D1.col(jjj) = D.col(block_membership[cc][jjj]);
							}
							components[cc].add_to_trace_estimator(D1, jacknife_index);
						}
					}
				} else {
					for (auto &comp : components) {
						comp.add_to_trace_estimator(D, jacknife_index);
					}

					if(p.mode_RHEreg_NM ||  p.mode_RHEreg_LM) {
						Eigen::MatrixXd XtEy(D.cols(), n_env);
						for (long ll = 0; ll < n_env; ll++) {
							XtEy.col(ll) = D.transpose() * E.col(ll).asDiagonal() * Y;
						}
						XtEy = mpiUtils::mpiReduce_inplace(XtEy);

						for (long ll = 0; ll < n_env; ll++) {
							for (long mm = 0; mm <= ll; mm++) {
								ytEXXtEys[jacknife_index](mm, ll) += XtEy.col(ll).dot(XtEy.col(mm));
								ytEXXtEys[jacknife_index](ll, mm) = ytEXXtEys[jacknife_index](mm, ll);
							}
						}
					}
				}
				ch++;
			}
		}
		if (p.verbose) std::cout << n_var << " variants pass QC filters" << std::endl;
		if (p.xtra_verbose && p.RHE_multicomponent) {
			std::cout << n_find_operations << " find operations performed" << std::endl;
		}
	}
	for (long ii = 0; ii < n_components; ii++) {
		components[ii].finalise();
	}
}

void RHEreg::solve_RHE(std::vector<RHEreg_Component>& components) {
	boost_io::filtering_ostream outf;

	// Solve system to estimate sigmas
	n_components = components.size();
	for (long ii = 0; ii < n_components; ii++) {
		components[ii].rm_jacknife_block = -1;
	}
	Eigen::MatrixXd CC = construct_vc_system(components);
	Eigen::MatrixXd A = CC.block(0, 0, n_components, n_components);
	Eigen::VectorXd bb = CC.col(n_components);

	if(world_rank == 0) {
		auto filename = fileUtils::fstream_init(outf, p.out_file, "", "_linSystem_dump");
		std::cout << "Dumping RHEreg linear system to " << filename << std::endl;
		outf << CC << std::endl;
		boost_io::close(outf);
	}
	if(!p.RHE_multicomponent) {
		std::cout << "A: " << std::endl << A << std::endl;
		std::cout << "b: " << std::endl << bb << std::endl;
	}
	sigmas = A.colPivHouseholderQr().solve(bb);
	h2 = calc_h2(A, bb, true);

	if(p.RHE_multicomponent) {
		double h2_G_tot = 0, h2_GxE_tot = 0;
		for (int cc = 0; cc < n_components; cc++) {
			if(components[cc].effect_type == "G") {
				h2_G_tot += h2[cc];
			} else if (components[cc].effect_type == "GxE") {
				h2_GxE_tot += h2[cc];
			}
		}

		std::cout << "Additive and multiplicative interaction h2 estimates:" << std::endl;
		std::cout << "h2_G = " << h2_G_tot << std::endl;
		if(n_env > 0) {
			std::cout << "h2_GxE = " << h2_GxE_tot << std::endl;
		}
	} else {
		std::cout << "Heritability estimates:" << std::endl;
		std::cout << h2 << std::endl;
	}

	// Write to file
	if(world_rank == 0 && p.xtra_verbose) {
		auto filename = fileUtils::fstream_init(outf, p.out_file, "", "_h2_dump");
		std::cout << "Dumping Heritability to " << filename << std::endl;
		Eigen::VectorXd tmp(Eigen::Map<Eigen::VectorXd>(CC.data(),CC.cols()*CC.rows()));
		outf << "Component h2 n_var" << std::endl;
		for (long ii = 0; ii < n_components; ii++) {
			outf << components[ii].label << " " << h2[ii];
			outf << " " << components[ii].n_var_local << std::endl;
		}
		boost_io::close(outf);
	}

	// jacknife estimates
	if(p.n_jacknife > 1) {
		std::cout << "Computing standard errors using " << p.n_jacknife << " jacknife blocks" << std::endl;
		sigmas_jack.resize(p.n_jacknife, n_components);

		// h2_jack contains total G and GxE heritabilities at end
		h2_jack.resize(p.n_jacknife, n_components);
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
			h2_jack.row(jj) = calc_h2(AA, bb, true);

			if(p.xtra_verbose && jj % 10 == 0 && jj > 0) {
				std::cout << "Computed " << jj << " of ";
				std::cout << p.n_jacknife << " jacknife blocks" << std::endl;
			}
		}
		for (long ii = 0; ii < n_components; ii++) {
			components[ii].rm_jacknife_block = -1;
		}
		if (p.mode_debug) {
			boost_io::close(outf);
		}
	}

	if(n_env > 0) {
		// Main effects model
		int ind_main, ind_noise;
		for (int cc = 0; cc < n_components; cc++) {
			if (components[cc].effect_type == "G") {
				ind_main = cc;
			} else if (components[cc].effect_type == "noise") {
				ind_noise = cc;
			}
		}
		Eigen::MatrixXd A1(2, 2);
		Eigen::VectorXd bb1(2);
		A1(0, 0) = A(ind_main, ind_main);
		A1(0, 1) = A(ind_main, ind_noise);
		A1(1, 0) = A(ind_noise, ind_main);
		A1(1, 1) = A(ind_noise, ind_noise);

		bb1 << bb(ind_main), bb(ind_noise);
		Eigen::VectorXd sigmas1 = A1.colPivHouseholderQr().solve(bb1);
		Eigen::VectorXd h2_1 = sigmas1 / sigmas1.sum();
		std::cout << "h2-G = " << h2_1(0, 0) << " (main effects model only)" << std::endl;
	}
}

Eigen::VectorXd RHEreg::run_RHE_levenburgMarquardt() {
	std::cout << std::endl << "Optimising env-weights with Levenburg-Marquardt" << std::endl;

	Eigen::VectorXd env_weights;
	double best_rss = std::numeric_limits<double>::max();
	std::mt19937 generator{p.random_seed};
	std::normal_distribution<scalarData> noise_normal(0.0, 1.0 / n_env / n_env);
	for (long ii = 0; ii < p.n_LM_starts; ii++) {
		std::cout << "Computing LM " << ii << " of " << p.n_LM_starts << std::endl;
		Eigen::VectorXd tmp_env_weights;
		double tmp_rss;
		LevenbergMarquardt LM(p, components, Y, E, C, CtC_inv, ytEXXtEys, env_names);
		if(ii == 0) {
			Eigen::VectorXd params = LM.runLM();
			tmp_env_weights = params.segment(1, n_env);
			tmp_rss = LM.getete();
		} else {
			Eigen::VectorXd env_noise(n_env);
			for (long ll = 0; ll < n_env; ll++) {
				env_noise(ll) = noise_normal(generator);
			}

			Eigen::VectorXd params = LM.runLM(env_noise);
			tmp_env_weights = params.segment(1, n_env);
			tmp_rss = LM.getete();
		}

		if(tmp_rss < best_rss) {
			std::cout << "Updating best LM; SumOfSquares = " << tmp_rss;
			std::cout << " after " << LM.count << " iterations" << std::endl;
			best_rss = tmp_rss;
			env_weights = tmp_env_weights;
			LM.push_interim_updates();
		}
	}

	// Rescale env weights
	Eigen::ArrayXd eta = E * env_weights;
	double mean, sd;
	mean = mpiUtils::mpiReduce_inplace(eta.sum()) / Nglobal;
	sd = mpiUtils::mpiReduce_inplace((eta - mean).square().sum());
	sd /= (Nglobal - 1.0);
	sd = std::sqrt(sd);
	env_weights /= sd;
	if(p.mode_debug) {
		std::cout << "Eta-mean = " << mean << std::endl;
		std::cout << "Eta-SD = " << sd << std::endl;
	}

	// Dump env weights to file
	boost_io::filtering_ostream outf;
	auto filename = fileUtils::fstream_init(outf, p.out_file, "", "_LM_env_weights");
	std::cout << "Writing env weights to " << filename << std::endl;
	std::vector<std::string> header = {"env", "mu"};
	EigenUtils::write_matrix(outf, env_weights, header, env_names);

	if(p.xtra_verbose) {
		if (n_env > 0) {
			std::string path = fileUtils::filepath_format(p.out_file, "", "_LM_converged_eta");
			std::cout << "Writing eta to file: " << path << std::endl;
			fileUtils::dump_predicted_vec_to_file(eta, path, "eta", sample_location);
		}
	}

	// Use weights to collapse GxE component and continue with rest of algorithm
	std::vector<RHEreg_Component> my_components;
	get_GxE_collapsed_system(components, my_components, E, env_weights, ytEXXtEys, true);

	// Solve system to estimate sigmas
	solve_RHE(my_components);

	// WARNING: overwriting original components
	// Not possible to use mutliple methods in same run with this
	components = my_components;

	return env_weights;
}

Eigen::VectorXd RHEreg::run_RHE_nelderMead() {
	std::cout << std::endl << "Optimising env-weights with Nelder Mead" << std::endl;
	boost_io::filtering_ostream outf_env, outf_obj;
	auto filename_env = fileUtils::fstream_init(outf_env, p.out_file, ".lemma_files/", "_nm_env_iter");
	auto filename_obj = fileUtils::fstream_init(outf_obj, p.out_file, ".lemma_files/", "_nm_obj_iter");
	outf_env << "count ";
	for (long ll = 0; ll < n_env; ll++) {
		outf_env << env_names[ll];
		if (ll < n_env - 1) {
			outf_env << " ";
		}
	}
	outf_env << std::endl;
	outf_obj << "count value" << std::endl;

	/* Nelder Mead */
	Eigen::VectorXd env_weights = Eigen::VectorXd::Constant(n_env, 1.0);
	env_weights *= 1.0 / n_env;

	// Create simplex
	const long n_vals = env_weights.rows();
	Eigen::VectorXd simplex_fn_vals(n_vals+1);
	Eigen::MatrixXd simplex_points(n_vals+1,n_vals);
	setupNelderMead(env_weights,
	                std::bind(&RHEreg::RHE_nelderMead_obj, this, std::placeholders::_1, std::placeholders::_2),
	                simplex_points, simplex_fn_vals);

	// Run algorithm
	long iter = 0;
	const double err_tol = 1E-08;
	double err = 2*err_tol, min_val = simplex_fn_vals.minCoeff();


	while (err > err_tol && iter < p.nelderMead_max_iter) {
		if(iter % 10 == 0) {
			std::cout << "Starting NM iteration " << iter << ", best SumOfSquares = " << min_val << std::endl;
			// std::cout << "Env-weights: " << env_weights.transpose() << std::endl;
		}
		outf_env << iter << " ";
		for (long ll = 0; ll < n_env; ll++) {
			outf_env << env_weights[ll];
			if (ll < n_env - 1) {
				outf_env << " ";
			}
		}
		outf_env << std::endl;
		outf_obj << iter << " " << min_val << std::endl;

		iterNelderMead(simplex_points, simplex_fn_vals,
		               std::bind(&RHEreg::RHE_nelderMead_obj, this, std::placeholders::_1, std::placeholders::_2),
		               p);

		// Update function values
		for (size_t i=0; i < n_vals + 1; i++) {
			simplex_fn_vals(i) = RHE_nelderMead_obj(simplex_points.row(i).transpose(), nullptr);
		}

		err = std::abs(min_val - simplex_fn_vals.maxCoeff());
		min_val = simplex_fn_vals.minCoeff();

		long index_min = get_index_min(simplex_fn_vals);
		env_weights = simplex_points.row(index_min);
		iter++;
	}
	std::cout << "Completed NM iteration " << iter << ", best SumOfSquares = " << min_val << std::endl << std::endl;
	outf_env << iter << " " << env_weights.transpose() << std::endl;
	outf_obj << iter << " " << min_val << std::endl;

	// Dump env weights to file
	boost_io::filtering_ostream outf;
	auto filename = fileUtils::fstream_init(outf, p.out_file, "", "_NM_env_weights");
	std::cout << "Writing env weights to " << filename << std::endl;
	std::vector<std::string> header = {"env", "mu"};
	EigenUtils::write_matrix(outf, env_weights, header, env_names);

	// Use weights to collapse GxE component and continue with rest of algorithm
	Eigen::MatrixXd placeholder = Eigen::MatrixXd::Zero(n_samples, n_draws);
	RHEreg_Component combined_comp(p, Y, C, CtC_inv, placeholder);
	get_GxE_collapsed_component(components, combined_comp, E, env_weights, ytEXXtEys);

	// Delete old
	std::vector<long> to_erase;
	for (long ii = n_components - 1; ii>= 0; ii--) {
		if (components[ii].effect_type == "GxE") {
			components.erase(components.begin() + ii);
		}
	}
	components.insert(components.begin() + 1, std::move(combined_comp));
	n_components = components.size();

	// Solve system to estimate sigmas
	solve_RHE(components);

	return env_weights;
}

double RHEreg::RHE_nelderMead_obj(Eigen::VectorXd env_weights, void *grad_out) const {
	// Take in env_weights
	// Solve to find best sigma's
	// Give back solution.

	std::vector<RHEreg_Component> my_components;
	my_components.reserve(3);

	// Get main and noise components
	for (int ii = 0; ii < n_components; ii++) {
		if(components[ii].effect_type == "G") {
			my_components.push_back(components[ii]);
		} else if (components[ii].effect_type == "noise") {
			my_components.push_back(components[ii]);
		}
	}
	assert(my_components.size() == 2);

	// Create component for \diag{Ew} \left( \sum_l w_l \bm{v}_{b, l} \right)
	Eigen::MatrixXd placeholder = Eigen::MatrixXd::Zero(n_samples, n_draws);
	RHEreg_Component combined_comp(p, Y, C, CtC_inv, placeholder);
	get_GxE_collapsed_component(components, combined_comp, E, env_weights, ytEXXtEys);

	my_components.push_back(std::move(combined_comp));

	// Solve
	long n_components = my_components.size();
	Eigen::MatrixXd CC = construct_vc_system(my_components);
	Eigen::MatrixXd AA = CC.block(0, 0, n_components, n_components);
	Eigen::VectorXd bb = CC.col(n_components);
	Eigen::VectorXd sigmas = AA.colPivHouseholderQr().solve(bb);

	double obj = std::pow(Y.squaredNorm(), 2) -2 * sigmas.dot(bb) + sigmas.dot(AA * sigmas);
	return obj;
}

Eigen::MatrixXd RHEreg::construct_vc_system(const std::vector<RHEreg_Component> &vec_of_components) const {
	long n_components = vec_of_components.size();
	Eigen::MatrixXd res(n_components, n_components + 1);
	for (long ii = 0; ii < n_components; ii++) {
		res(ii, n_components) = vec_of_components[ii].get_bb_trace();
		for (long jj = 0; jj <= ii; jj++) {
			if(ii == jj && vec_of_components[ii].label == "noise") {
				res(ii, jj) = Nglobal - n_covar;
			} else {
				res(ii, jj) = vec_of_components[ii] * vec_of_components[jj];
				res(jj, ii) = res(ii, jj);
			}
		}
	}
	return res;
}

Eigen::ArrayXd RHEreg::calc_h2(const Eigen::Ref<const Eigen::MatrixXd>& AA,
                               const Eigen::Ref<const Eigen::VectorXd>& bb,
                               const bool &reweight_sigmas) const {
	Eigen::ArrayXd ss = AA.colPivHouseholderQr().solve(bb);
	if(reweight_sigmas) {
		ss *= (AA.row(AA.rows()-1)).array() / Nglobal;
	}
	ss = ss / ss.sum();
	return ss;
}

void RHEreg::process_jacknife_samples() {
	// Rescale h2 to avoid bias
	for (long ii = 0; ii < h2_jack.cols(); ii++) {
		h2_jack.col(ii) *= n_var / n_var_jack;
	}

	// SE of h2
	h2_se_jack.resize(h2_jack.cols());
	for (long ii = 0; ii < h2_jack.cols(); ii++) {
		h2_se_jack[ii] = std::sqrt(get_jacknife_var(h2_jack.col(ii)));
	}

	// bias correction
	h2_bias_corrected.resize(h2_jack.cols());
	for (long ii = 0; ii < h2_jack.cols(); ii++) {
		h2_bias_corrected[ii] = get_jacknife_bias_correct(h2_jack.col(ii), h2(ii));
	}
}

Eigen::MatrixXd RHEreg::project_out_covars(Eigen::Ref<Eigen::MatrixXd> rhs) {
	if(n_covar > 0) {
		if (CtC_inv.rows() != n_covar) {
			if(p.mode_debug) std::cout << "Starting compute of CtC_inv" << std::endl;
			Eigen::MatrixXd CtC = C.transpose() * C;
			CtC = mpiUtils::mpiReduce_inplace(CtC);
			CtC_inv = CtC.inverse();
			if(p.mode_debug) std::cout << "Ending compute of CtC_inv" << std::endl;
		}
		return EigenUtils::project_out_covars(rhs, C, CtC_inv);
	} else {
		return rhs;
	}
}

void RHEreg::to_file(const std::string &file) {
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
		if(p.n_jacknife > 1) {
			outf << h2_se_jack[ii] << " ";
			outf << h2_bias_corrected[ii] << std::endl;
		} else {
			outf << "NA NA" << std::endl;
		}
	}

	if(p.RHE_multicomponent) {
		double h2_G_tot = 0, h2_G_sd = 0, h2_GxE_tot = 0, h2_GxE_sd = 0;
		for (int cc = 0; cc < n_components; cc++) {
			if(components[cc].effect_type == "G") {
				h2_G_tot += h2[cc];
				h2_G_sd += h2_se_jack[cc] * h2_se_jack[cc];
			} else if (components[cc].effect_type == "GxE") {
				h2_GxE_tot += h2[cc];
				h2_GxE_sd += h2_se_jack[cc] * h2_se_jack[cc];
			}
		}
		h2_GxE_sd = std::sqrt(h2_GxE_sd);
		h2_G_sd = std::sqrt(h2_G_sd);

		outf << "G_tot NA " << h2_G_tot << " " << h2_G_sd << " NA" << std::endl;
		if(n_env > 0) {
			outf << "GxE_tot NA " << h2_GxE_tot << " " << h2_GxE_sd << " NA" << std::endl;
		}
	}

	boost_io::close(outf);

	if(p.xtra_verbose && p.n_jacknife > 1) {
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
}

double RHEreg::get_jacknife_var(Eigen::ArrayXd jack_estimates) {
	double jack_var = (jack_estimates - jack_estimates.mean()).square().sum();
	jack_var *= (p.n_jacknife - 1.0) / p.n_jacknife;
	return jack_var;
}

double RHEreg::get_jacknife_bias_correct(Eigen::ArrayXd jack_estimates, double full_data_est) {
	double res = p.n_jacknife * full_data_est - (p.n_jacknife - 1.0) * jack_estimates.mean();
	return res;
}

void RHEreg::read_RHE_groups(const std::string& filename){
	// File should have two columns; SNPID and group (both strings)
	// Check has ex[ected header
	std::vector<std::string> file_header;
	read_file_header(filename, file_header);
	std::vector< std::string > case1 = {"SNPID", "group"};
	assert(file_header == case1);

	SNPGROUPS_snpid.clear();
	SNPGROUPS_group.clear();

	// Reading from file
	boost_io::filtering_istream fg;
	std::string gz_str = ".gz";
	if (filename.find(gz_str) != std::string::npos) {
		fg.push(boost_io::gzip_decompressor());
	}
	fg.push(boost_io::file_source(filename));
	if (!fg) {
		std::cout << "ERROR: " << filename << " not opened." << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// Read file twice to ascertain number of lines
	int n_lines = 0;
	std::string line, s;
	// skip header
	getline(fg, line);

	while (getline(fg, line)) {
		std::stringstream ss(line);
		int col_index = 0;
		while (ss >> s) {
			if(col_index == 0) {
				SNPGROUPS_snpid.push_back(s);
			} else if (col_index == 1) {
				SNPGROUPS_group.push_back(s);
			} else {
				throw std::runtime_error(filename + " should only contain two columns");
			}

			col_index += 1;
		}
	}

	std::cout << " Read component groups for " << SNPGROUPS_snpid.size() << " SNPs from " << filename << std::endl;
}

void RHEreg::fill_gaussian_noise(unsigned int seed, Eigen::Ref<Eigen::MatrixXd> zz, long nn, long n_draws) {
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
		MPI_Recv(zz.data(), n_samples * n_draws, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}
