//
// Created by kerin on 2019-09-09.
// Implementation adapted from
// https://github.com/svenpilz/LevenbergMarquardt/blob/master/LevenbergMarquardt.h
//
// Lecture notes:
// http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf

#ifndef LEMMA_LEVENBERG_MARQUARDT_HPP
#define LEMMA_LEVENBERG_MARQUARDT_HPP

#include "rhe_reg_component.hpp"
#include "parameters.hpp"
#include "file_utils.hpp"

#include "tools/eigen3.3/Dense"

#include <vector>
#include <stdexcept>
#include <string>
#include <functional>

namespace boost_io = boost::iostreams;

class LevenbergMarquardt {
public:
	parameters p;
	const std::vector<RHEreg_Component>& components;
	std::vector<RHEreg_Component> gradient_components;
	std::vector<RHEreg_Component> gxe_collapsed_components;

	const std::vector<Eigen::MatrixXd> ytEXXtEys;
	Eigen::MatrixXd ytEXXtEy;
	const Eigen::MatrixXd& E, C, CtC_inv;
	const Eigen::VectorXd& Y;

	const std::vector<std::string>& env_names;
	boost_io::filtering_ostream outf_env;

	long n_components, n_params, n_samples, n_env, n_draws;
	double tau, e1, e2, e3, damping;

	// Time code
	double elapsed_copyData;

	// Used in LM updates
	double u, v, ete, rho;
	long count;
	Eigen::MatrixXd JtJ, Jte;
	Eigen::VectorXd theta, delta;
	Eigen::MatrixXd historic_env_weights;

	LevenbergMarquardt(parameters params,
	                   std::vector<RHEreg_Component>& my_components,
	                   const Eigen::VectorXd& myY,
	                   const Eigen::MatrixXd& myE,
	                   const Eigen::MatrixXd& myC,
	                   const Eigen::MatrixXd& myCtC_inv,
	                   const std::vector<Eigen::MatrixXd>& my_ytEXXtEys,
	                   const std::vector<std::string>& my_env_names) :
		p(params), components(my_components),
		ytEXXtEys(my_ytEXXtEys), Y(myY), E(myE), C(myC),
		CtC_inv(myCtC_inv), env_names(my_env_names){
		assert(!p.RHE_multicomponent);

		// Hardcoded parameters
		tau = 10e-3;
		e1 = 10e-15;
		e2 = 10e-15;
		e3 = 10e-15;
		damping = 2.0;

		// Time code
		elapsed_copyData = 0;

		// WARNING: Have hardcoded assumptions about the order of parameters
		n_components = my_components.size();
		n_params = my_components.size();
		n_env = E.cols();
		n_samples = E.rows();
		n_draws = components[0].n_draws;

		ytEXXtEy = Eigen::MatrixXd::Zero(n_env, n_env);
		for (long jj = 0; jj < ytEXXtEys.size(); jj++) {
			ytEXXtEy += ytEXXtEys[jj];
		}

		for (long ii = 0; ii < n_components; ii++) {
			if(ii == 0) {
				assert(components[ii].effect_type == "G");
			} else if(ii >= 1 && ii < n_env + 1) {
				assert(components[ii].effect_type == "GxE");
				assert(components[ii].env_var_index == ii - 1);
			} else {
				assert(components[ii].effect_type == "noise");
			}
		}
	}

	~LevenbergMarquardt(){
		boost_io::close(outf_env);
	}

	Eigen::VectorXd runLM(){
		Eigen::VectorXd null_noise = Eigen::VectorXd::Zero(n_env);
		Eigen::VectorXd res = runLM(null_noise);
		return res;
	}

	Eigen::VectorXd runLM(Eigen::VectorXd env_noise){
		setupLM(env_noise);
		bool stop = Jte.cwiseAbs().maxCoeff() <= e1;
		while(count < p.levenburgMarquardt_max_iter && !stop) {
			stop = iterLM();
		}

		if (std::isnan(u) || std::isinf(u)) {
			throw std::domain_error("LevenbergMarquardt: µ is NAN or INF.");
		}

		return theta;
	}

	void setupLM(){
		Eigen::VectorXd null_noise = Eigen::VectorXd::Zero(n_env);
		setupLM(null_noise);
	}

	void setupLM(Eigen::VectorXd env_noise){
		assert(env_noise.rows() == n_env);
		// Init: u, v, theta, JtJ, Jte, ete, count

		historic_env_weights = Eigen::MatrixXd::Zero(p.levenburgMarquardt_max_iter + 1, n_env);

		// Use optimal variance comp fit with uniform weighting over envs
		Eigen::VectorXd env_weights = Eigen::VectorXd::Constant(n_env, 1.0 / n_env);
		Eigen::VectorXd initialGuess = Eigen::VectorXd::Zero(n_params);
		initialGuess.segment(1, n_env) = env_weights + env_noise;
		update_aggregated_components(initialGuess, false);

		Eigen::VectorXd bb(3);
		Eigen::MatrixXd AA(3, 3);
		for (long ii = 0; ii < 3; ii++) {
			for (long jj = 0; jj <= ii; jj++) {
				AA(ii, jj) = gxe_collapsed_components[ii] * gxe_collapsed_components[jj];
				AA(jj, ii) = AA(ii, jj);
			}
			bb(ii) = gxe_collapsed_components[ii].get_bb_trace();
		}

		Eigen::VectorXd sigmas = AA.colPivHouseholderQr().solve(bb);
		initialGuess.segment(1, n_env) = (env_weights + env_noise) * sigmas[1];
		initialGuess[0] = sigmas[0];
		initialGuess[n_params - 1] = sigmas[2];
		assert(initialGuess.rows() == n_params);

		std::cout << std::endl << initialGuess << std::endl;

		theta = initialGuess;
		update_aggregated_components(theta);
		JtJ = getJtJ();
		Jte = getJte(theta, JtJ);
		ete = getete();

		v = damping;
		u = tau * JtJ.diagonal().maxCoeff();
		count = 0;

		cache_interim_update(count, theta);
	}

	bool iterLM(){
		auto I = Eigen::MatrixXd::Identity(JtJ.rows(), JtJ.cols());
		bool stop = false;

		delta = (JtJ + u*I).colPivHouseholderQr().solve(Jte);

		if (delta.norm() <= e2*(theta.norm() + e2)) {
			std::cout << "LM terminated as L2(delta) < " << e2*theta.norm() << std::endl;
			stop = true;
		} else {
			// Attempt new step
			Eigen::VectorXd newGuess = theta + delta;

			update_aggregated_components(newGuess);
			Eigen::MatrixXd newJtJ = getJtJ();
			Eigen::MatrixXd newJte = getJte(newGuess, newJtJ);
			double newete = getete();

			rho = ete - newete;
			double denom = 0.5 * (delta.transpose() * (u*delta+Jte))[0];
			rho /= denom;

			if (rho > 0) {
				// Accept step
				theta = newGuess;
				JtJ = newJtJ;
				Jte = newJte;
				ete = newete;
				count++;

				// Reduce damping
				u *= std::max(1.0/3.0, 1.0 - std::pow(2.0*rho-1.0, 3.0));
				v = damping;

				cache_interim_update(count, newGuess);

				// Additional stop conditions
				// max(g) <= e1 OR length(error)^2 <= e3
				if(Jte.cwiseAbs().maxCoeff() <= e1) {
					std::cout << "LM terminated as magnitude of the gradient Jte < " << e1 << std::endl;
					stop = true;
				}
				if(ete <= e3) {
					std::cout << "LM terminated as error dropped below < " << e3 << std::endl;
					stop = true;
				}
			} else {
				// Increase damping
				u *= v;
				v *= damping;
			}
		}
		if(p.mode_debug) std::cout << "LM: rho = " << rho << "; damping = " << u << std::endl;
		return stop;
	}

	void cache_interim_update(const long &iter, const Eigen::Ref<const Eigen::VectorXd> &params){
		Eigen::VectorXd env_weights = params.segment(1, n_env);

		historic_env_weights.row(iter) = env_weights;
	}

	void push_interim_updates(){
		auto filename_env = fileUtils::fstream_init(outf_env, p.out_file, ".lemma_files/", "_lm_env_iter");
		outf_env << "count ";
		for (long ll = 0; ll < n_env; ll++) {
			outf_env << env_names[ll];
			if (ll < n_env - 1) {
				outf_env << " ";
			}
		}
		outf_env << std::endl;

		for (long ii = 0; ii < count; ii++) {
			outf_env << ii << " ";
			for (long ll = 0; ll < n_env; ll++) {
				outf_env << historic_env_weights(ii, ll);
				if (ll < n_env - 1) {
					outf_env << " ";
				}
			}
			outf_env << std::endl;
		}
	}

	void update_aggregated_components(const Eigen::Ref<const Eigen::VectorXd>& params,
	                                  bool update_gradient_comps = true){
		// agg comps
		Eigen::VectorXd env_weights = params.segment(1, n_env);

		Eigen::MatrixXd placeholder = Eigen::MatrixXd::Zero(n_samples, n_draws);
		RHEreg_Component combined_comp(p, Y, C, CtC_inv, placeholder);
		get_GxE_collapsed_component(components, combined_comp, E, env_weights, ytEXXtEys);

		auto start = std::chrono::system_clock::now();

		// Components with gxe term collapsed for getting sum of squared errors
		gxe_collapsed_components.clear();
		gxe_collapsed_components.reserve(3);
		bool transfered_gxe = false;
		for (long ii = 0; ii < n_components; ii++) {
			if(components[ii].effect_type != "GxE") {
				gxe_collapsed_components.push_back(components[ii]);
			} else if(!transfered_gxe) {
				transfered_gxe = true;
				gxe_collapsed_components.push_back(combined_comp);
			}
		}

		// For getting gradient
		gradient_components.clear();
		gradient_components.reserve(1 + n_env);
		for (long ii = 0; ii < n_components; ii++) {
			if(components[ii].effect_type == "GxE") {
				long env_var_index = components[ii].env_var_index;
				gradient_components.push_back(combined_comp);
				gradient_components[ii].change_env_var(E.col(env_var_index));
				gradient_components[ii].ytXXty = (ytEXXtEy * env_weights)(env_var_index);
			} else {
				gradient_components.push_back(components[ii]);
			}
		}

		auto end = std::chrono::system_clock::now();
		auto diff = end - start;
		elapsed_copyData += diff.count();
	}

	Eigen::MatrixXd getJtJ() const {
		Eigen::MatrixXd A(n_params, n_params);
		for (long ii = 0; ii < n_params; ii++) {
			for (long jj = 0; jj <= ii; jj++) {
				A(ii, jj) = gradient_components[ii] * gradient_components[jj];
				A(jj, ii) = A(ii, jj);
			}
		}
		return A;
	}

	Eigen::MatrixXd getJte(const Eigen::Ref<const Eigen::VectorXd>& params,
	                       const Eigen::Ref<const Eigen::MatrixXd>& JtJ) const {
		Eigen::VectorXd gg(n_params);
		for (long ii = 0; ii < n_params; ii++) {
			gg(ii) = gradient_components[ii].get_bb_trace();
		}
		gg -= JtJ * params;
		return gg;
	}

	double getete() const {
		Eigen::VectorXd bb(3);
		Eigen::MatrixXd AA(3, 3);

		for (long ii = 0; ii < 3; ii++) {
			for (long jj = 0; jj <= ii; jj++) {
				AA(ii, jj) = gxe_collapsed_components[ii] * gxe_collapsed_components[jj];
				AA(jj, ii) = AA(ii, jj);
			}
			bb(ii) = gxe_collapsed_components[ii].get_bb_trace();
		}
		Eigen::VectorXd sigmas = AA.colPivHouseholderQr().solve(bb);

		double obj = Y.squaredNorm();
		obj = mpiUtils::mpiReduce_inplace(obj);
		obj *= obj;
		obj = obj -2 * sigmas.dot(bb) + sigmas.dot(AA * sigmas);
		return obj;
	}
};

#endif //LEMMA_LEVENBERG_MARQUARDT_HPP
