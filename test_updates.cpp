#include <iostream>
#include "src/vbayes_tracker.hpp"
#include "src/vbayes_x2.hpp"
#include "src/genotype_matrix.hpp"
#include "src/tools/eigen3.3/Dense"
#include <random>
#include <chrono>
#include <ctime>
#include <limits>
#include "sys/types.h"
#include "sys/sysinfo.h"
#include <cmath>

inline double sigmoid(double x){
	return 1.0 / (1.0 + std::exp(-x));
}

void updateAlphaMu(const GenotypeMatrix& X,
                   const Eigen::Ref<Eigen::VectorXd>& dHtH,
                   const Eigen::Ref<Eigen::VectorXd>& Hty,
                   const std::vector< std::uint32_t >& iter,
                   const std::uint32_t& L,
                   const Hyps& i_hyps,
                   FreeParameters& i_par);

long int N, P;
int mode;


bool read(){
	std::cout << "Input N:" << std::endl;
	if(!(std::cin >> N)) return false;
	std::cout << "Input P:" << std::endl;
	if(!(std::cin >> P)) return false;
	std::cout << "Mode; " << std::endl;
	std::cout << "0 - GenotypeMatrix, normal." << std::endl;
	std::cout << "1 - GenotypeMatrix, low-mem" << std::endl;
	// std::cout << "2 - Eigen matrix" << std::endl;
	if(!(std::cin >> mode)) return false;
	return true;
}

int main() {
	std::default_random_engine gen_unif, gen_gauss;
	std::normal_distribution<double> gaussian(0.0,1.0);
	std::uniform_real_distribution<double> uniform(0.0,2.0);
	std::uniform_real_distribution<double> standard_uniform(0.0,1.0);

	// read in N, P from commandline
	while(read()){
		std::cout << "Chosen mode:" << std::endl;
		if(mode == 0){
			std::cout << "0 - GenotypeMatrix, normal." << std::endl;
		} else if(mode == 1){
			std::cout << "1 - GenotypeMatrix, low-mem" << std::endl;
		} else {
			break;
		}

		// Constants
		double eps = std::numeric_limits<double>::min();
		Eigen::Vector dHtH = Eigen::VectorXd::Random(2*P);
		Eigen::Vector Hty  = Eigen::VectorXd::Random(2*P);
		Hyps hyps;
		hyps.lam_g = 0.001;
		hyps.lam_b = 0.005;
		hyps.sigma_g = 2.1;
		hyps.sigma_b = 1.9;
		hyps.sigma = 1.0;

		// Eigen::MatrixXd G;
		GenotypeMatrix X((bool) mode); // 0 -> false, else true
		Eigen::VectorXd aa(N);
		X.resize(N, P);
		for (long int ii = 0; ii < N; ii++){
			for (long int jj = 0; jj < P; jj++){
				X.assign_index(ii, jj, uniform(gen_unif));
			}
			aa[ii] = gaussian(gen_gauss);
		}
		X.aa = aa;
		X.calc_scaled_values();

		// Free params
		FreeParameters i_par;
		i_par.Hr = Eigen::VectorXd::Random(N);
		i_par.alpha.resize(2*P);
		i_par.mu.resize(2*P);
		i_par.s_sq.resize(2*P);
		for (long int jj = 0; jj < 2*P; jj++){
			i_par.alpha[jj] = standard_uniform(gen_unif);
			i_par.mu[jj] = standard_uniform(gen_gauss);
		}
		for (std::uint32_t kk = 0; kk < P; kk++){
			i_par.s_sq[kk] = i_hyps.sigma_b * i_hyps.sigma / (i_hyps.sigma_b * dHtH(kk) + 1.0);
		}
		for (std::uint32_t kk = P; kk < 2*P; kk++){
			i_par.s_sq[kk] = i_hyps.sigma_g * i_hyps.sigma / (i_hyps.sigma_g * dHtH(kk) + 1.0);
		}

		std::cout << "Data initialised" << std::endl;


		// Calling matrix * vector multiplication
		std::cout << "Testing updateAlphaMu method" << std::endl;
		auto now = std::chrono::system_clock::now();

		updateAlphaMu(X, dHtH, Hty, iter, 2*P, i_hyps, i_par);

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end-now;
		std::cout << "Mean update time (" << P << " calls): " << elapsed_seconds.count() / P << std::endl;
		std::cout << "Est. update time for P = 640k: " << elapsed_seconds.count() / P * 640000 << std::endl;
	}

	return 0;
}


void updateAlphaMu(const GenotypeMatrix& X,
                   const Eigen::Ref<Eigen::VectorXd>& dHtH,
                   const Eigen::Ref<Eigen::VectorXd>& Hty,
                   const std::vector< std::uint32_t >& iter,
                   const std::uint32_t& L,
                   const Hyps& i_hyps,
                   FreeParameters& i_par){
	const eps = std::numeric_limits<double>::min();
	std::uint32_t kk;
	double rr_k, ff_k;
	Eigen::VectorXd X_kk(X.NN);
	for(std::uint32_t jj = 0; jj < L; jj++){
		kk = iter[jj];
		assert(kk < L);

		rr_k = i_par.alpha(kk) * i_par.mu(kk);
		X_kk = X.col(kk);
		// Update mu (eq 9); faster to take schur product with aa inside genotype_matrix
		i_par.mu(kk) = i_par.s_sq[kk] * (Hty(kk) - i_par.Hr.dot(X_kk) + dHtH(kk) * rr_k) / i_hyps.sigma;

		// Update alpha (eq 10)  TODO: check syntax / i_  / sigmoid here!
		if (kk < P){
			ff_k = std::log(i_hyps.lam_b / (1.0 - i_hyps.lam_b) + eps) + std::log(i_par.s_sq[kk] / i_hyps.sigma_b / i_hyps.sigma + eps) / 2.0;
			ff_k += i_par.mu(kk) * i_par.mu(kk) / i_par.s_sq[kk] / 2.0;
		} else {
			ff_k = std::log(i_hyps.lam_g / (1.0 - i_hyps.lam_g) + eps) + std::log(i_par.s_sq[kk] / i_hyps.sigma_g / i_hyps.sigma + eps) / 2.0;
			ff_k += i_par.mu(kk) * i_par.mu(kk) / i_par.s_sq[kk] / 2.0;
		}
		i_par.alpha(kk) = sigmoid(ff_k);

		// Update i_Hr; faster to take schur product with aa inside genotype_matrix
		i_par.Hr = i_par.Hr + (i_par.alpha(kk)*i_par.mu(kk) - rr_k) * X_kk;
	}
}


	// 	// Calling matrix * vector multiplication
	// 	std::cout << "Testing updateAlphaMu method" << std::endl;
	// 	auto now = std::chrono::system_clock::now();
	// 
	// 	double rr_k, ff_k;
	// 	for(std::uint32_t kk = 0; kk < P; kk++){
	// 		rr_k = alpha(kk) * mu(kk);
	// 
	// 		// Update mu (eq 9)
	// 		mu(kk) = s_sq * (Hty - Hr.dot(X.col(kk)) + dHtH * rr_k) / sigma;
	// 
	// 		// Update alpha (eq 10)  TODO: check syntax / i_  / sigmoid here!
	// 		if (kk < P){
	// 			ff_k = std::log(lam_b / (1.0 - lam_b) + eps) + std::log(s_sq / sigma_b / sigma + eps) / 2.0;
	// 			ff_k += mu(kk) * mu(kk) / s_sq / 2.0;
	// 		} else {
	// 			ff_k = std::log(lam_g / (1.0 - lam_g) + eps) + std::log(s_sq / sigma_g / sigma + eps) / 2.0;
	// 			ff_k += mu(kk) * mu(kk) / s_sq / 2.0;
	// 		}
	// 		alpha(kk) = sigmoid(ff_k);
	// 
	// 		// Update i_Hr; faster to take schur product with aa inside genotype_matrix
	// 		new_Hr = Hr + (alpha(kk)*mu(kk) - rr_k) * (X.col(kk));
	// 	}
	// 
	// 	auto end = std::chrono::system_clock::now();
	// 	std::chrono::duration<double> elapsed_seconds = end-now;
	// 	std::cout << "Mean update time (main, " << P << " calls): " << elapsed_seconds.count() / P << std::endl;
	// 	std::cout << "Est. update time for P = 600k: " << elapsed_seconds.count() / P * 600000 << std::endl;
	// 
	// 	now = std::chrono::system_clock::now();
	// 	for(std::uint32_t kk = P; kk < 2*P; kk++){
	// 		rr_k = alpha(kk) * mu(kk);
	// 
	// 		// Update mu (eq 9)
	// 		mu(kk) = s_sq * (Hty - Hr.dot(X.col(kk)) + dHtH * rr_k) / sigma;
	// 
	// 		// Update alpha (eq 10)  TODO: check syntax / i_  / sigmoid here!
	// 		if (kk < P){
	// 			ff_k = std::log(lam_b / (1.0 - lam_b) + eps) + std::log(s_sq / sigma_b / sigma + eps) / 2.0;
	// 			ff_k += mu(kk) * mu(kk) / s_sq / 2.0;
	// 		} else {
	// 			ff_k = std::log(lam_g / (1.0 - lam_g) + eps) + std::log(s_sq / sigma_g / sigma + eps) / 2.0;
	// 			ff_k += mu(kk) * mu(kk) / s_sq / 2.0;
	// 		}
	// 		alpha(kk) = sigmoid(ff_k);
	// 
	// 		// Update i_Hr; faster to take schur product with aa inside genotype_matrix
	// 		new_Hr = Hr + (alpha(kk)*mu(kk) - rr_k) * (X.col(kk));
	// 	}
	// 	end = std::chrono::system_clock::now();
	// 	elapsed_seconds = end-now;
	// 	std::cout << "Mean update time (interaction, " << P << " calls): " << elapsed_seconds.count() / P << std::endl;
	// 	std::cout << "Est. update time for P = 600k: " << elapsed_seconds.count() / P * 600000 << std::endl;
	// }
