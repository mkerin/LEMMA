#include <iostream>
#include "src/vbayes_x2.hpp"
#include "src/vbayes_tracker.hpp"
#include "src/genotype_matrix.hpp"
#include "src/tools/eigen3.3/Dense"
#include <random>
#include <chrono>
#include <ctime>
#include <limits>
#include "sys/types.h"
#include "sys/sysinfo.h"
#include <cmath>


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



class TMP {
public:
	GenotypeMatrix X;
	Eigen::VectorXd dHtH;
	Eigen::VectorXd Hty;
	const double eps = std::numeric_limits<double>::min();
	std::uint32_t n_var;
	long int n_samples;

	TMP(bool use_compression) : X(use_compression) {};
	

	void updateAlphaMu(const std::vector< std::uint32_t >& iter,
	                    const std::uint32_t& L,
	                    const Hyps& i_hyps,
	                    FreeParameters& i_par){
		std::uint32_t kk;
		double rr_k, ff_k, rr_k_diff;
		Eigen::VectorXd X_kk(X.NN);

		double bbb_b, bbb_g;
		bbb_b = std::log(i_hyps.lam_b / (1.0 - i_hyps.lam_b) + eps) - std::log(i_hyps.sigma_b * i_hyps.sigma) / 2.0;
		bbb_g = std::log(i_hyps.lam_g / (1.0 - i_hyps.lam_g) + eps) - std::log(i_hyps.sigma_g * i_hyps.sigma) / 2.0;

		for(std::uint32_t kk : iter ){
			rr_k = i_par.alpha(kk) * i_par.mu(kk);
			X_kk = X.col(kk);

			// Update mu (eq 9); faster to take schur product with aa inside genotype_matrix
			i_par.mu(kk) = i_par.s_sq[kk] * (Hty(kk) - i_par.Hr.dot(X_kk) + dHtH(kk) * rr_k) / i_hyps.sigma;
	
			// Update alpha (eq 10)  TODO: check syntax / i_  / sigmoid here!
			if (kk < n_var){
				ff_k = bbb_b + (std::log(i_par.s_sq[kk]) + i_par.mu(kk) * i_par.mu(kk) / i_par.s_sq[kk]) / 2.0;
			} else {
				ff_k = bbb_g + (std::log(i_par.s_sq[kk]) + i_par.mu(kk) * i_par.mu(kk) / i_par.s_sq[kk]) / 2.0;
			}
			i_par.alpha(kk) = sigmoid(ff_k);
	
			// Update i_Hr; faster to take schur product with aa inside genotype_matrix
			// rr_k_diff = (i_par.alpha(kk)*i_par.mu(kk) - rr_k);
			// if(std::abs(rr_k_diff) > 1e-9){
			if(kk < n_var / 5){
				i_par.Hr += (i_par.alpha(kk)*i_par.mu(kk) - rr_k) * X_kk;
			}
		}
	}

	// void updateAlphaMu(std::vector< std::uint32_t > iter, std::uint32_t L,
	//                    Hyps i_hyps, FreeParameters& i_par){
	// 	std::uint32_t kk;
	// 	double rr_k, ff_k;
	// 	Eigen::VectorXd X_kk(n_samples);
	// 	for(std::uint32_t jj = 0; jj < L; jj++){
	// 		kk = iter[jj];
	// 		assert(kk < L);
	// 
	// 		rr_k = i_par.alpha(kk) * i_par.mu(kk);
	// 		X_kk = X.col(kk);
	// 		// Update mu (eq 9); faster to take schur product with aa inside genotype_matrix
	// 		i_par.mu(kk) = i_par.s_sq[kk] * (Hty(kk) - i_par.Hr.dot(X_kk) + dHtH(kk) * rr_k) / i_hyps.sigma;
	// 
	// 		// Update alpha (eq 10)  TODO: check syntax / i_  / sigmoid here!
	// 		if (kk < n_var){
	// 			ff_k = std::log(i_hyps.lam_b / (1.0 - i_hyps.lam_b) + eps) + std::log(i_par.s_sq[kk] / i_hyps.sigma_b / i_hyps.sigma + eps) / 2.0;
	// 			ff_k += i_par.mu(kk) * i_par.mu(kk) / i_par.s_sq[kk] / 2.0;
	// 		} else {
	// 			ff_k = std::log(i_hyps.lam_g / (1.0 - i_hyps.lam_g) + eps) + std::log(i_par.s_sq[kk] / i_hyps.sigma_g / i_hyps.sigma + eps) / 2.0;
	// 			ff_k += i_par.mu(kk) * i_par.mu(kk) / i_par.s_sq[kk] / 2.0;
	// 		}
	// 		i_par.alpha(kk) = sigmoid(ff_k);
	// 
	// 		// Update i_Hr; faster to take schur product with aa inside genotype_matrix
	// 		i_par.Hr = i_par.Hr + (i_par.alpha(kk)*i_par.mu(kk) - rr_k) * (X_kk);
	// 	}
	// }
};


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
		TMP vb_obj((bool) mode);
		vb_obj.n_samples = N;
		vb_obj.n_var = P;
		vb_obj.dHtH = Eigen::VectorXd::Random(2*P);
		vb_obj.Hty  = Eigen::VectorXd::Random(2*P);
		Hyps hyps;
		hyps.lam_g = 0.001;
		hyps.lam_b = 0.005;
		hyps.sigma_g = 2.1;
		hyps.sigma_b = 1.9;
		hyps.sigma = 1.0;

		// Eigen::MatrixXd G;
		// GenotypeMatrix X((bool) mode); // 0 -> false, else true
		Eigen::VectorXd aa(N);
		vb_obj.X.resize(N, P);
		for (long int ii = 0; ii < N; ii++){
			for (long int jj = 0; jj < P; jj++){
				vb_obj.X.assign_index(ii, jj, uniform(gen_unif));
			}
			aa[ii] = gaussian(gen_gauss);
		}
		vb_obj.X.aa = aa;
		vb_obj.X.calc_scaled_values();

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
			i_par.s_sq[kk] = hyps.sigma_b * hyps.sigma / (hyps.sigma_b * vb_obj.dHtH(kk) + 1.0);
		}
		for (std::uint32_t kk = P; kk < 2*P; kk++){
			i_par.s_sq[kk] = hyps.sigma_g * hyps.sigma / (hyps.sigma_g * vb_obj.dHtH(kk) + 1.0);
		}

		std::vector<std::uint32_t > iter;
		for(std::uint32_t kk = 0; kk < 2*P; kk++){
			iter.push_back(kk);
		}

		std::cout << "Data initialised" << std::endl;


		// Calling matrix * vector multiplication
		std::cout << "Current implementation:" << std::endl;
		auto now = std::chrono::system_clock::now();

		vb_obj.updateAlphaMu(iter, 2*P, hyps, i_par);

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end-now;
		std::cout << "Mean update time (" << P << " calls): " << elapsed_seconds.count() / P << std::endl;
		std::cout << "Est. update time for P = 640k: " << elapsed_seconds.count() / P * 640000 << std::endl;

		std::cout << std::endl << "Proposed implementation:" << std::endl;
		now = std::chrono::system_clock::now();

		vb_obj.updateAlphaMu2(iter, 2*P, hyps, i_par);

		end = std::chrono::system_clock::now();
		elapsed_seconds = end-now;
		std::cout << "Mean update time (" << P << " calls): " << elapsed_seconds.count() / P << std::endl;
		std::cout << "Est. update time for P = 640k: " << elapsed_seconds.count() / P * 640000 << std::endl;
	}

	return 0;
}
