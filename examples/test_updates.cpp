#include <iostream>
#include "../src/variational_parameters.hpp"
#include "../src/vbayes_tracker.hpp"
#include "../src/genotype_matrix.hpp"
#include "../src/tools/eigen3.3/Dense"
#include "../src/utils.hpp"  // sigmoid
#include <random>
#include <limits>
#include "sys/types.h"
#include "sys/sysinfo.h"
#include <cmath>
#include <boost/timer/timer.hpp>


long int N, P;
int mode;
int n_effects = 2;


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



class PlaceHolder {
public:
	GenotypeMatrix X;
	Eigen::MatrixXd E;
	Eigen::ArrayXXd dHtH;
	Eigen::ArrayXXd Hty;
	const double eps = std::numeric_limits<double>::min();
	std::uint32_t n_var;
	long int n_samples;

	PlaceHolder(bool use_compression) : X(use_compression) {};


	void updateAlphaMu(const std::vector< std::uint32_t >& iter,
                       const Hyps& hyps,
                       VariationalParameters& vp){
		Eigen::VectorXd X_kk(n_samples);

		Eigen::ArrayXd alpha_cnst;
		alpha_cnst = (hyps.lambda / (1.0 - hyps.lambda) + eps).log() - hyps.slab_var.log() / 2.0;
		for(std::uint32_t kk : iter ){
			int ee            = kk / n_var;
			std::uint32_t jj = (kk % n_var);

			X_kk = X.col(kk);

			double rr_k = vp.alpha(jj, ee) * vp.mu(jj, ee);

			// Update mu (eq 9); faster to take schur product inside genotype_matrix
			vp.mu(jj, ee) = vp.s_sq(jj, ee) * (Hty(jj, ee) - vp.Hr.dot(X_kk) + dHtH(jj, ee) * rr_k) / hyps.sigma;

			// Update alpha (eq 10)
			double ff_k      = (std::log(vp.s_sq(jj, ee)) + vp.mu(jj, ee) * vp.mu(jj, ee) / vp.s_sq(jj, ee)) / 2.0;
			vp.alpha(jj, ee) = sigmoid(ff_k + alpha_cnst(ee));

			// Update i_Hr; only if coeff is large enough to matter
			double rr_k_diff = (vp.alpha(jj, ee)*vp.mu(jj, ee) - rr_k);
			// if(p.mode_approximate_residuals && std::abs(rr_k_diff) > p.min_residuals_diff){
				// hty_updates++;
				vp.Hr += rr_k_diff * X_kk;
			// }
		}
	}

	void updateAlphaMu2(const std::vector< std::uint32_t >& iter,
					   const Hyps& hyps,
					   VariationalParameters& vp){
		Eigen::VectorXd X_kk(n_samples);

		Eigen::ArrayXd alpha_cnst;
		alpha_cnst = (hyps.lambda / (1.0 - hyps.lambda) + eps).log() - hyps.slab_var.log() / 2.0;
		for(std::uint32_t kk : iter ){
			int ee            = kk / n_var;
			std::uint32_t jj = (kk % n_var);

			X_kk = X.col(kk);

			double rr_k = vp.alpha(jj, ee) * vp.mu(jj, ee);

			// Update mu (eq 9); faster to take schur product inside genotype_matrix
			vp.mu(jj, ee) = vp.s_sq(jj, ee) * (Hty(jj, ee) - vp.Hr.dot(X_kk) + dHtH(jj, ee) * rr_k) / hyps.sigma;

			// Update alpha (eq 10)
			double ff_k      = (std::log(vp.s_sq(jj, ee)) + vp.mu(jj, ee) * vp.mu(jj, ee) / vp.s_sq(jj, ee)) / 2.0;
			vp.alpha(jj, ee) = sigmoid(ff_k + alpha_cnst(ee));

			// Update i_Hr; only if coeff is large enough to matter
			double rr_k_diff = (vp.alpha(jj, ee)*vp.mu(jj, ee) - rr_k);
			// if(p.mode_approximate_residuals && std::abs(rr_k_diff) > p.min_residuals_diff){
				// hty_updates++;
				vp.Hr += rr_k_diff * X_kk;
			// }

			// if(p.mode_alternating_updates){
			Eigen::VectorXd Z_kk(n_samples);
			for(int ee = 1; ee < n_effects; ee++){
				Z_kk = X_kk.cwiseProduct(E.col(ee-1));

				double rr_k = vp.alpha(jj, ee) * vp.mu(jj, ee);

				// Update mu (eq 9); faster to take schur product inside genotype_matrix
				vp.mu(jj, ee) = vp.s_sq(jj, ee) * (Hty(jj, ee) - vp.Hr.dot(Z_kk) + dHtH(jj, ee) * rr_k) / hyps.sigma;

				// Update alpha (eq 10)
				double ff_k      = (std::log(vp.s_sq(jj, ee)) + vp.mu(jj, ee) * vp.mu(jj, ee) / vp.s_sq(jj, ee)) / 2.0;
				vp.alpha(jj, ee) = sigmoid(ff_k + alpha_cnst(ee));

				// Update i_Hr; only if coeff is large enough to matter
				double rr_k_diff = vp.alpha(jj, ee) * vp.mu(jj, ee) - rr_k;
				vp.Hr += rr_k_diff * Z_kk;
			}
		}
	}

	void updateAlphaMu3(const std::vector< std::uint32_t >& iter,
					   const Hyps& hyps,
					   VariationalParameters& vp){
		Eigen::VectorXd X_kk(n_samples);

		Eigen::ArrayXd alpha_cnst;
		alpha_cnst = (hyps.lambda / (1.0 - hyps.lambda) + eps).log() - hyps.slab_var.log() / 2.0;
		for(std::uint32_t kk : iter ){
			int ee            = kk / n_var;
			std::uint32_t jj = (kk % n_var);

			X_kk = X.col(kk);

			_internal_updateAlphaMu(X_kk, ee, jj, vp, hyps, alpha_cnst);

			// if(p.mode_alternating_updates){
			Eigen::VectorXd Z_kk(n_samples);
			for(int ee = 1; ee < n_effects; ee++){
				Z_kk = X_kk.cwiseProduct(E.col(ee-1));

				_internal_updateAlphaMu(Z_kk, ee, jj, vp, hyps, alpha_cnst);
			}
		}
	}

	void _internal_updateAlphaMu(const Eigen::Ref<const Eigen::VectorXd>& H_kk,
	                             const int& ee, std::uint32_t jj,
	                             VariationalParameters& vp,
	                             const Hyps& hyps,
	                             const Eigen::Ref<const Eigen::ArrayXd>& alpha_cnst) __attribute__ ((hot)){
	 	//
		double rr_k_diff;

		if(false){
			double rr_k = vp.alpha(jj, ee) * (vp.mu(jj, ee) - vp.mup(jj, ee)) + vp.mup(jj, ee);

			// Update mu (eq 9); faster to take schur product inside genotype_matrix
			double A       = Hty(jj, ee) - vp.Hr.dot(H_kk) + dHtH(jj, ee) * rr_k;
			vp.mu(jj, ee)  = vp.s_sq(jj, ee) * A / hyps.sigma;
			vp.mup(jj, ee) = vp.sp_sq(jj, ee) * A / hyps.sigma;

			// Update alpha (eq 10)  TODO: check syntax / i_  / sigmoid here!
			double ff_k      = vp.mu(jj, ee) * vp.mu(jj, ee) / vp.s_sq(jj, ee)  / 2.0;
			ff_k            -= vp.mup(jj, ee) * vp.mup(jj, ee) / vp.sp_sq(jj, ee) / 2.0;
			ff_k            += (std::log(vp.s_sq(jj, ee)) - std::log(vp.sp_sq(jj, ee))) / 2.0;
			vp.alpha(jj, ee) = sigmoid(ff_k + alpha_cnst(ee));

			// Update i_Hr; only if coeff is large enough to matter
			double rr_k_new  = vp.alpha(jj, ee) * (vp.mu(jj, ee) - vp.mup(jj, ee)) + vp.mup(jj, ee);
			rr_k_diff        = rr_k_new - rr_k;
		} else {

			double rr_k = vp.alpha(jj, ee) * vp.mu(jj, ee);

			// Update mu (eq 9); faster to take schur product inside genotype_matrix
			vp.mu(jj, ee) = vp.s_sq(jj, ee) * (Hty(jj, ee) - vp.Hr.dot(H_kk) + dHtH(jj, ee) * rr_k) / hyps.sigma;

			// Update alpha (eq 10)
			double ff_k      = (std::log(vp.s_sq(jj, ee)) + vp.mu(jj, ee) * vp.mu(jj, ee) / vp.s_sq(jj, ee)) / 2.0;
			vp.alpha(jj, ee) = sigmoid(ff_k + alpha_cnst(ee));

			// Update i_Hr; only if coeff is large enough to matter
			rr_k_diff        = vp.alpha(jj, ee) * vp.mu(jj, ee) - rr_k;
		}
		// if(p.mode_approximate_residuals && std::abs(rr_k_diff) > p.min_residuals_diff){
		// 	hty_updates++;
		// 	vp.Hr += rr_k_diff * X_kk;
		// }
		vp.Hr += rr_k_diff * H_kk;
	}
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
		std::uint32_t n_var = P;
		PlaceHolder vb_obj((bool) mode);
		vb_obj.n_samples = N;
		vb_obj.n_var = P;
		vb_obj.dHtH = Eigen::ArrayXXd::Random(n_var, n_effects);
		vb_obj.Hty  = Eigen::ArrayXXd::Random(n_var, n_effects);
		Hyps hyps;
		hyps.lambda.resize(2);
		hyps.lambda << 0.005, 0.001;
		hyps.slab_var.resize(2);
		hyps.slab_var << 1.9, 2.1;
		hyps.slab_relative_var.resize(2);
		hyps.slab_relative_var << 1.9, 2.1;
		hyps.sigma = 1.0;
		std::cout << "Hyps initialised" << std::endl;

		// Eigen::MatrixXd G;
		// GenotypeMatrix X((bool) mode); // 0 -> false, else true
		Eigen::MatrixXd E(N, 1);
		vb_obj.X.resize(N, P, n_effects);
		for (long int ii = 0; ii < N; ii++){
			for (long int jj = 0; jj < P; jj++){
				vb_obj.X.assign_index(ii, jj, uniform(gen_unif));
			}
			E(ii, 0) = gaussian(gen_gauss);
		}
		vb_obj.X.E = E;
		vb_obj.E = E;
		vb_obj.X.calc_scaled_values();
		std::cout << "vb_obj initialised" << std::endl;

		// Free params
		VariationalParameters vp;
		vp.Hr = Eigen::VectorXd::Random(N);
		std::cout << "vp.Hr initialised" << std::endl;
		vp.mu.resize(n_var, n_effects);
		vp.alpha.resize(n_var, n_effects);
		for (int ee = 0; ee < n_effects; ee++){
			for (std::uint32_t kk = 0; kk < n_var; 	kk++){
				vp.alpha(kk, ee) = uniform(gen_unif);
				vp.mu(kk, ee)    = gaussian(gen_gauss);
			}
		}
		vp.alpha.rowwise() /= vp.alpha.colwise().sum();
		std::cout << "vp.alpha vp.mu initialised" << std::endl;

		vp.s_sq.resize(n_var, n_effects);
		for (int ee = 0; ee < n_effects; ee++){
			for (std::uint32_t kk = 0; kk < n_var; kk++){
				vp.s_sq(kk, ee)  = hyps.slab_var(ee);
				vp.s_sq(kk, ee) /= (hyps.slab_relative_var(ee) * vb_obj.dHtH(kk, ee) + 1.0);
			}
		}

		std::vector<std::uint32_t > iter;
		for(std::uint32_t kk = 0; kk < n_effects * P; kk++){
			iter.push_back(kk);
		}
		std::vector<std::uint32_t > iter2;
		for(std::uint32_t kk = 0; kk < P; kk++){
			iter2.push_back(kk);
		}

		std::cout << "Data initialised" << std::endl;


		// Calling matrix * vector multiplication
		double secs;
		boost::timer::auto_cpu_timer t1(5, "\nCurrent implementation: %ts \n");

		vb_obj.updateAlphaMu(iter, hyps, vp);

		t1.stop();
		t1.report();
		boost::timer::cpu_times cpu_time1(t1.elapsed());
		secs = (cpu_time1.user + cpu_time1.system) / 1000.0/1000.0/1000.0;
		std::cout << "Est. update time for P = 640k: " << secs / (double) P * 640000.0 << std::endl;


		boost::timer::auto_cpu_timer t2(5, "\nProposed implementation: %ts \n");

		vb_obj.updateAlphaMu2(iter2, hyps, vp);

		t2.stop();
		t2.report();
		boost::timer::cpu_times cpu_time2(t2.elapsed());
		secs = (cpu_time2.user + cpu_time2.system) / 1000.0/1000.0/1000.0;
		std::cout << "Est. update time for P = 640k: " << secs / (double) P * 640000.0 << std::endl;

		boost::timer::auto_cpu_timer t3(5, "\nProposed implementation2: %ts \n");

		vb_obj.updateAlphaMu3(iter2, hyps, vp);

		t3.stop();
		t3.report();
		boost::timer::cpu_times cpu_time3(t3.elapsed());
		secs = (cpu_time3.user + cpu_time3.system) / 1000.0/1000.0/1000.0;
		std::cout << "Est. update time for P = 640k: " << secs / (double) P * 640000.0 << std::endl;
	}

	return 0;
}
