#include <iostream>
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

long int N, P;
int mode;

namespace performance {
    int parseLineRAM(char* line){
        // This assumes that a digit will be found and the line ends in " Kb".
        int i = strlen(line);
        const char* p = line;
        while (*p <'0' || *p > '9') p++;
        line[i-3] = '\0';
        i = atoi(p);
        return i;
    }

    int getValueRAM(){ //Note: this value is in KB!
        FILE* file = fopen("/proc/self/status", "r");
        int result = -1;
        char line[128];

        while (fgets(line, 128, file) != NULL){
            if (strncmp(line, "VmRSS:", 6) == 0){
                result = parseLineRAM(line);
                break;
            }
        }
        fclose(file);
        return result;
    }
}

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

		// Eigen::MatrixXd G;
		GenotypeMatrix X((bool) mode); // 0 -> false, else true
		Eigen::VectorXd aa(N), rr(P);
		for (long int jj = 0; jj < P; jj++){
			rr[jj] = gaussian(gen_gauss);
		}

		X.resize(N, P);
		for (long int ii = 0; ii < N; ii++){
			for (long int jj = 0; jj < P; jj++){
				X.assign_index(ii, jj, uniform(gen_unif));
			}
			aa[ii] = gaussian(gen_gauss);
		}
		X.aa = aa;
		X.calc_scaled_values();

		Eigen::VectorXd alpha(2*P), mu(2*P), Hr(N), new_Hr(N);
		for (long int jj = 0; jj < P; jj++){
			alpha[jj] = standard_uniform(gen_unif);
			mu[jj] = standard_uniform(gen_gauss);
		}
		for (long int ii = 0; ii < N; ii++){
			Hr[ii] = standard_uniform(gen_gauss);
		}

		std::cout << "Data initialised" << std::endl;



		double lam_g = 0.001, lam_b = 0.005, sigma_g = 2.1, sigma_b = 1.9, sigma = 1.0;
		double eps = std::numeric_limits<double>::min();

		// Cba to track
		double dHtH = 1.2, Hty = 23.0, s_sq = 0.9;


		// Calling matrix * vector multiplication
		std::cout << "Testing updateAlphaMu method" << std::endl;
		auto now = std::chrono::system_clock::now();

		double rr_k, ff_k;
		for(std::uint32_t kk = 0; kk < P; kk++){
			rr_k = alpha(kk) * mu(kk);

			// Update mu (eq 9)
			mu(kk) = s_sq * (Hty - Hr.dot(X.col(kk)) + dHtH * rr_k) / sigma;

			// Update alpha (eq 10)  TODO: check syntax / i_  / sigmoid here!
			if (kk < P){
				ff_k = std::log(lam_b / (1.0 - lam_b) + eps) + std::log(s_sq / sigma_b / sigma + eps) / 2.0;
				ff_k += mu(kk) * mu(kk) / s_sq / 2.0;
			} else {
				ff_k = std::log(lam_g / (1.0 - lam_g) + eps) + std::log(s_sq / sigma_g / sigma + eps) / 2.0;
				ff_k += mu(kk) * mu(kk) / s_sq / 2.0;
			}
			alpha(kk) = sigmoid(ff_k);

			// Update i_Hr; faster to take schur product with aa inside genotype_matrix
			new_Hr = Hr + (alpha(kk)*mu(kk) - rr_k) * (X.col(kk));
		}

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end-now;
		std::cout << "Mean update time (main, " << P << " calls): " << elapsed_seconds.count() / P << std::endl;
		std::cout << "Est. update time for P = 600k: " << elapsed_seconds.count() / P * 600000 << std::endl;

		now = std::chrono::system_clock::now();
		for(std::uint32_t kk = P; kk < 2*P; kk++){
			rr_k = alpha(kk) * mu(kk);

			// Update mu (eq 9)
			mu(kk) = s_sq * (Hty - Hr.dot(X.col(kk)) + dHtH * rr_k) / sigma;

			// Update alpha (eq 10)  TODO: check syntax / i_  / sigmoid here!
			if (kk < P){
				ff_k = std::log(lam_b / (1.0 - lam_b) + eps) + std::log(s_sq / sigma_b / sigma + eps) / 2.0;
				ff_k += mu(kk) * mu(kk) / s_sq / 2.0;
			} else {
				ff_k = std::log(lam_g / (1.0 - lam_g) + eps) + std::log(s_sq / sigma_g / sigma + eps) / 2.0;
				ff_k += mu(kk) * mu(kk) / s_sq / 2.0;
			}
			alpha(kk) = sigmoid(ff_k);

			// Update i_Hr; faster to take schur product with aa inside genotype_matrix
			new_Hr = Hr + (alpha(kk)*mu(kk) - rr_k) * (X.col(kk));
		}
		end = std::chrono::system_clock::now();
		elapsed_seconds = end-now;
		std::cout << "Mean update time (interaction, " << P << " calls): " << elapsed_seconds.count() / P << std::endl;
		std::cout << "Est. update time for P = 600k: " << elapsed_seconds.count() / P * 600000 << std::endl;
	}

	return 0;
}
