#include <iostream>
#include "src/genotype_matrix.hpp"
#include "src/tools/eigen3.3/Dense"
#include <random>
#include <chrono>
#include <ctime>
#include "sys/types.h"
#include "sys/sysinfo.h"

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
	std::cout << "2 - Eigen matrix" << std::endl;
	if(!(std::cin >> mode)) return false;
	return true;
}

int main() {
	std::default_random_engine gen_unif, gen_gauss;
	std::normal_distribution<double> gaussian(0.0,1.0);
	std::uniform_real_distribution<double> uniform(0.0,2.0);

	// read in N, P from commandline
	while(read()){
		std::cout << "Chosen mode:" << std::endl;
		if(mode == 0){
			std::cout << "0 - GenotypeMatrix, normal." << std::endl;
		} else if(mode == 1){
			std::cout << "1 - GenotypeMatrix, low-mem" << std::endl;
		} else if (mode == 2){
			std::cout << "2 - Eigen matrix" << std::endl;
		} else {
			break;
		}

		Eigen::MatrixXd G;
		GenotypeMatrix GM((bool) mode); // 0 -> false, else true
		Eigen::VectorXd aa(N), rr(P);
		for (long int jj = 0; jj < P; jj++){
			rr[jj] = gaussian(gen_gauss);
		}

		if(mode < 2){
			GM.resize(N, P);
			for (long int ii = 0; ii < N; ii++){
				for (long int jj = 0; jj < P; jj++){
					GM.assign_index(ii, jj, uniform(gen_unif));
				}
				aa[ii] = gaussian(gen_gauss);
			}
			GM.aa = aa;
			GM.calc_scaled_values();
		} else {
			G.resize(N, P);
			for (long int ii = 0; ii < N; ii++){
				for (long int jj = 0; jj < P; jj++){
					G(ii, jj) = uniform(gen_unif);
				}
				aa[ii] = gaussian(gen_gauss);
			}
		}
		std::cout << "Data initialised" << std::endl;

		// call .col many times
		std::cout << "Testing .col() method" << std::endl;
		Eigen::VectorXd tmp;
		auto now = std::chrono::system_clock::now();
		if(mode < 2){
			for(std::size_t jj = 0; jj < P; jj++){
				tmp = GM.col(jj);
			}
		} else {
			for(std::size_t jj = 0; jj < P; jj++){
				tmp = G.col(jj);
			}
		}

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end-now;
		std::cout << P << " main .col() calls finished after " << elapsed_seconds.count() << std::endl;
		now = std::chrono::system_clock::now();

		if(mode < 2){
			for(std::size_t jj = P; jj < 2*P; jj++){
				tmp = GM.col(jj);
			}
		} else {
			for(std::size_t jj = 0; jj < P; jj++){
				tmp = G.col(jj).cwiseProduct(aa);
			}
		}

		end = std::chrono::system_clock::now();
		elapsed_seconds = end-now;
		std::cout << P << " interaction .col() calls finished after " << elapsed_seconds.count() << std::endl;

		// Calling matrix * vector multiplication
		std::cout << "Testing operator()* method" << std::endl;
		now = std::chrono::system_clock::now();
		if(mode < 2){
			for (int ii = 0; ii < 5; ii++){
				tmp = GM * rr;
			}
		} else {
			for (int ii = 0; ii < 5; ii++){
				tmp = G * rr;
			}
		}
		end = std::chrono::system_clock::now();
		elapsed_seconds = end-now;
		std::cout << P << "mean matrix x vector call time: " << elapsed_seconds.count() / 5 << std::endl;
	}

	
	return 0;
}
