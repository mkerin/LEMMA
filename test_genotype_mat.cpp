#include <iostream>
#include "src/genotype_matrix.hpp"
#include "src/tools/eigen3.3/Dense"
#include <random>
#include <chrono>
#include <ctime>
#include "sys/types.h"
#include "sys/sysinfo.h"

long int N, P;
bool low_mem;

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
	std::cout << "Use low_mem:" << std::endl;
	if(!(std::cin >> low_mem)) return false;
	return true;
}

int main() {
	std::default_random_engine gen_unif, gen_gauss;
	std::normal_distribution<double> gaussian(0.0,1.0);
	std::uniform_real_distribution<double> uniform(0.0,2.0);

	// read in N, P from commandline
	while(read()){
		std::cout << "Low-mem mode: " << low_mem << std::endl;
		GenotypeMatrix GM(low_mem);
		GM.resize(N, P);
		Eigen::VectorXd aa(N);
		for (long int ii = 0; ii < N; ii++){
			for (long int jj = 0; jj < P; jj++){
				GM.assign_index(ii, jj, uniform(gen_unif));
			}
			aa[ii] = gaussian(gen_gauss);
		}
		std::cout << "aa initialised" << std::endl;
		GM.aa = aa;
		std::cout << "aa assigned" << std::endl;
		auto now = std::chrono::system_clock::now();

		// call .col many times
		Eigen::VectorXd tmp;
		for(std::size_t jj = P; jj < 2*P; jj++){
			tmp = GM.col(jj);
		}

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end-now;
		std::cout << P << " .col() calls finished after " << elapsed_seconds.count() << std::endl;
	}

	
	return 0;
}
