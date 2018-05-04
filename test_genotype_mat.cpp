
#include <iostream>
#include "src/genotype_matrix.hpp"
#include "src/tools/eigen3.3/Dense"

//Memory used by a program in this moment

#include "sys/types.h"
#include "sys/sysinfo.h"

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

//Call it like
//LOG.print("total memory used [" + std::to_string(performance::getValueRAM()/1024) + " MB]\n");

int main() {
	// GenotypeMatrix GM(2, 3);
	// Eigen::MatrixXd M(2,3);
	// Eigen::VectorXd vv(3), res, c2a, c2b(2), c2c(2);
	// 
	// // assignment
	// GM.assign_index(0, 0, 0.2);
	// GM.assign_index(0, 1, 0.2);
	// GM.assign_index(0, 2, 0.345);
	// GM.assign_index(1, 0, 0.8);
	// GM.assign_index(1, 1, 0.3);
	// GM.assign_index(1, 2, 0.213);
	// M << 0.2, 0.2, 0.345, 0.8, 0.3, 0.213;
	// vv << 0.3, 0.55, 0.676;
	// 
	// 
	// std::cout << "Uncompressed matrix: " << std::endl << M << std::endl;
	// std::cout << "Compressed matrix: " << std::endl << GM.G << std::endl;
	// std::cout << "Vector used for testing matrix multiplication: " << std::endl << vv << std::endl;
	// 
	// std::cout << "Matrix multiplication with dosage matrix" << std::endl << M * vv << std::endl;
	// res = GM * vv;
	// std::cout << "Matrix multiplication with compressed matrix" << std::endl << res << std::endl;
	// 
	// c2a = GM.col(1);
	// std::cout << "column 1: " << std::endl << c2a << std::endl;
	// 
	// GM.col(1, c2b);
	// std::cout << "column 1: " << std::endl << c2b << std::endl;
	// 
	// GM.assign_col(0, c2a);
	// std::cout << "Compressed matrix after assigning c2 to c1: " << std::endl << GM.G << std::endl;
	// 
	// std::cout << "Entry 2, 2" << std::endl;
	// std::cout << GM.G(1,1) << std::endl;

	typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> MatrixXuc;
	MatrixXuc G = MatrixXuc::Random(2000,1000000);
    
	Eigen::VectorXd res(2000), v = Eigen::VectorXd::Random(1000000);
    
	std::cout << performance::getValueRAM()/1024 <<  " MB of RAM used" << std::endl;

	std::cout << "Running matrix multiplication with explicit cast.";
	std::cout << "This will crash unless eigen is clever in its casting.." << std::endl;

	res = (G.cast<double>() * v);
	std::cout << performance::getValueRAM()/1024 <<  " MB of RAM used" << std::endl;

	// std::cout << "Creating matrixXd with 2 x 10^9 entries. This should crash an A node." << std::endl;
	// Eigen::MatrixXd G = Eigen::MatrixXd::Random(2000,1000000);

	
	return 0;
}
