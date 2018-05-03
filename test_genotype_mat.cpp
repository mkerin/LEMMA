
#include <iostream>
#include "src/genotype_matrix.hpp"
#include "src/tools/eigen3.3/Dense"

int main() {
	GenotypeMatrix GM(2, 3);
	Eigen::MatrixXd M(2,3);
	Eigen::VectorXd vv(3), res, c2a, c2b(2), c2c(2);

	// assignment
	GM.assign_index(0, 0, 0.2);
	GM.assign_index(0, 1, 0.2);
	GM.assign_index(0, 2, 0.345);
	GM.assign_index(1, 0, 0.8);
	GM.assign_index(1, 1, 0.3);
	GM.assign_index(1, 2, 0.213);
	M << 0.2, 0.2, 0.345, 0.8, 0.3, 0.213;
	vv << 0.3, 0.55, 0.676;


	std::cout << "Uncompressed matrix: " << std::endl << M << std::endl;
	std::cout << "Compressed matrix: " << std::endl << GM.G << std::endl;
	std::cout << "Vector used for testing matrix multiplication: " << std::endl << vv << std::endl;
	
	std::cout << "Matrix multiplication with dosage matrix" << std::endl << M * vv << std::endl;
	res = GM * vv;
	std::cout << "Matrix multiplication with compressed matrix" << std::endl << res << std::endl;

	c2a = GM.col(1);
	std::cout << "column 1: " << std::endl << c2a << std::endl;

	GM.col(1, c2b);
	std::cout << "column 1: " << std::endl << c2b << std::endl;

	GM.assign_col(0, c2a);
	std::cout << "Compressed matrix after assigning c2 to c1: " << std::endl << GM.G << std::endl;

	return 0;
}
