
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "src/tools/eigen3.3/Dense"

 
int main() {
	std::vector< std::string > names = {"hello", "you"};
	for (int kk = 0; kk < 2; kk++){
		std::cout << names[kk] << std::endl;
	}

	// check isfinite
	printf ("isfinite(0.0)       : %d\n",std::isfinite(0.0));
	printf ("isfinite(1.0/0.0)   : %d\n",std::isfinite(1.0/0.0));
	printf ("isfinite(-1.0/0.0)  : %d\n",std::isfinite(-1.0/0.0));
	printf ("isfinite(sqrt(-1.0)): %d\n",std::isfinite(std::sqrt(-1.0)));

	std::cout << 0.0 << std::endl;
	std::cout << 1.0/0.0 << std::endl;
	std::cout << -1.0/0.0 << std::endl;
	std::cout << std::sqrt(-1.0) << std::endl;

	// Check Eigen << operator
	Eigen::VectorXd v(3);
	v << 1, 2, 3;
	std::cout << v << std::endl;
	v << 4, 5, 6;
	std::cout << v << std::endl;

	// check includes function
	std::vector< std::string > v1 = {"one", "two", "three", "four"};
	std::vector< std::string > v2 = {"one", "two", "three"};

	if(std::includes(v1.begin(), v1.end(), v2.begin(), v2.end())){
		std::cout << "v1 contains v2" << std::endl;
	} else {
		std::cout << "ERROR: expected v1 to contain v2" << std::endl;
	}


	return 0;
}
