#include <iostream>
#include <boost/timer/timer.hpp>
#include "../src/tools/eigen3.3/Dense"

/*
Comparison of vector vector to vector-matrix computations
*/

int main() {
	long int N;
    int L = 1000;
    std::cout << "Input N:" << std::endl;
	if(!(std::cin >> N)) return false;

	int n = Eigen::nbThreads( );
	std::cout << "Number of threads used: " << n << std::endl;

	// Comparison of vector vector vs matrix-vector
	Eigen::MatrixXd M = Eigen::MatrixXd::Random(64, N);
	Eigen::VectorXd m = Eigen::VectorXd::Random(N, 1);
	Eigen::VectorXd v = Eigen::VectorXd::Random(N, 1);
	Eigen::VectorXd Y(64);

	// vector-Vector
	std::cout << "vector-vector [write to double]" << std::endl;
    boost::timer::auto_cpu_timer t3a(5, "1000 repeats takes: %ts \n");
	for(int kk = 0; kk < L; kk++){
		for (int bb = 0; bb < 64; bb++){
			double y = m.transpose() * v;
		}
	}
	t3a.stop();
    t3a.report();

	std::cout << "vector-vector [write to eigen]" << std::endl;
    boost::timer::auto_cpu_timer t3b(5, "1000 repeats takes: %ts \n");
	for(int kk = 0; kk < L; kk++){
		for (int bb = 0; bb < 64; bb++){
			Y[bb] = m.transpose() * v;
		}
	}
	t3b.stop();
    t3b.report();

	// matrix vector
	std::cout << "matrix-vector" << std::endl;
    boost::timer::auto_cpu_timer t3c(5, "1000 repeats takes: %ts \n");
	for(int kk = 0; kk < L; kk++){
		Y = M * v;
	}
	t3c.stop();
    t3c.report();

	return 0;
}
