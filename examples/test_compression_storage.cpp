#include <iostream>
#include <boost/timer/timer.hpp>
#include "../src/tools/eigen3.3/Dense"

/*
Lookup from table of normalised values vs computing on the fly.
Seems like computing on the fly substantially faster.
*/

int main() {
	long int N;
    int L = 1000;
    std::cout << "Input N:" << std::endl;
	if(!(std::cin >> N)) return false;

	int n = Eigen::nbThreads( );
	std::cout << "Number of threads used: " << n << std::endl;

    Eigen::ArrayXd x0(N), x1(N), x2(N), table(256);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> xx = Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>::Random(N);

    double w = 1.0 / 256.0;
    for (int ii = 0; ii < 256; ii++){
        table[ii] = (double) ii * w + 0.5;
    }

    // Fill x0
    std::cout << "Fill x0 with computed values (niave)" << std::endl;
    boost::timer::auto_cpu_timer t0(5, "Compute value: %ts \n");
    for (int kk = 0; kk < L; kk++){
        for (long int ii = 0; ii < N; ii++){
            x0[ii] = (xx[ii] + 0.5) * w;
        }
    }
    t0.stop();
    t0.report();

    // Fill x1
    std::cout << "Fill x1 with computed values (vector operations)" << std::endl;
    boost::timer::auto_cpu_timer t1(5, "Compute value: %ts \n");
    for (int kk = 0; kk < L; kk++){
        x1 = xx.cast<double>();
        x1 *= w;
        x1 += w * 0.5;
    }
    t1.stop();
    t1.report();

    // Fill x2
    std::cout << "Fill x2 with lookup values" << std::endl;
    boost::timer::auto_cpu_timer t2(5, "Lookup value: %ts \n");
    for (int kk = 0; kk < L; kk++){
        for (long int ii = 0; ii < N; ii++){
            x2[ii] = table[xx[ii]];
        }
    }
    t2.stop();
    t2.report();

	// Comparison of vector vector vs matrix-vector
	Eigen::MatrixXd M = Eigen::MatrixXd::Random(64, N);
	Eigen::VectorXd m = Eigen::VectorXd::Random(N, 1);
	Eigen::VectorXd v = Eigen::VectorXd::Random(N, 1);
	Eigen::VectorXd Y(64);

	// vector-Vector
	std::cout << "vector-vector [write to double]" << std::endl;
    boost::timer::auto_cpu_timer t3a(5, "Lookup value: %ts \n");
	for(int kk = 0; kk < L; kk++){
		for (int bb = 0; bb < 64; bb++){
			double y = m.transpose() * v;
		}
	}
	t3a.stop();
    t3a.report();

	std::cout << "vector-vector [write to eigen]" << std::endl;
    boost::timer::auto_cpu_timer t3b(5, "Lookup value: %ts \n");
	for(int kk = 0; kk < L; kk++){
		for (int bb = 0; bb < 64; bb++){
			Y[bb] = m.transpose() * v;
		}
	}
	t3b.stop();
    t3b.report();

	// matrix vector
	std::cout << "matrix-vector" << std::endl;
    boost::timer::auto_cpu_timer t3c(5, "Lookup value: %ts \n");
	for(int kk = 0; kk < L; kk++){
		Y = M * v;
	}
	t3c.stop();
    t3c.report();

	return 0;
}
