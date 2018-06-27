
// normal_distribution
#include <iostream>
#include <string>
#include <random>

double sigma;
bool read(){
	std::cout << "Input sigma:" << std::endl;
	if(!(std::cin >> sigma)) return false;
	return true;
}

int main()
{
  const int nrolls=10000;  // number of experiments
  const int nstars=100;    // maximum number of stars to distribute

while(read()){
	  std::default_random_engine generator;
	  std::normal_distribution<double> distribution(5.0,sigma);

	  int p[10]={};

	  for (int i=0; i<nrolls; ++i) {
	    double number = distribution(generator);
	    if ((number>=0.0)&&(number<10.0)) ++p[int(number)];
	  }

	  std::cout << "normal_distribution (5.0," << sigma << "):" << std::endl;

	  for (int i=0; i<10; ++i) {
	    std::cout << i << "-" << (i+1) << ": ";
	    std::cout << std::string(p[i]*nstars/nrolls,'*') << std::endl;
	  }
	}

  return 0;
}
