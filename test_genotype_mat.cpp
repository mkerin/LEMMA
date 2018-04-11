
#include <iostream>
#include "src/genotype_matrix.hpp"

int main() {
	GenotypeMatrix gg;
	gg.resize(1, 2);
	gg.AssignUncompressedProb(0, 0, 0.02);
	gg.AssignUncompressedProb(0, 1, 0.52);
	
	std::cout << "Printing gg" << std::endl << gg.G << std::endl;
	return 0;
}
