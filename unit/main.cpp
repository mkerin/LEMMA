#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[]){

	MPI_Init(nullptr, nullptr);
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	std::ofstream sink("/dev/null");
	std::streambuf *coutbuf = std::cout.rdbuf();
    if (world_rank != 0) {
        std::cout.rdbuf(sink.rdbuf());
    }

	int result = Catch::Session().run(argc, argv);
    if (world_rank != 0) {
        std::cout.rdbuf(coutbuf);
    }
	MPI_Finalize();
	return result;
}
