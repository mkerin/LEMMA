# Installation

## External dependencies
LEMMA requires the [BGEN file format](http://www.bgenformat.org/), [Boost](https://www.boost.org/) and OpenMPI. We also recommend compiling with the Intel MKL library.

Required dependencies:
- [GCC](https://gcc.gnu.org/) version 9.4 or above
- [CMake](https://cmake.org/) version 3.16 or above
- [Boost](https://www.boost.org/) version 1.78 or above
- [OpenMPI](https://www.open-mpi.org/) version 3.1 or above

Note that older versions of GCC and Boost may also work to compile LEMMA. However as GCC v9.4 and Boost v1.78 are the minimum versions available for testing with the github continuous integration service, we are not able to verify this.

Optional dependencies:
- Intel MKL (Math Kernel Library)

We observed that compiling with the intel MKL library resulted in some improvement in runtime speed, but these may be platform dependent.

## Running CMake
First clone the [GitHub repository](https://github.com/mkerin/LEMMA)
```
git clone https://github.com/mkerin/LEMMA.git
cd LEMMA
```
Then compile
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=RELEASE
cmake --build build -- -j4
```
Then test
```
ctest --test-dir build
```

See options available via LEMMA
```
./build/lemma_1_0_3 -h
```


If you wish to compile with the Intel MKL Library then instead run the following:
```
cmake -S . -B build \
    -DMKL_ROOT=<absolute/path/to/intel_mkl_root>
cmake --build build -- -j4
```
Note that current compile flags are compatible with the Intel MKL Library 2019 Update 1.
