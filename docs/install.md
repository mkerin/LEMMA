# Installation
LEMMA requires the [BGEN file format](http://www.bgenformat.org/), [Boost](https://www.boost.org/) and OpenMPI. We also recommend compiling with the Intel MKL library.

First clone the [GitHub repository](https://github.com/mkerin/LEMMA)
```
git clone https://github.com/mkerin/LEMMA.git
cd LEMMA
```

Bare minimum build:
```
cmake -S . -B build \
    -DBGEN_ROOT=<absolute/path/to/bgen_lib> \
    -DBOOST_ROOT=<absolute/path/to/boost>
cmake --build build -- -j4
```
This should create a new directory `build` which contains some auto-generated files by CMake and the lemma executable.

If you wish to compile with the Intel MKL Library then instead run the following:
```
cmake -S . -B build \
    -DBGEN_ROOT=<absolute/path/to/bgen_lib> \
    -DBOOST_ROOT=<absolute/path/to/boost> \
    -DMKL_ROOT=<absolute/path/to/intel_mkl_root>
cmake --build build -- -j4
```
Note that current compile flags are compatible with the Intel MKL Library 2019 Update 1.
