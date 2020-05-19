# Installation
LEMMA requires the [BGEN file format](https://bitbucket.org/gavinband/bgen/src/default/), Boost (https://www.boost.org/) and OpenMPI. We also recommend compiling with the Intel MKL library.

First clone the repository
```
git clone git@github.com:mkerin/LEMMA.git
cd LEMMA
mkdir build
```

Bare minimum build:
```
cd build
cmake .. \
-DBGEN_ROOT=<absolute/path/to/bgen_lib> \
-DBOOST_ROOT=<absolute/path/to/boost>
cd ..
cmake --build build --target lemma_1_0_1 -- -j 4
```

If you wish to compile with the Intel MKL Library then instead run the following:
```
cd build
cmake .. \
-DBGEN_ROOT=<absolute/path/to/bgen_lib> \
-DBOOST_ROOT=<pabsolute/path/to/boost> \
-DMKL_ROOT=<absolute/path/to/IntelMklRoot>
cd ..
cmake --build build --target lemma_1_0_1 -- -j 4
```
Note that current compile flags are compatible with the Intel MKL Library 2019 Update 1.
