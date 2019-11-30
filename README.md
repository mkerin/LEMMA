TODO - add Eigen3 files to git


# LEMMA

LEMMA (**L**inear **E**nvironment **M**ixed **M**odel **A**nalysis) aims to uncover GxE interactions between SNPs and a linear combination of multiple environmental variables. To do this efficiently, LEMMA leverages an assumption that, where GxE interactions exist, SNPs will interact with a common environmetal 'profile'.

## Complexity
LEMMA uses a iterative algorithm to approximate the posterior distribution. The per-iteration complexity is O(NM).

## RAM
To store the genotype matrix, LEMMA uses approximately MN bytes of RAM where M is the number of genotyped SNPs and N is the number of samples.

To store an array of SNP-environment correlations, LEMMA uses a further 8ML(L+1)/2 bytes of RAM, where L is the number of environents and M is the number of SNPs. For M = 600k and L < 100 this should not be a dominating requirement.

# Installation
LEMMA requires BGEN file format, Boost and OpenMPI. We also strongly recommend running with OpenMPI and the Intel MKL library.

Bare minimum build:
```
git clone XXX
cd LEMMA
cmake -DCMAKE_BUILD_TYPE="Release" \
-DMKL_ROOT=<path/to/intel_mkl_root> \
-DBGEN_ROOT=<path/to/bgen> \
-DBOOST_ROOT=<path/to/boost_1_55_0>
```

# Example

```
lemma \
--mode_vb \
--bgen ${simDIR}/n5k_p20k_example.bgen \
--pheno \
--environment \
--out

```