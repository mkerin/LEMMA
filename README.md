# LEMMA

LEMMA (**L**inear **E**nvironment **M**ixed **M**odel **A**nalysis) aims to uncover GxE interactions between SNPs and a linear combination of multiple environmental variables. To do this efficiently, LEMMA leverages an assumption that (where GxE interactions exist) SNPs will interact with a common environmental 'profile'.

## Getting Started
### Installation
LEMMA requires the [BGEN file format](https://bitbucket.org/gavinband/bgen/src/default/), Boost (https://www.boost.org/) and OpenMPI. We also recommend compiling with the Intel MKL library.

Bare minimum build:
```
git clone git@github.com:mkerin/LEMMA.git
cd LEMMA
mkdir build
cd build
cmake .. \
-DBGEN_ROOT=<path_to_bgen_lib> \
-DBOOST_ROOT=<path_to_boost>
cd ..
cmake --build build --target lemma_1_0_0 -- -j 4
```

*TODO*: Make OpenMPI optional?

### Example usage
The `example` directory contains a simulated dataset with:
- 5000 individuals
- 20,000 SNPs
- 5 environmental variables
- a simulated phenotype

The phenotype has been simulated to have:
- 2000 SNPs with non-zero main effects
- 500 SNPs with non-zero GxE effects
- GxE effects with a linear combination of two of the five environments (i.e. two are active).
- SNP-Heritability of 20% (main effects) and 10% (GxE effects)

#### Running the LEMMA variational inference algorithm
```
mpirun -n 1 build/lemma_1_0_0 \
  --VB \
  --pheno example/pheno.txt.gz \
  --environment example/env.txt.gz \
  --bgen example/n5k_p20k_example.bgen \
  --out example/inference.out.gz
```
In this case the algorithm should converge in 59 iterations.

Output files:
- `example/inference.out.gz` :                                converged hyperparameter values + ELBO
- `example/inference_converged_eta.out.gz` :                  converged Environmental Score
- `example/inference_converged_resid_pheno_chr${cc}.out.gz` : residual phenotypes for chromosomes c = 1:22
- `example/inference_converged_vparams_*.out.gz` :            variational parameters estimated by the LEMMA algorithm
- `example/inference_converged_yhat.out.gz` :                 predicted vectors and residualised phenotypes
- `example/inference_loco_pvals.out.gz` :                     single SNP hypothesis tests applied to SNPs from the file passed to `--bgen`

#### Heritability estimation
```
mpirun -n 1 build/lemma_1_0_0 \
  --RHEreg --n-RHEreg-samples 20 --n-RHEreg-jacknife 100 --random-seed 1 \
  --pheno example/pheno.txt.gz \
  --bgen example/n5k_p20k_example.bgen \
  --environment example/inference_converged_eta.out.gz \
  --out example/pve.out.gz
```
This should return heritability estimates of h2-G = 0.23 (0.032) and h2-GxE = 0.08 (0.016), where the value in brackets is the standard error.

#### Association testing with imputed SNPs
Note that genetic data is now streamed from file.
```
for cc in `seq 1 22`; do
  mpirun -n 1 build/lemma_1_0_0 \
    --singleSnpStats --maf 0.01 \
    --range ${cc}:0-1000000000000 \
    --pheno example/inference_converged_resid_pheno_chr${cc}.out.gz \
    --streamBgen example/n5k_p20k_example.bgen \
    --environment example/inference_converged_eta.out.gz \
    --out example/loco_pvals_chr${cc}.out.gz;
done
```

#### All at once
```
mpirun -n 1 build/lemma_1_0_0 \
  --pheno example/pheno.txt.gz \
  --environment example/env.txt.gz \
  --VB \
  --bgen example/n5k_p20k_example.bgen \
  --singleSnpStats \
  --RHEreg --n-RHEreg-samples 20 --n-RHEreg-jacknife 100 --random-seed 1 \
  --streamBgen example/n5k_p20k_example.bgen \
  --out example/inference.out.gz
```

## Advanced Usage

### Precomputing the dXtEEX array
Before running the variational algorithm, LEMMA requires the quantities $\sum_i X_{ij}^2 E_{il} E_{im}$ for $1 \le j \le M$ and $1 \le l \le m \le L$. LEMMA is able to compute this internally, however for large datasets this imposes substantial costs. As this is easily parallelised over variants and/or environments, we recommend that users precompute this quantity beforehand and provide a file to LEMMA at runtime.

Install `bgen_utils` using instructions from <https://github.com/mkerin/bgen_utils>.

Build the `example/dxteex.out.gz` array using commands
```
BGEN_UTILS=<path_to_bgen_utils>
for cc in `seq 1 22`; do
  ${BGEN_UTILS} \
    --compute-env-snp-correlations \
    --mode_low_mem \
    --range ${cc}:0-100000000000 \
    --bgen $(bgen) \
    --environment $(dir)/env.txt \
    --out example/dxteex_chr${cc}.out.gz;
done
zcat example/dxteex_chr*.out.gz > example/dxteex.out.gz
```
Then provide the file `example/dxteex.out.gz` to LEMMA with the commandline flag `--dxteex`.
```
mpirun -n 1 build/lemma_1_0_0 \
  --VB \
  --pheno example/pheno.txt.gz \
  --environment example/env.txt.gz \
  --bgen example/n5k_p20k_example.bgen \
  --dxteex example/dxteex.out.gz \
  --out example/inference.out.gz
```

### Heritability partitioned by MAF and LD
For this you will need:
- LD-scores (we recommend using GCTA with a window of size `--ld-wind 1000`)
- MAF of each SNP obtained from the UKBiobank MFI files.

To convert into the file format expected by LEMMA we have provided a brief Rscript `scripts/preprocess_ldms_groups.R`.

Then run the heritability analysis as follows
```
mpirun -n 1 build/lemma_1_0_0 \
  --RHEreg --n-RHEreg-samples 20 --n-RHEreg-jacknife 100 --random-seed 1 \
  --pheno example/pheno.txt.gz \
  --bgen example/n5k_p20k_example.bgen \
  --environment example/inference_converged_eta.out.gz \
  --RHEreg-groups example/ldms_groups.txt  \
  --out example/rhe_ldms.out.gz
```

### Build with the Intel MKL Library
Download the Intel MKL Library. Build with

```
cd build
cmake .. \
-DBGEN_ROOT=<path/to/bgen> \
-DBOOST_ROOT=<path/to/boost_1_55_0> \
-DMKL_ROOT=<path_to_IntelMklRoot>
cd ..
cmake --build build --target bgen_prog_0_11_6 -- -j 4
```

Note that current compile flags compatible with the Intel MKL Library 2019 Update 1.

### Resuming from a previous parameter state
In case of runtime crashes, LEMMA can save the parameter state at periodic intervals by providing the commandline flag `--resume_from_state`. LEMMA can then subsequently resume inference from this saved state. For example
```
mpirun -n 1 build/lemma_1_0_0 \
  --VB \
  --pheno example/pheno.txt.gz \
  --environment example/env.txt.gz \
  --bgen example/n5k_p20k_example.bgen \
  --param_dump_interval 10 \
  --out example/inference.out.gz

mpirun -n 1 build/lemma_1_0_0 \
  --VB \
  --pheno example/pheno.txt.gz \
  --environment example/env.txt.gz \
  --bgen example/n5k_p20k_example.bgen \
  --resume_from_state example/lemma_interim_files/inference_dump_it30 \
  --out example/inference_from_it30.out.gz

zdiff example/inference_from_it30.out.gz example/inference.out.gz
```
Outputs from the two should match, up to some small numerical difference in the ELBO. Note that if the iteration number that you start from is not a multiple of 3, then output will not match exactly because the SQUAREM algorithm adapts the trajectory of the hyperparameter updates in multiples of three.

## Other
### Complexity
LEMMA uses a iterative algorithm to approximate the posterior distribution. The per-iteration complexity is O(NM).

### RAM Usage
To store the genotype matrix, LEMMA uses approximately MN bytes of RAM where M is the number of genotyped SNPs and N is the number of samples.

To store an array of SNP-environment correlations, LEMMA uses a further 8ML(L+1)/2 bytes of RAM, where L is the number of environents and M is the number of SNPs. For M = 600k and L < 100 this should not be a dominating requirement.
