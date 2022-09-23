# GPLEMMA

## Example data
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

## Getting started
To run the GPLEMMA method on the dataset given above, run the following commands
```
mpirun build/lemma_1_0_4 \
  --gplemma --random-seed 1 \
  --pheno example/pheno.txt.gz \
  --streamBgen example/n5k_p20k_example.bgen \
  --environment example/env.txt.gz \
  --out example/gplemma.out
```
This should return heritability estimates of h2-G = 0.229 (0.028) and h2-GxE = 0.085 (0.01), where the value in brackets is the standard error.

To run the MEMMA method on the same dataset, run the following commands
```
mpirun build/lemma_1_0_4 \
  --RHEreg --random-seed 1 \
  --pheno example/pheno.txt.gz \
  --streamBgen example/n5k_p20k_example.bgen \
  --environment example/env.txt.gz \
  --out example/gplemma.out
```
This should return heritability estimates of h2-G = 0.229 (0.028) and h2-GxE = 0.085 (0.01), where the value in brackets is the standard error.

## Advanced usage

### Parallelism with OpenMPI
GPLEMMA performs parallel processing with OpenMPI, and does so using the SPMD (Single Process Multiple Data) paradigm.

More explicitly, samples are partitioned such that blocks of rows of the phenotype `y`, 
genotypes `X` and environmental variables `E` are assigned to each core. Each core then runs inference only on the locally held block of samples. At relevant points in the algorithm, cores then message summary-level statistics to each other, such that the algorithm is invariant to the number of cores; or rather, we would get the same result by loading all of the data onto only one core.

When running GPLEMMA on the UK Biobank we found parallelising with OpenMPI to highly efficient
(ie doubling the number of cores almost doubles computational speed) up to when the number of samples per core is a couple 
of thousand, after which adding extra cores yielded diminishing returns. Using OpenMPI 
has the additional advantage of allowing users to utilise cores from across a cluster 
rather than being restricted to a single node.

To set the number of cores on the commandline explicitly, use
```
mpirun -n <cores> build/lemma_1_0_4
```

