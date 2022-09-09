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
mpirun -n 1 build/lemma_1_0_3 \
  --gplemma --random-seed 1 \
  --pheno example/pheno.txt.gz \
  --streamBgen example/n5k_p20k_example.bgen \
  --environment example/env.txt.gz \
  --out example/gplemma.out
```
This should return heritability estimates of h2-G = 0.229 (0.028) and h2-GxE = 0.085 (0.01), where the value in brackets is the standard error.

To run the MEMMA method on the same dataset, run the following commands
```
mpirun -n 1 build/lemma_1_0_3 \
  --RHEreg --random-seed 1 \
  --pheno example/pheno.txt.gz \
  --streamBgen example/n5k_p20k_example.bgen \
  --environment example/env.txt.gz \
  --out example/gplemma.out
```
This should return heritability estimates of h2-G = 0.229 (0.028) and h2-GxE = 0.085 (0.01), where the value in brackets is the standard error.



