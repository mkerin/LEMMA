# LEMMA

LEMMA (**L**inear **E**nvironment **M**ixed **M**odel **A**nalysis) aims to uncover GxE interactions between SNPs and a linear combination of multiple environmental variables. To do this efficiently, LEMMA leverages an assumption that, where GxE interactions exist, SNPs will interact with a common environmetal 'profile'. 

## Complexity
LEMMA uses a iterative algorithm to approximate a posterior distribution. The per-iteration complexity is O(NM) and we can expect the number of iterations required for convergence to scale as O(N^{1.5}). Hence LEMMA has complexity O(N^{1.5}M).


## RAM
LEMMA uses approximately MN bytes of RAM where M is the number of genotyped SNPs and N is the number of samples. 

Currently LEMMA also uses a further 8ML(L+1)/2 bytes of RAM to store an array of SNP-environment correlations, where L is the number of environents and M is the number of SNPs. For M = 600k and L < 100 this should not be a dominating requirement. _If this presents a problem then it might be possible to halve memory usage by using floats rather than doubles for this specific array_.