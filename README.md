[![build-and-test Actions Status](https://github.com/mkerin/LEMMA/workflows/build-and-test/badge.svg)](https://github.com/mkerin/LEMMA/actions)

# Overview

This repository provides software for the following two methods:

- LEMMA (**L**inear **E**nvironment **M**ixed **M**odel **A**nalysis) is a whole genome wide regression method for flexible modeling of gene-environment interactions in large datasets such as the UK Biobank.  
- GPLEMMA (**G**aussian **P**rior **L**inear **E**nvironment **M**ixed **M**odel **A**nalysis) non-linear randomized Haseman-Elston regression method for flexible modeling of gene-environment interactions in large datasets such as the UK Biobank.

For documentation please see the following webpage: [https://mkerin.github.io/LEMMA/](https://mkerin.github.io/LEMMA/)

TODOs before the next release:
- Look for a CMake module for Intel MKL
- How does CTest work? Can I link in the bgen library unit tests and call it from the commandline?
- Make sure users compile cmake with release flags (check these are appropriate)
- Check I'm happy with the docs
- bgen_utils is a super weird hidden dependancy; should at least be part of this repo. What are the advantages of a separate executable vs having a new option in the LEMMA executable? I think separate executable... should this have it's own sub folder?

## Citation

If you use **LEMMA** in your research, please cite the following publication:

*Matthew Kerin and Jonathan Marchini (2020) Inferring Gene-by-Environment Interactions with a Bayesian Whole-Genome Regression Model* [[AJHG](https://doi.org/10.1016/j.ajhg.2020.08.009)]

If you use **GPLEMMA** in your research, please cite the following publication:

*Matthew Kerin and Jonathan Marchini (2020) Non-linear randomized Haseman-Elston regression for estimation of gene-environment heritability* [[Bioinformatics](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btaa1079/6050717)][[bioRxiv](https://www.biorxiv.org/content/10.1101/2020.05.18.098459v1)]
