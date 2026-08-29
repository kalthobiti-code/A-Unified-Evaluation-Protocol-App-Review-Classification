# Keyword-generation resources

This directory contains the resources used to generate the fixed keyword sets for Compound Sparse Attention.

`keyword_extraction.py` loads `Computer_Science_D_2.bin`, retrieves semantically related terms for the predefined category seeds, applies lexical/POS filtering, removes duplicates across classes, and writes `clean_keywords.csv`.

The final CSV is the output used to define the keyword sets in the PEFT experiment code. The extraction uses only the external Computer Science Word2Vec resource; it does not inspect the benchmark review text.

The extraction idea is adapted from the external-domain word-embedding methodology described by Alhoshan, Ferrari, and Zhao (2023), but the extracted terms are used here as attention-control signals rather than as zero-shot category representations.
