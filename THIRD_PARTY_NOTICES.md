# Third-Party Materials

This replication package includes materials originating from previously published research. These materials remain subject to their original ownership and licensing terms.

## Benchmark datasets

The F-Droid, CLAP, and Pan datasets were obtained from the corresponding benchmark-study materials and are included to support reproduction of the experiments. Their original publications should be cited when the datasets are reused.

The package includes the dataset files used by the experimental code, but does not redistribute copies of the benchmark papers themselves.

## Computer Science Word2Vec resource

`04_PEFT/keyword_sets/Computer_Science_D_2.bin` is an external Computer Science Word2Vec resource obtained from research materials associated with:

W. Alhoshan, A. Ferrari, and L. Zhao, “Zero-shot learning for requirements classification: An exploratory study,” *Information and Software Technology*, vol. 159, 107202, 2023. https://doi.org/10.1016/j.infsof.2023.107202

That article describes word-embedding-generated terms learned from Wikipedia pages belonging to the Computer Science portal. The present study reuses the general keyword-extraction idea for a different purpose: the extracted words act as fixed external signals for keyword-activated global attention in Compound Sparse Attention.

The Alhoshan et al. article is published under CC BY 4.0. The article also points to its public replication materials. The licensing status of individual external binary/data artifacts should be checked against the source repository before redistribution outside this replication package.

## Repository licence

No licence in this repository should be interpreted as relicensing third-party datasets or external model resources. Their original terms take precedence.
