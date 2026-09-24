# Third-Party Materials

This replication package includes materials originating from previously published research. These materials remain subject to their original ownership and licensing terms.

## Benchmark datasets

The F-Droid, CLAP, and Pan datasets were obtained from the corresponding benchmark-study materials and are included to support reproduction of the experiments. Their original publications should be cited when the datasets are reused.

The package includes the dataset files used by the experimental code, but does not redistribute copies of the benchmark papers themselves.

## Meta-Llama-3-8B-Instruct

The Meta-Llama-3-8B-Instruct model weights are not included in this package. The weights are obtained from HuggingFace at runtime using the `HF_TOKEN` credential. Meta's Llama 3 Community License applies to any redistribution, fine-tuning, or downstream use of these weights.

## Computer Science Word2Vec resource

`04_PEFT/Computer_Science_D_2.bin` is an external Computer Science Word2Vec resource obtained from research materials associated with:

W. Alhoshan, A. Ferrari, and L. Zhao, "Zero-shot learning for requirements classification: An exploratory study," *Information and Software Technology*, vol. 159, 107202, 2023. https://doi.org/10.1016/j.infsof.2023.107202

That article describes word-embedding-generated terms learned from Wikipedia pages belonging to the Computer Science portal. The present study reuses the general keyword-extraction idea for a different purpose: the extracted words act as fixed external signals for keyword-activated global attention in Compound Sparse Attention.

The Alhoshan et al. article is published under CC BY 4.0. This does not by itself establish redistribution rights for the separate `Computer_Science_D_2.bin` binary resource. Its licensing status should therefore be verified against the source repository or other authoritative licensing information before the binary is included in a public GitHub or Zenodo release. If redistribution rights cannot be verified, the binary should be omitted and users should be directed to obtain the resource from its original source.

## GPT-4o (OpenAI API)

GPT-4o is a proprietary model accessed via the OpenAI API. It is not redistributed in any form. Users must obtain their own `OPENAI_API_KEY`. Because a dated snapshot was not pinned, exact numerical reproduction of GPT-4o results long-term cannot be guaranteed due to potential provider-side model updates.

## Repository content

No entry in this repository should be interpreted as relicensing third-party datasets, models, or external resources. Their original terms take precedence.
