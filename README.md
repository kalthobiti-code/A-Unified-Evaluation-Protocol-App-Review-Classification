# A Unified Evaluation Protocol for Cross-Paradigm Mobile App Review Classification

This repository is the replication package for the study **"A Unified
Evaluation Protocol for Cross-Paradigm Mobile App Review
Classification."**

The study evaluates three learning paradigms for mobile app review
classification under a common experimental protocol: (1) zero-shot
inference, (2) prompting-based classification, and (3)
parameter-efficient fine-tuning (PEFT). The package contains the
datasets, the experimental code, prompt materials, keyword-generation
resources, raw experimental outputs, analysis files, and independent
re-run validation results.

------------------------------------------------------------------------

## Repository Structure

``` text
Replication_Package_v1.0.2/
├── README.md
├── .env.example
├── requirements.txt
├── CITATION.cff
├── LICENSE
├── THIRD_PARTY_NOTICES.md
│
├── 01_Datasets/
│   ├── F-Droid/
│   ├── CLAP/
│   └── Pan/
│
├── 02_Zero_Shot/
│   ├── code/
│   │   └── Experiments_1_and_2_ZeroShot_and_Prompting.py
│   └── outputs/
│       ├── Embedding/
│       ├── NLI/
│       └── LLM/
│
├── 03_Prompting/
│   ├── code/
│   │   └── Experiments_1_and_2_ZeroShot_and_Prompting.py
│   ├── prompt_materials/
│   │   ├── Prompt_Templates.txt
│   │   └── Few_Shot_Examples.docx
│   └── outputs/
│       ├── GPT/
│       └── LLAMA/
│
├── 04_PEFT/
│   ├── code/
│   │   └── Experiment_3_PEFT_FineTuning.py
│   ├── keyword_sets/
│   │   ├── keyword_extraction.py
│   │   ├── clean_keywords.csv
│   │   ├── Computer_Science_D_2.bin
│   │   └── README.md
│   └── outputs/
│       ├── only lora/
│       ├── only Dora/
│       ├── Lora with spars on/
│       └── Dora with spars on/
│
├── 05_Analysis/
│   ├── README_Analysis.md
│   ├── ALL_results_summary.xlsx
│   ├── ANALYSIS.xlsx
│   └── Error_Analysis/
│
└── Reproducibility_Rerun/
    ├── README_Rerun.md
    ├── rerun_metadata.txt
    ├── logs/
    └── results/
```

------------------------------------------------------------------------

## Datasets

Three benchmark datasets are used independently. Original label
taxonomies are retained; no cross-dataset label remapping is performed.

  Dataset     Reviews   Classes
  --------- --------- ---------
  F-Droid       2,841         2
  CLAP          3,000         7
  Pan           1,390         4

Source publications and attribution should be consulted before reusing
these datasets outside the purpose of reproducing this study.

------------------------------------------------------------------------

## Experiment 1: Zero-Shot Classification

Experiment 1 evaluates 13 models across three families with no
task-specific training:

-   **Embedding (7):** ALBERT-base-v2, all-MPNet-base-v2,
    BERT-base-uncased, DistilBERT-base-uncased, MiniLM
    (all-MiniLM-L6-v2), RoBERTa-base, XLNet-base-cased
-   **NLI (4):** Cross-Encoder DeBERTa-v3-base, DeBERTa-v3-large
    (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli),
    RoBERTa-large-MNLI, BART-large-MNLI
-   **LLM (2):** Meta-Llama-3-8B-Instruct (local, HuggingFace), GPT-4o
    (OpenAI API)

**Total: 13 × 3 = 39 runs.**

Source:
`02_Zero_Shot/code/Experiments_1_and_2_ZeroShot_and_Prompting.py` Raw
outputs: `02_Zero_Shot/outputs/{Embedding,NLI,LLM}/`

------------------------------------------------------------------------

## Experiment 2: Prompting

Experiment 2 evaluates **Meta-Llama-3-8B-Instruct** and **GPT-4o** using
seven prompting configurations constructed from Few-Shot, Persona, and
Chain-of-Thought (CoT) components:

1.  Few-Shot
2.  Persona + Zero-Shot
3.  Persona + Few-Shot
4.  CoT + Zero-Shot
5.  CoT + Few-Shot
6.  Persona + CoT + Zero-Shot
7.  Persona + CoT + Few-Shot

**Total: 2 × 7 × 3 = 42 runs.**

The exact prompt components are provided in
`03_Prompting/prompt_materials/Prompt_Templates.txt`. The externally
constructed few-shot examples (two per category, independent of the
benchmark datasets) are in
`03_Prompting/prompt_materials/Few_Shot_Examples.docx`.

Source: same file as Experiment 1. Raw outputs:
`03_Prompting/outputs/{GPT,LLAMA}/`

------------------------------------------------------------------------

## Experiment 3: PEFT

Experiment 3 adapts **Meta-Llama-3-8B-Instruct** using four
configurations:

1.  LoRA (Sparse OFF)
2.  DoRA (Sparse OFF)
3.  LoRA + Compound Sparse Attention
4.  DoRA + Compound Sparse Attention

**Total: 4 × 3 = 12 runs, each with stratified 10-fold
cross-validation.**

Fixed settings:

  Setting                   Value
  ------------------------- -------------------------------------
  Base model                meta-llama/Meta-Llama-3-8B-Instruct
  Cross-validation          Stratified 10-fold
  Validation split          10% of the original training fold
  Oversampling              Training portion only
  Learning rate             5e-5
  Epochs                    3
  Train batch size          4
  Evaluation batch size     16
  Maximum sequence length   256
  LoRA rank                 8
  LoRA alpha                16
  LoRA dropout              0.05
  Warmup                    10% of total training steps
  Early stopping            Patience = 2
  Random seed               42
  Quantisation              4-bit NF4

**Notes:** - The test fold is not oversampled. - Within each fold, the
validation split is created from the original training portion *before*
oversampling; oversampling is then applied only to the remaining
training data. - Sparse Attention is reported as a **secondary
ablation** rather than a central contribution.

Source: `04_PEFT/code/Experiment_3_PEFT_FineTuning.py` Raw outputs:
`04_PEFT/outputs/{only lora, only Dora, Lora with spars on, Dora with spars on}/`

### Keyword Extraction (for Compound Sparse Attention)

Keyword-generation materials are provided in `04_PEFT/keyword_sets/`:

-   `keyword_sets/keyword_extraction.py` --- extraction script
-   `keyword_sets/Computer_Science_D_2.bin` --- external Computer
    Science Word2Vec resource
-   `keyword_sets/clean_keywords.csv` --- final extracted keyword sets
    used by the PEFT code

The extraction procedure adapts the general methodology of Alhoshan,
Ferrari, and Zhao (2023). The extracted terms are **not** used as
category representations for zero-shot classification; they act as fixed
external signals to activate global positions in Compound Sparse
Attention. Extraction is performed offline and does **not** use the
F-Droid, CLAP, or Pan review texts.

To reproduce keyword extraction:

``` bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd 04_PEFT/keyword_sets
python keyword_extraction.py
```

------------------------------------------------------------------------

## Reproducibility Re-Run

To validate reproducibility, five representative configurations (one per
family/paradigm) were independently re-executed:

  \#   Paradigm             Config                    Dataset
  ---- -------------------- ------------------------- ---------
  1    Exp1 --- Embedding   DistilBERT                F-Droid
  2    Exp1 --- NLI         RoBERTa-large-MNLI        F-Droid
  3    Exp1 --- LLM         GPT-4o (Zero-Shot)        Pan
  4    Exp2 --- Prompting   GPT-4o + CoT + Few-Shot   CLAP
  5    Exp3 --- PEFT        DoRA + Sparse ON          F-Droid

**Maximum observed absolute deviation in Macro-F1 = 0.0056** (GPT-4o
CoT+FS on CLAP). Because the `gpt-4o` alias was used without a dated
snapshot, exact long-term reproduction of API-based results cannot be
guaranteed. The locally executed representative configurations
reproduced their reported values exactly in the re-execution.

Full results, logs, metadata, and PEFT retrained fold data are in
`Reproducibility_Rerun/`.

------------------------------------------------------------------------

## Installation

A Python environment with CUDA support is required for local model
execution and for the PEFT experiments.

``` bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The reported experiments were executed on **Lambda Labs, NVIDIA
A100-SXM4 40 GB**, with the following environment:

-   Python 3.10.12
-   torch 2.14.0+cu130
-   transformers 5.17.0
-   peft 0.20.0
-   accelerate 1.15.0
-   CUDA 13.0

------------------------------------------------------------------------

## API Credentials

API keys are **not stored** in this repository. Create the required
environment variables before running API-based or gated-model
experiments.

A template is provided in `.env.example`:

    OPENAI_API_KEY=       # required for GPT-4o
    HF_TOKEN=             # required for Meta-Llama-3-8B-Instruct (HuggingFace gated)

The code reads these values from the environment. Do **not** commit
personal API keys to the repository.

**GPT-4o access dates for the reported experiments:** - Zero-Shot
GPT-4o: 12 September 2026 - Prompting GPT-4o: 13 September 2026 - Re-run
GPT-4o (Pan ZS): 16 September 2026 06:23:50 UTC - Re-run GPT-4o (CLAP
CoT+FS): 16 September 2026 06:55:47 UTC

Since a dated snapshot was not pinned (the default `gpt-4o` alias was
used), exact long-term reproducibility of GPT-4o results cannot be
guaranteed.

------------------------------------------------------------------------

## Execution

The replication package provides the experiment scripts and the reported
raw outputs. It does not include a single automation wrapper for all 93
reported runs. Experiments should be configured and executed using the
corresponding scripts and configuration variables documented in the
package.

### Zero-Shot and Prompting

Source script:

``` text
02_Zero_Shot/code/Experiments_1_and_2_ZeroShot_and_Prompting.py
```

The script contains the model, dataset, and prompting configuration used
for the reported Zero-Shot and Prompting experiments. The exact prompt
materials are provided under `03_Prompting/prompt_materials/`.

### PEFT

Source script:

``` text
04_PEFT/code/Experiment_3_PEFT_FineTuning.py
```

The reported PEFT configurations use LoRA or DoRA with Compound Sparse
Attention enabled or disabled. Each reported configuration is evaluated
using stratified 10-fold cross-validation.

Because dataset locations and experiment selections are
configuration-dependent, verify the dataset path and selected
configuration in the relevant script before execution. The raw outputs
supplied in the package provide the reported results for verification.

------------------------------------------------------------------------

## Analysis Files

Consolidated analysis is provided under `05_Analysis/`:

-   `ALL_results_summary.xlsx` --- consolidated experimental result
    tables used to construct the manuscript results.
-   `ANALYSIS.xlsx` --- statistical analyses, including McNemar tests,
    paired bootstrap confidence intervals, the GEE paradigm--dataset
    interaction analysis, and efficiency/latency analyses.
-   `Error_Analysis/` --- qualitative error-analysis materials,
    including the coded sample, second-coder materials, disagreement
    resolution, and analysis scripts.
-   `Reproducibility_Rerun/` --- independent re-execution materials,
    logs, metadata, and outputs for the five representative
    configurations reported in the manuscript.

The qualitative error analysis contains 118 coded instances. A
stratified 30-instance subsample (25.4%) was independently double-coded;
the pre-consensus agreement was 53.3% and Cohen's κ was 0.417.
Disagreements were resolved by discussion before the final coding was
reported.

Original experiment-level outputs are retained in the experiment folders
so that the consolidated results can be checked against the raw outputs.

------------------------------------------------------------------------

## Reproducibility Notes

-   The three datasets are evaluated independently; no cross-dataset
    label remapping.
-   The **same instruction-tuned checkpoint (Meta-Llama-3-8B-Instruct)**
    is used across all three paradigms to eliminate the checkpoint-level
    confound of prior work.
-   Prompting few-shot examples are external to the evaluation datasets.
-   PEFT uses stratified 10-fold cross-validation; validation and test
    data are fixed before training-only oversampling.
-   The random seed is fixed at 42 for the reported PEFT experiments.
-   Comments and embedded credentials were removed from the public
    copies of the code without changing the experimental logic.
-   The `Reproducibility_Rerun/` directory documents an independent
    re-execution; the maximum observed Macro-F1 deviation among the
    representative configurations was 0.0056.

------------------------------------------------------------------------

## Citation

If you use this package, please cite the associated paper. Publication
venue and article DOI will be added after publication.

    K. Althobiti and H. A. A. Al-Hashimi, "A Unified Evaluation Protocol for Cross-Paradigm Mobile App Review Classification," publication details forthcoming.

------------------------------------------------------------------------

## Third-Party Materials

Some datasets, the Meta-Llama-3-8B-Instruct weights, GPT-4o, and the
Computer Science Word2Vec resource originate from prior or proprietary
sources. Their inclusion here does not change the ownership or licensing
of those materials. Redistribution rights for third-party resources,
including `Computer_Science_D_2.bin`, must be verified before public
release. See `THIRD_PARTY_NOTICES.md`.
