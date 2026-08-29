# A Unified Evaluation Protocol for Cross-Paradigm Mobile App Review Classification

This repository contains the replication package for the study **“A Unified Evaluation Protocol for Cross-Paradigm Mobile App Review Classification.”**

The study evaluates three learning paradigms for mobile app review classification under a common experimental protocol: zero-shot inference, prompting-based classification, and parameter-efficient fine-tuning (PEFT). The package contains the datasets used in the experiments, the experimental code, prompt materials, keyword-generation resources, raw experimental outputs, and the analysis files used to prepare the reported results.

## Repository contents

```text
01_Datasets/
    CLAP/
    F-Droid/
    Pan/

02_Zero_Shot/
    code/
    outputs/

03_Prompting/
    code/
    prompt_materials/
    outputs/

04_PEFT/
    code/
    keyword_sets/
    outputs/

05_Analysis/

.env.example
requirements.txt
CITATION.cff
THIRD_PARTY_NOTICES.md
SHA256SUMS.txt
```

The same Experiment 1/2 source file is included under both `02_Zero_Shot/code/` and `03_Prompting/code/` because the original implementation uses one script for both zero-shot and prompting configurations. The file contents are identical.

## Datasets

Three existing benchmark datasets are used independently. Their original label taxonomies are retained; no cross-dataset label remapping is performed.

| Dataset | Reviews | Classes |
|---|---:|---:|
| F-Droid | 2,841 | 2 |
| CLAP | 3,000 | 7 |
| Pan | 1,390 | 4 |

The dataset files used by the experiments are provided under `01_Datasets/`. Source publications and attribution information should be consulted before reusing these datasets outside the purpose of reproducing this study.

## Experiment 1: Zero-shot classification

Experiment 1 evaluates embedding-based, NLI-based, and LLM-based zero-shot classifiers. No task-specific training is performed.

The LLM runs use:

- **GPT-4o** through the OpenAI API.
- **Llama-3.1-8B-Instant** through the Groq API.

The experimental source is:

`02_Zero_Shot/code/experiment1_zero_shot_and_experiment2_prompting.py`

Raw output workbooks are stored under `02_Zero_Shot/outputs/` and are organised by model family and model name.

## Experiment 2: Prompting

Experiment 2 evaluates GPT-4o and Llama-3.1-8B-Instant using seven prompting configurations constructed from Few-Shot, Persona, and Chain-of-Thought (CoT) components.

The exact prompt components are provided in:

`03_Prompting/prompt_materials/Prompt_Templates.txt`

The externally constructed few-shot examples are provided in:

`03_Prompting/prompt_materials/Few_Shot_Examples.docx`

Two examples per category are used in the few-shot configurations. These examples were constructed independently of the benchmark datasets.

The experimental source is:

`03_Prompting/code/experiment1_zero_shot_and_experiment2_prompting.py`

Raw output workbooks for all prompting configurations are stored under `03_Prompting/outputs/`.

## Experiment 3: PEFT

Experiment 3 adapts **Meta-LLaMA-3-8B** using four configurations:

1. LoRA
2. DoRA
3. LoRA + Compound Sparse Attention
4. DoRA + Compound Sparse Attention

The main fixed settings used in this experiment are:

| Setting | Value |
|---|---|
| Cross-validation | Stratified 10-fold |
| Validation | 10% of the original training fold |
| Oversampling | Training portion only |
| Learning rate | 5e-5 |
| Epochs | 3 |
| Train batch size | 4 |
| Evaluation batch size | 16 |
| Maximum sequence length | 256 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Warmup ratio | 0.1 |
| Early stopping | Patience = 2 |
| Random seed | 42 |
| Quantisation | 4-bit NF4 |

The source used for these experiments is:

`04_PEFT/code/experiment3_lora_dora_finetuning.py`

The test fold is not oversampled. Within each fold, the validation split is created from the original training portion before oversampling, and oversampling is then applied only to the remaining training data.

Raw fold-level predictions, classification reports, summaries, and aggregate result files are stored under `04_PEFT/outputs/`.

## Keyword extraction and Compound Sparse Attention

The keyword-generation materials are stored under `04_PEFT/keyword_sets/`:

- `keyword_extraction.py` — extraction procedure.
- `Computer_Science_D_2.bin` — external Computer Science Word2Vec resource used by the extraction script.
- `clean_keywords.csv` — final extracted keyword sets used in the experiment.

The extraction procedure follows the general methodology used by Alhoshan, Ferrari, and Zhao in *Zero-shot learning for requirements classification: An exploratory study* (Information and Software Technology, 2023), where semantically related terms are obtained from a Wikipedia-based Computer Science resource using word embeddings.

The purpose is different in this study. The extracted terms are not used as category representations for zero-shot classification. They are used as fixed external semantic signals to activate global positions in the Compound Sparse Attention mechanism. Keyword extraction is performed offline and does not use the F-Droid, CLAP, or Pan review texts.

The extraction script uses `TOP_K = 10`. After cross-class duplicate removal, the final CSV contains the exact keyword sets used by the PEFT code. The CLAP `Other` category intentionally has no keywords.

To run the extraction script, install the dependencies and the spaCy English model:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python 04_PEFT/keyword_sets/keyword_extraction.py
```

Run it from the `04_PEFT/keyword_sets/` directory so that `Computer_Science_D_2.bin` is available at the path expected by the script.

## Analysis files

The main consolidated analysis files are under `05_Analysis/`:

- `All_Results_Summary.xlsx` — consolidated performance, latency, PEFT efficiency, stability, and re-run tables.
- `Statistical_Analysis.xlsx` — performance, variability, and McNemar analysis tables.
- `Efficiency_Tables_Final.xlsx` — final efficiency tables for Experiments 1–3.
- `Validation_Results.xlsx` — independent re-run validation results.

The original experiment-level workbooks are retained under the corresponding experiment output directories so the consolidated tables can be checked against the raw results.

## Installation

A Python environment with CUDA support is recommended for local model execution and is required for the PEFT experiments.

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

On Windows, activate the environment with:

```text
.venv\Scripts\activate
```

The PEFT experiments reported in the study were executed on Lambda Labs using an NVIDIA A100-SXM4 40 GB GPU.

## API credentials

API keys are not stored in this repository. Create the required environment variables before running API-based experiments.

A template is provided in `.env.example`:

```text
OPENAI_API_KEY=
GROQ_API_KEY=
```

The code reads these values from the environment. Do not commit personal API keys to the repository.

The API-based experiments reported in the paper were executed in April 2026. Provider-hosted models can change over time, so exact future reproduction of API outputs may be affected by provider-side model updates even when the prompt and experimental settings are unchanged.

## Basic execution

### Zero-shot or prompting

Edit or provide the dataset/model configuration expected by the combined Experiment 1/2 script, set API credentials when an API model is selected, and run:

```bash
python 02_Zero_Shot/code/experiment1_zero_shot_and_experiment2_prompting.py
```

The same source file under `03_Prompting/code/` is provided for convenience when reproducing Experiment 2.

### PEFT

The PEFT script uses environment variables for the main run selection. For example:

```bash
DATASET=clap \
DATA_PATH="01_Datasets/CLAP/3000 review.csv" \
PEFT_METHOD=dora \
BASE_MODEL="meta-llama/Meta-Llama-3-8B" \
python 04_PEFT/code/experiment3_lora_dora_finetuning.py
```

Set the sparse-attention switch in the experimental configuration according to the configuration being reproduced. The source code contains the keyword sets used in the reported sparse-attention runs.

## Reproducibility notes

- The three datasets are evaluated independently.
- No preprocessing pipeline specific to one dataset is introduced.
- Prompting examples are external to the evaluation datasets.
- PEFT uses stratified 10-fold cross-validation.
- Validation and test data are fixed before training-only oversampling.
- The random seed is fixed at 42 for the reported PEFT experiments.
- The clean source files supplied here correspond to the implementations used for the reported experiments; comments and embedded credentials were removed from the public copies without changing the experimental logic.
- `SHA256SUMS.txt` can be used to check whether package files were modified after release.

## Citation

If you use this package, please cite the associated paper. The final publication details and DOI will be added after publication.

```text
K. Althobiti, "A Unified Evaluation Protocol for Cross-Paradigm Mobile App Review Classification," publication details forthcoming.
```

## Third-party materials

Some datasets and the Computer Science Word2Vec resource originate from prior research. Their inclusion here does not change the ownership or licensing of those materials. See `THIRD_PARTY_NOTICES.md` for attribution and redistribution notes.
