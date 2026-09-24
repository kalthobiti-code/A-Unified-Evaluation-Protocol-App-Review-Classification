# Analysis & Results Summary

This folder contains the consolidated analyses, statistical results, qualitative error-analysis artefacts, and re-execution materials used to support the results reported in the paper.

## 1. Main Results

### `ALL_results_summary.xlsx`

Contains ten consolidated experimental result sheets used to construct the manuscript results:

| Sheet | Current Paper Table | Content |
|---|---:|---|
| T1_ZeroShot | Table 9 | Zero-Shot classification results (13 models × 3 datasets) |
| T2_Prompting | Table 10 | Prompting strategy results (7 configurations × 2 models × 3 datasets) |
| T3_PEFT | Table 11 | PEFT results (LoRA/DoRA × Sparse ON/OFF × 3 datasets) |
| T4_LLaMA_Paradigms | Table 12 | Meta-Llama-3-8B-Instruct across paradigms |
| T5_GPT_Paradigms | Table 13 | GPT-4o across Zero-Shot and Prompting |
| T6_Latency_ZSL | Table 14 | Zero-Shot inference latency and throughput |
| T7_Latency_Prompt | Table 15 | Prompting inference latency |
| T8_PEFT_Efficiency | Table 16 | PEFT computational efficiency |
| T9_PEFT_Stability | Table 18 | PEFT fold-level dispersion across 10 folds |
| T10_ReRun | Table 19 | Independent re-execution consistency |

The workbook sheet names retain their internal `T1`–`T10` identifiers; the table-number column above maps them to the current manuscript numbering.

## 2. Statistical Analysis

### `ANALYSIS.xlsx`

Contains the statistical and descriptive analyses supporting the paper's claims:

| Sheet | Content |
|---|---|
| A1 Pre-specified | Paradigm comparison using the fixed inferential configurations |
| A2 Best per dataset | Paradigm comparison using the best configuration per dataset |
| B Variability | Cross-dataset descriptive variability in Macro-F1 |
| C McNemar | Pairwise McNemar tests with global Bonferroni correction |
| D Bootstrap CI | Paired 95% bootstrap confidence intervals for Macro-F1 differences |
| E Absolute vs Relative | Absolute and relative gains between paradigms |
| F Efficiency 3D | Training cost, inference latency, and throughput |
| G Latency spread | Descriptive latency summaries across prompting configurations |
| H Interaction | GEE logistic regression and Wald test for the paradigm × dataset interaction in per-instance correctness |

For the inferential paradigm comparisons, the fixed configurations are instruction-only Zero-Shot, Few-Shot Prompting, and DoRA + Compound Sparse Attention PEFT, all using Meta-Llama-3-8B-Instruct. McNemar tests evaluate paired correctness disagreement, whereas the paired bootstrap intervals quantify differences in Macro-F1.

## 3. Error Analysis (`Error_Analysis/`)

The qualitative error analysis contains 118 coded misclassified instances sampled across the nine dataset × paradigm conditions to reach code saturation. A stratified random subsample of 30 instances (25.4%) was independently coded by a second coder. Pre-consensus agreement was 16/30 (53.3%), with Cohen's κ = 0.417. The 14 disagreements were subsequently resolved through discussion and consensus.

| File | Content |
|---|---|
| `ERROR_CODING_SHEETS_CORRECTED_NO_GUESSING.xlsx` | Error-analysis sheets and saturation audit |
| `SECOND_CODER_30.xlsx` | 30-case stratified second-coder sample |
| `ERROR_CODING_DISAGREEMENT_RESOLUTION_FINAL.xlsx` | Disagreement cases and final consensus resolution |
| `error_code.py` | Script used to generate the error-analysis sheets from result files |
| `second_coder_sample.py` | Script used to draw the stratified 30-case second-coder sample |

The reported category counts describe the sampled instances and are not estimates of category prevalence in the full error population.

## 4. Re-execution Consistency (`Reproducibility_Rerun/`)

This subfolder contains the independent re-execution materials for five representative configurations, including logs, metadata, result files, and the retrained PEFT fold outputs. The maximum observed Macro-F1 deviation was 0.0056 for GPT-4o CoT + Few-Shot on CLAP; the locally executed representative configurations reproduced their reported values exactly.

## How to reproduce these workbooks

The consolidated analyses are derived from the experiment-level result files under:

- `../02_Zero_Shot/`
- `../03_Prompting/`
- `../04_PEFT/`
- `Reproducibility_Rerun/`

Error-analysis inputs come from the `Predictions` sheets of the relevant `RESULT_*.xlsx` files and the out-of-fold PEFT prediction files.

