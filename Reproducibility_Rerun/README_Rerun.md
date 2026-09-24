# Independent Re-Run Validation (Run 2)

This folder contains a **single independent re-execution** of five representative configurations, one per family/paradigm, to verify the reproducibility of the reported results.

## Method

The same code, seed (42), environment, and hyperparameters were used. For PEFT, the model was retrained from scratch. For API-based models (GPT-4o), calls used the same `gpt-4o` alias (no dated snapshot was pinned).

## Configurations Re-Run (5 total)

| # | Paradigm             | Configuration                     | Dataset  |
|---|----------------------|-----------------------------------|----------|
| 1 | Exp1 — Embedding     | DistilBERT-base-uncased           | F-Droid  |
| 2 | Exp1 — NLI           | RoBERTa-large-MNLI                | F-Droid  |
| 3 | Exp1 — LLM           | GPT-4o (zero-shot)                | Pan      |
| 4 | Exp2 — Prompting     | GPT-4o + CoT + Few-Shot           | CLAP     |
| 5 | Exp3 — PEFT          | DoRA + Compound Sparse Attention  | F-Droid  |

## Results (see paper Table 19)

| # | Config              | Run 1 MF1 | Run 2 MF1 | Δ MF1  |
|---|---------------------|-----------|-----------|--------|
| 1 | DistilBERT / F-Droid| 0.6777    | 0.6777    | 0.0000 |
| 2 | RoBERTa-MNLI / F-Droid | 0.6471 | 0.6471    | 0.0000 |
| 3 | GPT-4o / Pan        | 0.7044    | 0.7044    | 0.0000 |
| 4 | GPT-4o+CoT+FS / CLAP| 0.7675    | 0.7731    | +0.0056|
| 5 | DoRA+Sparse / F-Droid | 0.9154  | 0.9154    | 0.0000 |

**Maximum observed absolute deviation in Macro-F1 = 0.0056** (GPT-4o + CoT + FS on CLAP). Because the `gpt-4o` alias was used without a dated snapshot, exact long-term reproduction of the API-based results cannot be guaranteed; the observed difference should not be attributed to a specific cause from this re-execution alone.

## Access Dates

- **Zero-Shot (local models):** 16 Sep 2026
- **GPT-4o (Pan ZS):** 16 Sep 2026
- **GPT-4o (CLAP CoT+FS):** 16 Sep 2026
- **PEFT (DoRA + Sparse ON, F-Droid):** 16 Sep 2026

## Environment

- **Hardware:** NVIDIA A100-SXM4-40GB, Driver 580.105.08
- **Software:** Python 3.10.12, torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0, accelerate 1.15.0, CUDA 13.0
