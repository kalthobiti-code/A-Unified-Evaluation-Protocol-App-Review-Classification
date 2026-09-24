# =========================================================
# Experiment 3 - Parameter-Efficient Fine-Tuning (PEFT)
# ---------------------------------------------------------
# Backbone: Meta-Llama-3-8B-Instruct (default; open-weight,
#           common checkpoint aligned with Experiments 1 & 2).
# Pipeline: 10-fold stratified CV + oversampling + LoRA/DoRA
#           with optional Compound Sparse Attention.
# Override the backbone via:
#   BASE_MODEL=<hf_model_id> python <this file>
# =========================================================

# =========================
# 0) Install if needed
# =========================
# !pip install -q -U transformers datasets peft accelerate bitsandbytes scikit-learn openpyxl psutil pynvml

import os
import re
import gc
import json
import time
import math
import shutil
import zipfile
import logging
import threading
import psutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from sklearn.utils import resample

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed
)
from peft import LoraConfig, get_peft_model

# =========================
# 1) Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("unified_finetuning")

# =========================
# 2) Unified Config
# =========================
SEED = 42
set_seed(SEED)

# -------- Dataset --------
DATASET = os.getenv("DATASET", "f-droid").lower()   # clap | pan | f-droid
# Fixed dataset -> file mapping (so DATASET alone selects the right file).
# DATA_PATH env still overrides if explicitly provided.
_DATA_FILES = {
    "clap":    "3000 review.csv",
    "pan":     "Pan Dataset.xlsx",
    "f-droid": "HLT_2841_Crashes_FeatureBugs.xlsx",
}
DATA_PATH = os.getenv("DATA_PATH", _DATA_FILES.get(DATASET, "HLT_2841_Crashes_FeatureBugs.xlsx"))

# -------- Model --------
BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
USE_4BIT = True

# -------- PEFT --------
PEFT_METHOD = os.getenv("PEFT_METHOD", "lora").lower()   # lora | dora
assert PEFT_METHOD in ["lora", "dora"]

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# -------- Sparse Attention --------
# NOTE: Set to True only after filling KEYWORD_SETS below with actual keywords.
# When False, standard attention is used and no sparse columns are added to datasets.
COMPOUND_SPARSE_ON = os.getenv("COMPOUND_SPARSE_ON", "0") == "1"
BAND_WIDTH = 32

# -------- Cross Validation --------
N_SPLITS = 10
VALID_RATIO = 0.1   # validation split from ORIGINAL (pre-oversampling) train fold only
APPLY_OVERSAMPLING = True

# -------- Training --------
MAX_LENGTH = 256
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 16
GRAD_ACCUM = 1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1           # FIX: added warmup for stable training
EARLY_STOPPING_PATIENCE = 2

# -------- Outputs --------
# Output folder is unique per (method, sparse, dataset) to prevent overwriting
# when the runner iterates over configurations.
_sparse_tag = "sparse-ON" if COMPOUND_SPARSE_ON else "sparse-OFF"
OUTPUT_ROOT = Path(f"finetune_outputs_{PEFT_METHOD}_{_sparse_tag}_{DATASET}")
FOLDS_DIR = OUTPUT_ROOT / "fold_files"
RESULTS_DIR = OUTPUT_ROOT / "results"
MODELS_DIR = OUTPUT_ROOT / "models"
ZIP_PATH = OUTPUT_ROOT / f"{DATASET}_10fold_files.zip"

for p in [OUTPUT_ROOT, FOLDS_DIR, RESULTS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# =========================
# 3) Dataset-specific labels
# =========================
if DATASET == "clap":
    LABELS = ["BUG", "FEATURE", "PERFORMANCE", "SECURITY", "ENERGY", "USABILITY", "OTHER"]
elif DATASET == "pan":
    LABELS = ["feature request", "information giving", "problem discovery", "information seeking"]
elif DATASET == "f-droid":
    LABELS = ["CRASHES", "FEATURE & UI BUGS"]
else:
    raise ValueError("DATASET must be one of: clap | pan | f-droid")

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

# =========================
# 4) Keywords scaffold
# Fill KEYWORD_SETS before setting COMPOUND_SPARSE_ON = True
# Fixed across experiments, external to datasets
# =========================
KEYWORD_SETS = {
    "clap": {
        # Keys must match LABELS exactly (uppercase)
        "BUG": [
            "fix", "fixing", "flaw", "patched", "patch",
            "unwanted", "vulnerability", "bounty", "releasing", "maintainer"
        ],
        "FEATURE": [
            "functionality", "enhancement", "customization", "customizable",
            "plugins", "option", "configurable", "enhanced", "capability", "custom"
        ],
        "PERFORMANCE": [
            "efficiency", "scalability", "reliability", "responsiveness",
            "improvement", "improving", "workload", "speed", "improve", "improves"
        ],
        "SECURITY": [
            "countermeasure", "protection", "protecting", "assurance",
            "privacy", "authentication", "securing", "hardening", "protect"
        ],
        "ENERGY": [
            "consumption", "capacity", "battery", "latency", "utilization",
            "limit", "cooling", "increase", "superconducting", "increased"
        ],
        "USABILITY": [
            "accessibility", "aesthetic", "experience", "contextual",
            "satisfaction", "sus", "ease", "walkthrough", "inquiry", "prototyping"
        ],
        "OTHER": []
    },
    "pan": {
        # Keys must match LABELS exactly (lowercase)
        "feature request": [
            "delete", "functionality", "customization", "customizable",
            "option", "configurable", "custom", "adding", "customized", "preview"
        ],
        "information giving": [
            "describing", "abstract", "formal", "syntactic", "rewriting",
            "specification", "described", "appendix", "describes", "defining"
        ],
        "problem discovery": [
            "defect", "detected", "deadlock", "guarantee", "faulty",
            "tolerance", "flaw", "vulnerability", "incorrect", "correct"
        ],
        "information seeking": [
            "answer", "answered", "answering", "possibility", "fact",
            "reply", "opinion", "subgoal", "moral", "asks"
        ]
    },
    "f-droid": {
        # Keys must match LABELS exactly (uppercase)
        "CRASHES": [
            "prevented", "corruption", "prevents", "causing", "escalation",
            "caused", "prevent", "preventing", "detected", "standby"
        ],
        "FEATURE & UI BUGS": [
            "customizable", "zooming", "graphical", "widget", "toolkits",
            "functionality", "customization", "tweak", "api", "skinning"
        ]
    }
}

# Validate: if sparse is ON, at least one keyword must exist
if COMPOUND_SPARSE_ON:
    total_kw = sum(len(v) for cats in KEYWORD_SETS.values() for v in cats.values())
    if total_kw == 0:
        raise ValueError(
            "COMPOUND_SPARSE_ON is True but KEYWORD_SETS are all empty. "
            "Fill keywords first or set COMPOUND_SPARSE_ON = False."
        )

# =========================
# 5) Flexible column detection
# =========================
TEXT_CANDS  = ["review", "text", "comment", "content", "body", "message", "reviews"]
LABEL_CANDS = ["category", "label", "class", "type", "gold", "target", "classification", "actual"]

def detect_columns(df: pd.DataFrame):
    cols = list(df.columns)
    lower_map = {str(c).lower().strip(): c for c in cols}

    text_col = None
    for c in TEXT_CANDS:
        if c in lower_map:
            text_col = lower_map[c]
            break
    if text_col is None:
        obj_cols = [c for c in cols if df[c].dtype == object]
        text_col = obj_cols[0] if obj_cols else cols[0]

    label_col = None
    for c in LABEL_CANDS:
        if c in lower_map:
            label_col = lower_map[c]
            break
    if label_col is None and len(cols) > 1:
        label_col = cols[1]

    return text_col, label_col

# =========================
# 6) Read dataset
# =========================
def read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)

raw = read_any(DATA_PATH)
TEXT_COL, LABEL_COL = detect_columns(raw)

if TEXT_COL not in raw.columns:
    raise ValueError(f"Text column not found. Detected: {TEXT_COL}")
if LABEL_COL not in raw.columns:
    raise ValueError(f"Label column not found. Detected: {LABEL_COL}")

log.info("Loaded dataset '%s' with %d rows", DATASET, len(raw))
log.info("Detected text column: %s | label column: %s", TEXT_COL, LABEL_COL)

# =========================
# 7) Normalize labels
# =========================
def normalize_label(x):
    if pd.isna(x):
        return None

    if DATASET == "f-droid":
        if isinstance(x, (int, np.integer, float, np.floating)):
            if int(x) == 1:
                return "CRASHES"
            if int(x) == 2:
                return "FEATURE & UI BUGS"
        s = str(x).strip().upper()
        if s in ["1", "CRASHES", "CRASH", "BUG"]:
            return "CRASHES"
        if s in ["2", "FEATURE & UI BUGS", "FEATURE", "UI BUG", "FEATURE BUG"]:
            return "FEATURE & UI BUGS"
        return s

    if DATASET == "clap":
        s = str(x).strip().upper()
        if s.startswith("BUG"):      return "BUG"
        if s.startswith("FEAT"):     return "FEATURE"
        if s.startswith("PERF"):     return "PERFORMANCE"
        if s.startswith("SEC"):      return "SECURITY"
        if s.startswith("ENE") or s.startswith("BAT"): return "ENERGY"
        if s.startswith("USA"):      return "USABILITY"
        if s.startswith("OTH"):      return "OTHER"
        return s

    if DATASET == "pan":
        s = str(x).strip().lower()
        if s.startswith("feature"):           return "feature request"
        if s.startswith("information giving"): return "information giving"
        if s.startswith("problem"):           return "problem discovery"
        if s.startswith("information seeking"): return "information seeking"
        return s

    return str(x).strip()

df = pd.DataFrame({
    "text":  raw[TEXT_COL].astype(str).fillna(""),
    "label": raw[LABEL_COL].apply(normalize_label)
})

df = df[df["label"].isin(LABELS)].reset_index(drop=True)

log.info("After label normalization: %d rows", len(df))
log.info("Class distribution:\n%s", df["label"].value_counts())

# =========================
# 8) Build folds + oversampling
# FIX: Save TWO files per fold:
#   fold{n}_train_original.csv  -> pre-oversampling (used for validation split)
#   fold{n}_train_oversampled.csv -> post-oversampling (used for actual training)
# This prevents oversampled duplicates from leaking into validation.
# =========================
def oversample_train(train_df: pd.DataFrame, label_col: str = "label", random_state: int = 42):
    counts = train_df[label_col].value_counts()
    max_count = counts.max()

    oversampled_parts = []
    for label, group in train_df.groupby(label_col):
        if len(group) < max_count:
            group = resample(
                group,
                replace=True,
                n_samples=max_count,
                random_state=random_state
            )
        oversampled_parts.append(group)

    train_os = pd.concat(oversampled_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train_os

def make_10fold_files(df: pd.DataFrame):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_meta = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(df["text"], df["label"]), start=1):
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df  = df.iloc[test_idx].copy().reset_index(drop=True)

        # Overlap check (informational only)
        train_texts = set(train_df["text"])
        test_texts  = set(test_df["text"])
        overlap = len(train_texts.intersection(test_texts))
        if overlap > 0:
            log.warning("Fold %d has %d duplicated reviews between train and test", fold, overlap)

        # FIX: Save original BEFORE oversampling
        original_path    = FOLDS_DIR / f"fold{fold}_train_original.csv"
        train_df.to_csv(original_path, index=False, encoding="utf-8")

        # Oversampling on train only -> save separately
        if APPLY_OVERSAMPLING:
            # FIX: use fold-specific random_state for diversity across folds
            train_os = oversample_train(train_df, label_col="label", random_state=SEED + fold)
        else:
            train_os = train_df.copy()

        oversampled_path = FOLDS_DIR / f"fold{fold}_train_oversampled.csv"
        test_path        = FOLDS_DIR / f"fold{fold}_test.csv"

        train_os.to_csv(oversampled_path, index=False, encoding="utf-8")
        test_df.to_csv(test_path, index=False, encoding="utf-8")

        fold_meta.append({
            "fold": fold,
            "train_original": len(train_df),
            "train_oversampled": len(train_os),
            "test_size": len(test_df),
            "text_overlap_count": overlap
        })

        log.info(
            "Fold %d saved | original=%d | oversampled=%d | test=%d",
            fold, len(train_df), len(train_os), len(test_df)
        )

    return pd.DataFrame(fold_meta)

fold_summary = make_10fold_files(df)
fold_summary_path = RESULTS_DIR / "fold_summary.xlsx"
fold_summary.to_excel(fold_summary_path, index=False)

# Zip fold files
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for file in FOLDS_DIR.glob("*.csv"):
        zf.write(file, arcname=file.name)

log.info("Fold files zipped: %s", ZIP_PATH)

try:
    from google.colab import files
    files.download(str(ZIP_PATH))
    log.info("Triggered Colab download for fold files ZIP")
except Exception:
    log.info("Not running in Colab; ZIP available locally: %s", ZIP_PATH)

# =========================
# 9) Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# 10) Compound sparse attention (REAL implementation)
#
# Logic (matches paper Section C):
#   - Keywords found  -> global attention for keyword tokens
#                        + local banded attention (BAND_WIDTH) for all others
#   - No keywords     -> standard full attention (fallback)
#
# Implementation:
#   - During tokenization : detect keyword positions, store as list
#   - During collation    : build 4D additive attention bias per sample
#   - SparseTrainer       : inject 4D mask into every forward pass
#
# 4D mask format (additive bias, transformers convention):
#   shape  : (batch, 1, seq_len, seq_len)
#   attend : 0.0
#   block  : -1e4  (large negative -> ~0 after softmax)
# =========================

# ---------- keyword detection ----------
def find_keyword_positions(text: str, token_ids, tokenizer, dataset_name: str):
    flat_keywords = set()
    for kws in KEYWORD_SETS[dataset_name].values():
        for kw in kws:
            if isinstance(kw, str) and kw.strip():
                flat_keywords.add(kw.strip().lower())

    if not flat_keywords:
        return []

    decoded = [re.sub(r"\s+", " ", tokenizer.decode([tid])).strip().lower()
               for tid in token_ids]
    return [i for i, tok in enumerate(decoded) if tok in flat_keywords]


# ---------- tokenization: store sparse metadata ----------
def apply_compound_sparse_metadata(batch_encodings, raw_texts, dataset_name):
    """
    Adds two fields per sample:
      sparse_flag       : 1 if keywords found, 0 -> fallback to full attention
      keyword_positions : list of matched token indices (variable length)
    """
    sparse_flags          = []
    keyword_positions_all = []

    for ids, txt in zip(batch_encodings["input_ids"], raw_texts):
        positions = find_keyword_positions(txt, ids, tokenizer, dataset_name)
        sparse_flags.append(1 if positions else 0)
        keyword_positions_all.append(positions)

    batch_encodings["sparse_flag"]       = sparse_flags
    batch_encodings["keyword_positions"] = keyword_positions_all
    return batch_encodings


# ---------- build 4D attention mask for one sample ----------
def build_sparse_mask_single(seq_len: int,
                              keyword_positions: list,
                              sparse_flag: int,
                              band_width: int = BAND_WIDTH) -> torch.Tensor:
    """
    Returns float mask of shape (1, seq_len, seq_len).
      0.0  = token pair CAN attend
     -1e4  = token pair is BLOCKED
    """
    if sparse_flag == 0 or len(keyword_positions) == 0:
        # Fallback: standard full attention (every token attends to every token)
        return torch.zeros(1, seq_len, seq_len, dtype=torch.float32)

    # Start with local banded attention (block everything outside the band)
    mask = torch.full((seq_len, seq_len), -1e4, dtype=torch.float32)
    for i in range(seq_len):
        lo = max(0, i - band_width // 2)
        hi = min(seq_len, i + band_width // 2 + 1)
        mask[i, lo:hi] = 0.0

    # Global attention for keyword positions:
    #   keyword token -> attends to ALL tokens   (row = 0.0)
    #   ALL tokens    -> attend to keyword token (col = 0.0)
    for kp in keyword_positions:
        if kp < seq_len:
            mask[kp, :] = 0.0   # keyword attends to all
            mask[:, kp] = 0.0   # all attend to keyword

    return mask.unsqueeze(0)    # (1, seq_len, seq_len)


# ---------- custom data collator ----------
class SparseAttentionCollator(DataCollatorWithPadding):
    """
    Extends DataCollatorWithPadding to build 4D attention masks
    when COMPOUND_SPARSE_ON is True.

    Output batch contains:
      input_ids, attention_mask, labels   (standard)
      sparse_attention_bias               (4D float tensor, only when sparse ON)
    """

    def __call__(self, features):
        # Pop sparse metadata BEFORE standard collation
        # (DataCollatorWithPadding errors on non-tensor / variable-length fields)
        sparse_flags      = [f.pop("sparse_flag",       0)  for f in features]
        keyword_positions = [f.pop("keyword_positions", []) for f in features]

        # Standard padding
        batch = super().__call__(features)

        if not COMPOUND_SPARSE_ON:
            return batch

        # Build 4D bias: (batch, 1, seq_len, seq_len)
        seq_len   = batch["input_ids"].shape[1]
        bias_list = []
        for i in range(len(features)):
            bias_list.append(
                build_sparse_mask_single(
                    seq_len=seq_len,
                    keyword_positions=keyword_positions[i],
                    sparse_flag=sparse_flags[i],
                    band_width=BAND_WIDTH
                )
            )

        # Stack -> (batch, 1, seq_len, seq_len)
        batch["sparse_attention_bias"] = torch.stack(bias_list, dim=0)
        return batch


# ---------- custom Trainer ----------
class SparseTrainer(Trainer):
    """
    Injects sparse_attention_bias into the model forward pass as an
    additive attention bias. When COMPOUND_SPARSE_ON is False or no
    bias is present, behaves exactly like the standard Trainer.

    Fallback guarantee:
      sparse_flag == 0  -> build_sparse_mask_single returns all-zeros mask
                        -> zero additive bias = no change to attention scores
                        -> identical to standard full attention
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sparse_bias = inputs.pop("sparse_attention_bias", None)

        if COMPOUND_SPARSE_ON and sparse_bias is not None:
            device      = next(model.parameters()).device
            sparse_bias = sparse_bias.to(device)   # (batch, 1, seq_len, seq_len)

            # ── Attention bias injection via forward hooks ──────────────────
            # We keep attention_mask as the standard 2D (batch, seq_len).
            # A hook on each attention layer adds sparse_bias to the raw
            # attention scores BEFORE softmax:
            #   scores = scores + sparse_bias   (broadcast over heads)
            # Since all-zeros bias = no change, sparse_flag==0 samples are
            # unaffected → guaranteed fallback to standard full attention.
            hooks = []

            def make_hook(bias):
                def hook_fn(module, args, kwargs_fwd):
                    # LlamaAttention forward signature (transformers >= 4.36):
                    # args: (hidden_states,)
                    # kwargs: {attention_mask, position_ids, ...}
                    # We inject into kwargs["attention_mask"] as a 4D additive bias.
                    # LLaMA internally checks: if mask.dim() == 4 → treat as additive bias.
                    if "attention_mask" in kwargs_fwd:
                        existing = kwargs_fwd["attention_mask"]
                        if existing is not None and existing.dim() == 4:
                            # Already a 4D causal mask produced by the model's
                            # _prepare_4d_causal_attention_mask — add our bias on top
                            kwargs_fwd["attention_mask"] = existing + bias
                        else:
                            # No 4D mask yet; set ours directly
                            kwargs_fwd["attention_mask"] = bias
                    return args, kwargs_fwd
                return hook_fn

            # Register hook on every LlamaAttention layer
            for module in model.modules():
                if module.__class__.__name__ in ("LlamaAttention",
                                                  "LlamaSdpaAttention",
                                                  "LlamaFlashAttention2"):
                    hooks.append(
                        module.register_forward_pre_hook(make_hook(sparse_bias),
                                                         with_kwargs=True)
                    )

            try:
                # Standard forward: attention_mask stays 2D as LLaMA expects
                outputs = model(**inputs)
            finally:
                # Always remove hooks to avoid affecting subsequent batches
                for h in hooks:
                    h.remove()

            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        # Standard path: sparse OFF or no bias present
        return super().compute_loss(model, inputs,
                                    return_outputs=return_outputs, **kwargs)

# =========================
# 11) Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _         = precision_recall_fscore_support(labels, preds, average="macro",    zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy":           acc,
        "precision_macro":    p_macro,
        "recall_macro":       r_macro,
        "f1_macro":           f1_macro,
        "precision_weighted": p_weighted,
        "recall_weighted":    r_weighted,
        "f1_weighted":        f1_weighted
    }

# =========================
# 12) Model builder
# FIX: Changed bnb_4bit_compute_dtype from float16 to bfloat16
#      to avoid FP16 gradient unscaling error with LLaMA-3
# =========================
def build_model():
    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,      # FIX: was torch.float16
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,                     # FIX: was torch.float16
        device_map="auto"
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_dora=(PEFT_METHOD == "dora")
    )

    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model

# =========================
# 13) Efficiency Metrics
# Covers: training time, computational cost, resource utilization
# For fine-tuning approaches per Section B of the study plan.
# =========================

def get_trainable_params(model):
    """Computational cost: count trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total

def get_resource_snapshot():
    """Snapshot of GPU and CPU memory at a point in time."""
    info = {
        "cuda_available":         torch.cuda.is_available(),
        "gpu_name":               None,
        "gpu_mem_allocated_mb":   None,
        "gpu_mem_reserved_mb":    None,
        "cpu_count":              os.cpu_count(),
        "cpu_mem_used_mb":        round(psutil.virtual_memory().used / 1024**2, 2),
        "cpu_mem_percent":        psutil.virtual_memory().percent
    }
    if torch.cuda.is_available():
        info["gpu_name"]             = torch.cuda.get_device_name(0)
        info["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 1024**2, 2)
        info["gpu_mem_reserved_mb"]  = round(torch.cuda.memory_reserved(0)  / 1024**2, 2)
    return info

def get_peak_gpu_memory_mb():
    """Peak GPU memory during training (computational cost)."""
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated(0) / 1024**2, 2)
    return None

class ResourceMonitor:
    """
    Background thread that samples GPU utilization % and CPU utilization %
    every `interval` seconds during training.
    Resource utilization metric per Section B.
    """
    def __init__(self, interval=2.0):
        self.interval      = interval
        self.gpu_util_samples  = []
        self.cpu_util_samples  = []
        self._stop_event   = threading.Event()
        self._thread       = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop_event.is_set():
            # CPU utilization %
            self.cpu_util_samples.append(psutil.cpu_percent(interval=None))
            # GPU utilization % via nvidia-smi through pynvml if available
            if torch.cuda.is_available():
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_util_samples.append(util.gpu)
                except Exception:
                    self.gpu_util_samples.append(None)
            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def summary(self):
        gpu_vals = [v for v in self.gpu_util_samples if v is not None]
        cpu_vals = self.cpu_util_samples
        return {
            "gpu_util_mean_pct":  round(sum(gpu_vals) / len(gpu_vals), 2) if gpu_vals else None,
            "gpu_util_max_pct":   max(gpu_vals) if gpu_vals else None,
            "cpu_util_mean_pct":  round(sum(cpu_vals) / len(cpu_vals), 2) if cpu_vals else None,
            "cpu_util_max_pct":   max(cpu_vals) if cpu_vals else None,
        }

def measure_inference_time(trainer, ds_test):
    """
    Computational cost: measure inference (prediction) time on test set.
    Returns total seconds and per-sample milliseconds.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    preds_output = trainer.predict(ds_test)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    total_sec     = round(t1 - t0, 4)
    per_sample_ms = round((total_sec / len(ds_test)) * 1000, 4)
    return preds_output, total_sec, per_sample_ms

# =========================
# 14) Prepare fold data
# FIX: Validation split taken from _original_ (pre-oversampling) file.
#      Oversampled file used only for training.
#      This eliminates oversampled duplicates leaking into validation.
# =========================
def tokenize_examples(examples):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )
    if COMPOUND_SPARSE_ON:
        # Collect keyword metadata; collator builds the 4D mask at batch time
        enc = apply_compound_sparse_metadata(enc, examples["text"], DATASET)
    return enc

def prepare_fold_datasets(fold: int):
    original_csv    = FOLDS_DIR / f"fold{fold}_train_original.csv"
    oversampled_csv = FOLDS_DIR / f"fold{fold}_train_oversampled.csv"
    test_csv        = FOLDS_DIR / f"fold{fold}_test.csv"

    original_df  = pd.read_csv(original_csv)
    oversamp_df  = pd.read_csv(oversampled_csv)
    test_df      = pd.read_csv(test_csv)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VALID_RATIO, random_state=SEED)
    idx_train_orig, idx_val = next(splitter.split(original_df["text"], original_df["label"]))

    val_df   = original_df.iloc[idx_val].reset_index(drop=True)
    train_df = oversamp_df.copy().reset_index(drop=True)

    train_df["label_id"] = train_df["label"].map(label2id)
    val_df["label_id"]   = val_df["label"].map(label2id)
    test_df["label_id"]  = test_df["label"].map(label2id)

    ds_train = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    ds_val   = Dataset.from_pandas(val_df[["text",  "label_id"]].rename(columns={"label_id": "labels"}))
    ds_test  = Dataset.from_pandas(test_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))

    ds_train = ds_train.map(tokenize_examples, batched=True)
    ds_val   = ds_val.map(tokenize_examples,   batched=True)
    ds_test  = ds_test.map(tokenize_examples,  batched=True)

    # Sparse metadata (sparse_flag, keyword_positions) must reach the collator.
    # Using with_format(..., output_all_columns=True) keeps torch tensors on the
    # standard columns while preserving the sparse metadata as Python objects
    # so the SparseAttentionCollator can pop them and build the 4D bias.
    if COMPOUND_SPARSE_ON:
        ds_train = ds_train.with_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
            output_all_columns=True,
        )
        ds_val   = ds_val.with_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
            output_all_columns=True,
        )
        ds_test  = ds_test.with_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
            output_all_columns=True,
        )
    else:
        ds_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        ds_val.set_format(  type="torch", columns=["input_ids", "attention_mask", "labels"])
        ds_test.set_format( type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_df, val_df, test_df, ds_train, ds_val, ds_test

# =========================
# 15) Train across 10 folds
# =========================
all_fold_rows        = []
all_predictions_rows = []

# Use SparseAttentionCollator always; it handles both ON and OFF modes
data_collator = SparseAttentionCollator(tokenizer=tokenizer)

for fold in range(1, N_SPLITS + 1):
    log.info("=" * 80)
    log.info("Starting Fold %d", fold)

    train_df, val_df, test_df, ds_train, ds_val, ds_test = prepare_fold_datasets(fold)

    model = build_model()

    fold_out_dir = MODELS_DIR / f"fold{fold}_{PEFT_METHOD}"
    fold_out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(fold_out_dir),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=max(1, int(len(ds_train) / (TRAIN_BATCH_SIZE * GRAD_ACCUM) * NUM_EPOCHS * WARMUP_RATIO)),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        fp16=False,                                     # FIX: disabled fp16
        bf16=torch.cuda.is_available()                  # FIX: enabled bf16 instead
    )

    # Use SparseTrainer: handles 4D mask injection and fallback transparently
    trainer = SparseTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )

    resource_before = get_resource_snapshot()

    # Trainable parameters (computational cost)
    trainable_params, total_params = get_trainable_params(model)

    # Reset peak GPU memory counter before training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    # Start background resource monitor (resource utilization)
    monitor = ResourceMonitor(interval=2.0)
    monitor.start()

    # FIX: synchronize GPU timer for accurate measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_start = time.time()
    trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_end = time.time()

    training_time_sec = train_end - train_start

    # Stop monitor and collect utilization stats
    monitor.stop()
    util_stats    = monitor.summary()
    peak_gpu_mb   = get_peak_gpu_memory_mb()
    resource_after = get_resource_snapshot()

    # Inference time measurement (computational cost)
    preds_output, inference_time_sec, inference_ms_per_sample = measure_inference_time(trainer, ds_test)
    logits = preds_output.predictions
    y_true = test_df["label_id"].to_numpy()
    y_pred = np.argmax(logits, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _         = precision_recall_fscore_support(y_true, y_pred, average="macro",    zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    report_dict = classification_report(
        y_true, y_pred,
        target_names=LABELS,
        zero_division=0,
        output_dict=True
    )
    report_df   = pd.DataFrame(report_dict).transpose()
    report_path = RESULTS_DIR / f"fold{fold}_classification_report.xlsx"
    report_df.to_excel(report_path)

    pred_df = test_df.copy()
    pred_df["pred_id"]    = y_pred
    pred_df["pred_label"] = [id2label[i] for i in y_pred]
    pred_df["true_label"] = [id2label[i] for i in y_true]
    pred_df["correct"]    = (pred_df["pred_label"] == pred_df["true_label"]).astype(int)
    pred_path = RESULTS_DIR / f"fold{fold}_predictions.xlsx"
    pred_df.to_excel(pred_path, index=False)

    row = {
        # ── Identity ──────────────────────────────────────────────────
        "fold":                           fold,
        "peft_method":                    PEFT_METHOD,
        "compound_sparse_on":             COMPOUND_SPARSE_ON,
        "train_size_oversampled":         len(train_df),
        "val_size_original":              len(val_df),
        "test_size":                      len(test_df),
        # ── Performance metrics ───────────────────────────────────────
        "accuracy":                       acc,
        "macro_f1":                       f1_macro,
        "weighted_f1":                    f1_weighted,
        "precision_macro":                p_macro,
        "recall_macro":                   r_macro,
        "precision_weighted":             p_weighted,
        "recall_weighted":                r_weighted,
        # ── Efficiency: Training time ─────────────────────────────────
        "training_time_sec":              training_time_sec,
        # ── Efficiency: Computational cost ────────────────────────────
        "trainable_params":               trainable_params,
        "total_params":                   total_params,
        "trainable_params_pct":           round(trainable_params / total_params * 100, 4),
        "peak_gpu_mem_mb":                peak_gpu_mb,
        "inference_time_sec":             inference_time_sec,
        "inference_ms_per_sample":        inference_ms_per_sample,
        # ── Efficiency: Resource utilization ─────────────────────────
        "gpu_name":                       resource_before["gpu_name"],
        "gpu_util_mean_pct":              util_stats["gpu_util_mean_pct"],
        "gpu_util_max_pct":               util_stats["gpu_util_max_pct"],
        "cpu_util_mean_pct":              util_stats["cpu_util_mean_pct"],
        "cpu_util_max_pct":               util_stats["cpu_util_max_pct"],
        "gpu_mem_allocated_before_mb":    resource_before["gpu_mem_allocated_mb"],
        "gpu_mem_reserved_before_mb":     resource_before["gpu_mem_reserved_mb"],
        "gpu_mem_allocated_after_mb":     resource_after["gpu_mem_allocated_mb"],
        "gpu_mem_reserved_after_mb":      resource_after["gpu_mem_reserved_mb"],
        "cpu_mem_used_after_mb":          resource_after["cpu_mem_used_mb"],
        "cpu_mem_percent_after":          resource_after["cpu_mem_percent"],
        "cpu_count":                      resource_after["cpu_count"]
    }
    all_fold_rows.append(row)

    pred_df["fold"] = fold
    all_predictions_rows.append(pred_df)

    log.info(
        "Fold %d finished | Acc=%.4f | Macro-F1=%.4f | Weighted-F1=%.4f | TrainTime=%.2fs",
        fold, acc, f1_macro, f1_weighted, training_time_sec
    )

    # FIX: Clean up all heavy objects to prevent memory accumulation
    del trainer, model
    del ds_train, ds_val, ds_test
    del train_df, val_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================
# 16) Save final summaries
# =========================
summary_df = pd.DataFrame(all_fold_rows)
summary_df.to_excel(RESULTS_DIR / "all_folds_summary.xlsx", index=False)

all_preds_df = pd.concat(all_predictions_rows, ignore_index=True)
all_preds_df.to_excel(RESULTS_DIR / "all_folds_predictions.xlsx", index=False)

mean_row = {
    "accuracy_mean":                  summary_df["accuracy"].mean(),
    "macro_f1_mean":                  summary_df["macro_f1"].mean(),
    "weighted_f1_mean":               summary_df["weighted_f1"].mean(),
    # Efficiency: training time
    "training_time_sec_mean":         summary_df["training_time_sec"].mean(),
    "training_time_sec_total":        summary_df["training_time_sec"].sum(),
    # Efficiency: computational cost
    "trainable_params":               summary_df["trainable_params"].iloc[0],
    "total_params":                   summary_df["total_params"].iloc[0],
    "trainable_params_pct":           summary_df["trainable_params_pct"].iloc[0],
    "peak_gpu_mem_mb_mean":           summary_df["peak_gpu_mem_mb"].mean(),
    "inference_time_sec_mean":        summary_df["inference_time_sec"].mean(),
    "inference_ms_per_sample_mean":   summary_df["inference_ms_per_sample"].mean(),
    # Efficiency: resource utilization
    "gpu_util_mean_pct":              summary_df["gpu_util_mean_pct"].mean(),
    "gpu_util_max_pct":               summary_df["gpu_util_max_pct"].max(),
    "cpu_util_mean_pct":              summary_df["cpu_util_mean_pct"].mean(),
    "cpu_util_max_pct":               summary_df["cpu_util_max_pct"].max(),
    "gpu_mem_allocated_after_mb_mean":summary_df["gpu_mem_allocated_after_mb"].mean(),
    "peak_gpu_mem_mb_max":            summary_df["peak_gpu_mem_mb"].max(),
}
mean_df = pd.DataFrame([mean_row])
mean_df.to_excel(RESULTS_DIR / "overall_mean_metrics.xlsx", index=False)

log.info("Saved final summaries to: %s", RESULTS_DIR)
print("\n✅ ALL DONE")
print(f"Fold files ZIP: {ZIP_PATH}")
print(f"Results folder: {RESULTS_DIR}")