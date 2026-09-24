# ================================================================
# Experiments 1 & 2 - Zero-Shot Classification and Prompt-Based Learning
# ----------------------------------------------------------------
# Supports 13 models across 3 families:
#   - Embedding-based  (7 models: BERT, RoBERTa, DistilBERT, MiniLM, ...)
#   - NLI-based        (4 models: DeBERTa, BART-MNLI, RoBERTa-MNLI, ...)
#   - Prompt-based LLM (2 models: local Meta-Llama-3-8B-Instruct,
#                       OpenAI GPT-4o)
#
# Usage:
#   MODEL_NAME=<model-name> DATASET=<clap|pan|f-droid> python <this file>
# ================================================================

import os, json, time, re, random, logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, classification_report,
    accuracy_score, precision_score, recall_score
)

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ================================================================
# CONFIG — 
# ================================================================
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama-3-8b-instruct")
DATASET    = os.getenv("DATASET",    "clap").lower()
# "clap" | "pan" | "f-droid"
#: "review.csv" | "Pan Dataset.xlsx" | "HLT_2841_Crashes_FeatureBugs.xlsx"
BATCH      = int(os.getenv("BATCH",  1))

FEWSHOT_ON = os.getenv("FEWSHOT_ON", "1") == "1"
PERSONA_ON = os.getenv("PERSONA_ON", "1") == "1"
COT_ON     = os.getenv("COT_ON",     "1") == "1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN       = os.getenv("HF_TOKEN",       "")

# ================================================================
# MODEL REGISTRY
# ================================================================
MODEL_REGISTRY = {
    "albert-base-v2":          {"hf_id": "albert-base-v2",                                            "family": "embedding", "style": "mean_pool"},
    "all-mpnet-base-v2":       {"hf_id": "sentence-transformers/all-mpnet-base-v2",                   "family": "embedding", "style": "sbert"},
    "bert-base-uncased":       {"hf_id": "bert-base-uncased",                                         "family": "embedding", "style": "mean_pool"},
    "distilbert-base-uncased": {"hf_id": "distilbert-base-uncased",                                   "family": "embedding", "style": "mean_pool"},
    "minilm":                  {"hf_id": "sentence-transformers/all-MiniLM-L6-v2",                    "family": "embedding", "style": "sbert"},
    "roberta-base":            {"hf_id": "roberta-base",                                              "family": "embedding", "style": "mean_pool"},
    "xlnet-base-cased":        {"hf_id": "xlnet-base-cased",                                          "family": "embedding", "style": "mean_pool"},
    "cross-encoder-deberta":   {"hf_id": "cross-encoder/nli-deberta-v3-base",                        "family": "nli"},
    "deberta-v3-large":        {"hf_id": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "family": "nli"},
    "roberta-large-mnli":      {"hf_id": "roberta-large-mnli",                                       "family": "nli"},
    "bart-large-mnli":         {"hf_id": "facebook/bart-large-mnli",                                 "family": "nli"},
    "meta-llama-3-8b-instruct": {"hf_id": "meta-llama/Meta-Llama-3-8B-Instruct", "family": "llm", "provider": "local"},
    "gpt-4o":                   {"hf_id": None,                                       "family": "llm", "provider": "openai"},
}

# ================================================================
# VALIDATION
# ================================================================
if MODEL_NAME not in MODEL_REGISTRY:
    raise ValueError(f"Unknown model '{MODEL_NAME}'.\nAvailable: {list(MODEL_REGISTRY.keys())}")

if DATASET not in ("clap", "pan", "f-droid"):
    raise ValueError(f"Unknown dataset '{DATASET}'. Choose: clap | pan | f-droid")

MODEL_CFG = MODEL_REGISTRY[MODEL_NAME]
FAMILY    = MODEL_CFG["family"]
HF_ID     = MODEL_CFG.get("hf_id")

if FAMILY == "llm":
    provider = MODEL_CFG["provider"]
    if provider == "openai" and not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required for GPT models but not set")
    if provider == "local" and not HF_TOKEN:
        log.warning("HF_TOKEN is not set. Meta-Llama-3-8B-Instruct is a gated model on HuggingFace and requires an access token.")

log.info("Model: %s | Family: %s | Dataset: %s", MODEL_NAME, FAMILY, DATASET)

# ================================================================
# LABELS
# ================================================================
if DATASET == "clap":
    LABELS = ["BUG", "FEATURE", "PERFORMANCE", "SECURITY", "ENERGY", "USABILITY", "OTHER"]
elif DATASET == "pan":
    LABELS = ["feature request", "information giving", "problem discovery", "information seeking"]
elif DATASET == "f-droid":
    LABELS = ["CRASHES", "FEATURE & UI BUGS"]

LABEL_DESCRIPTIONS = {
    "BUG":                "The app has a bug or error that causes incorrect behavior.",
    "FEATURE":            "The user requests a new feature or improvement.",
    "PERFORMANCE":        "The app is slow, laggy, or consumes too many resources.",
    "SECURITY":           "The app has a security or privacy issue.",
    "ENERGY":             "The app drains battery or causes the device to overheat.",
    "USABILITY":          "The app interface is confusing or hard to use.",
    "OTHER":              "The review does not fit any specific category.",
    "feature request":    "The user is requesting a new feature or enhancement.",
    "information giving": "The user is sharing information or experience about the app.",
    "problem discovery":  "The user discovered a problem or issue in the app.",
    "information seeking":"The user is asking a question or seeking help.",
    "CRASHES":            "The app crashes or stops working unexpectedly.",
    "FEATURE & UI BUGS":  "The app has UI bugs or missing feature functionality.",
}

# ================================================================
# PATHS & DATA LOADING
# ================================================================
DATASET_CONFIG = {
    "clap":    {"path": "3000 review.csv",                   "text_col": "review", "label_col": "category",       "label_map": {}},
    "pan":     {"path": "Pan Dataset.xlsx",                  "text_col": "review", "label_col": "class",          "label_map": {}},
    "f-droid": {"path": "HLT_2841_Crashes_FeatureBugs.xlsx", "text_col": "review", "label_col": "classification", "label_map": {"1": "CRASHES", "2": "FEATURE & UI BUGS"}},
}

def read_any(path):
    return pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)

_cfg      = DATASET_CONFIG[DATASET]
raw       = read_any(_cfg["path"])
TEXT_COL  = _cfg["text_col"]
LABEL_COL = _cfg["label_col"]
LABEL_MAP = _cfg["label_map"]

if TEXT_COL not in raw.columns:
    raise ValueError(f"TEXT_COL '{TEXT_COL}' not found. Available: {list(raw.columns)}")

if LABEL_COL not in raw.columns:
    log.warning("LABEL_COL '%s' not found — metrics will NOT be computed.", LABEL_COL)
    LABEL_COL = None

DF         = pd.DataFrame({"text": raw[TEXT_COL].astype(str)})
DF["gold"] = raw[LABEL_COL].astype(str) if LABEL_COL else ""

if LABEL_COL and LABEL_MAP:
    before     = DF["gold"].unique().tolist()
    DF["gold"] = DF["gold"].map(LABEL_MAP).fillna(DF["gold"])
    after      = DF["gold"].unique().tolist()
    unmapped   = [v for v in after if v not in LABELS]
    if unmapped:
        log.warning("Unmapped gold labels: %s — add to DATASET_CONFIG[%s]['label_map']", unmapped, DATASET)
    else:
        log.info("Gold label mapping applied: %s -> %s", before, after)

log.info("Loaded '%s' — %d rows", DATASET, len(DF))

# ================================================================
# FAMILY 1: EMBEDDING-BASED
# ================================================================
def load_embedding_model():
    style = MODEL_CFG.get("style", "mean_pool")
    if style == "sbert":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(HF_ID)
        log.info("Loaded SentenceTransformer: %s", HF_ID)
        return ("sbert", model)
    else:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(HF_ID)
        model     = AutoModel.from_pretrained(HF_ID)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        log.info("Loaded AutoModel (mean_pool): %s", HF_ID)
        return ("mean_pool", tokenizer, model)

def mean_pool(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded    = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

def get_embeddings_mean_pool(texts, tokenizer, model):
    device = next(model.parameters()).device
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    emb = mean_pool(output, inputs["attention_mask"])
    return F.normalize(emb, p=2, dim=1).cpu().numpy()

def run_embedding(texts):
    loaded = load_embedding_model()
    if loaded[0] == "sbert":
        _, sbert_model = loaded
        label_texts = [LABEL_DESCRIPTIONS.get(l, l) for l in LABELS]
        label_emb   = sbert_model.encode(label_texts, normalize_embeddings=True, show_progress_bar=False)
        preds = []
        for i in range(0, len(texts), BATCH):
            batch     = texts[i:i + BATCH]
            batch_emb = sbert_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            scores    = batch_emb @ label_emb.T
            preds.extend([LABELS[idx] for idx in scores.argmax(axis=1)])
        return preds
    else:
        _, tokenizer, model = loaded
        label_texts = [LABEL_DESCRIPTIONS.get(l, l) for l in LABELS]
        label_emb   = get_embeddings_mean_pool(label_texts, tokenizer, model)
        preds = []
        for i in range(0, len(texts), BATCH):
            batch     = texts[i:i + BATCH]
            batch_emb = get_embeddings_mean_pool(batch, tokenizer, model)
            scores    = batch_emb @ label_emb.T
            preds.extend([LABELS[idx] for idx in scores.argmax(axis=1)])
            log.info("%d/%d done", min(i + BATCH, len(texts)), len(texts))
        return preds

# ================================================================
# FAMILY 2: NLI-BASED
# ================================================================
def run_nli(texts):
    from transformers import pipeline
    classifier = pipeline(
        "zero-shot-classification",
        model=HF_ID,
        device=0 if torch.cuda.is_available() else -1,
    )
    log.info("Loaded NLI pipeline: %s", HF_ID)
    candidate_labels = [LABEL_DESCRIPTIONS.get(l, l) for l in LABELS]
    preds = []
    for i in range(0, len(texts), BATCH):
        batch   = texts[i:i + BATCH]
        results = classifier(batch, candidate_labels=candidate_labels, multi_label=False)
        for res in results:
            top_desc = res["labels"][0]
            idx      = candidate_labels.index(top_desc)
            preds.append(LABELS[idx])
        log.info("%d/%d done", min(i + BATCH, len(texts)), len(texts))
    return preds

# ================================================================
# FAMILY 3: PROMPT-BASED LLM
# ================================================================
FEWSHOT_EXAMPLES = {
    "clap": [
        ("The app crashes immediately after I tap the login button.", "BUG"),
        ("The application stops working when I try to upload a file.", "BUG"),
        ("Please add a dark mode option for better usability at night.", "FEATURE"),
        ("It would be useful to have an option to export reports as PDF.", "FEATURE"),
        ("The app takes more than 10 seconds to open on my device.", "PERFORMANCE"),
        ("Scrolling through the list is very slow and unresponsive.", "PERFORMANCE"),
        ("There is no two-factor authentication to secure user accounts.", "SECURITY"),
        ("The app does not seem to protect sensitive user information properly.", "SECURITY"),
        ("The app consumes a lot of battery even when used briefly.", "ENERGY"),
        ("My phone overheats quickly when this app is running.", "ENERGY"),
        ("The navigation menu is poorly organized and difficult to understand.", "USABILITY"),
        ("Important options are hard to find due to unclear interface design.", "USABILITY"),
        ("The app works fine overall with no major issues.", "OTHER"),
        ("This is a decent application, but nothing stands out.", "OTHER"),
    ],
    "pan": [
        ("It would be helpful to add a search function to quickly find content.", "feature request"),
        ("Please include an option to customize notifications.", "feature request"),
        ("I use this app daily to track my tasks and it works well.", "information giving"),
        ("The latest update improved the overall stability of the app.", "information giving"),
        ("The app fails to load data when the internet connection is weak.", "problem discovery"),
        ("I cannot save my progress after completing a task.", "problem discovery"),
        ("How can I reset my password within the app?", "information seeking"),
        ("Is there a way to sync this app with other devices?", "information seeking"),
    ],
    "f-droid": [
        ("The app crashes every time I try to open it after installation.", "CRASHES"),
        ("It keeps crashing when I switch between different screens.", "CRASHES"),
        ("The button layout is broken and overlaps with other elements.", "FEATURE & UI BUGS"),
        ("Some interface elements are not displayed correctly on my screen.", "FEATURE & UI BUGS"),
    ],
}

PERSONA_TEXT = (
    "You are an AI system tasked with classifying app reviews into predefined categories. "
    "Assign exactly one category to each review based solely on its content and the provided category definitions."
)

COT_TEXT = (
    "Classify the following review into exactly one of the predefined categories. "
    "Consider the category definitions carefully before making the final decision. "
    "Output only the final category label."
)

def build_prompt(reviews):
    n      = len(reviews)
    prompt = ""
    if PERSONA_ON:
        prompt += PERSONA_TEXT + "\n\n"
    if FEWSHOT_ON:
        for text, label in FEWSHOT_EXAMPLES[DATASET]:
            prompt += f"Review: {text}\nLabel: {label}\n\n"
    if COT_ON:
        prompt += COT_TEXT + "\n\n"
    prompt += (
        f"You will classify EXACTLY {n} app reviews.\n"
        f"Return ONLY JSON: {{\"labels\":[...]}}.\n"
        f"Labels: {', '.join(LABELS)}\n"
        f"Input:\n{json.dumps(reviews, ensure_ascii=False)}"
    )
    return prompt

def safe_parse(content):
    try:
        result = json.loads(content)["labels"]
        # flatten 
        flat = []
        for item in result:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        result = flat
        # normalize 
        result = [
            item["label"]  if isinstance(item, dict) and "label"  in item else
            item["labels"] if isinstance(item, dict) and "labels" in item else
            str(item)
            for item in result
        ]
        return result
    except Exception:
        pass
    match = re.search(r'"labels"\s*:\s*\[([^\]]+)\]', content, re.IGNORECASE)
    if match:
        items = re.findall(r'"([^"]+)"', match.group(1))
        if items:
            log.warning("JSON fallback — recovered %d labels via regex", len(items))
            return items
    log.warning("All parse strategies failed. Raw (300): %s", content[:300])
    return []

def fix_length(labels, n):
    labels = labels[:n]
    if len(labels) < n:
        raise RuntimeError(
            f"Model returned {len(labels)} labels but expected {n}. "
            f"Response may be truncated. Reduce BATCH size and retry."
        )
    return labels

def norm_pred(x):
  
    while isinstance(x, list):
        x = x[0] if x else ""
    t = (x or "").strip().lower()
    for lbl in sorted(LABELS, key=len, reverse=True):
        if lbl.lower() in t:
            return lbl
    log.warning("Cannot normalize '%s' — defaulting to '%s'", x, LABELS[0])
    return LABELS[0]

def build_llm_client():
    provider = MODEL_CFG["provider"]
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    elif provider == "local":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(HF_ID, token=HF_TOKEN or None)
        model = AutoModelForCausalLM.from_pretrained(
            HF_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN or None,
        )
        model.eval()
        log.info("Loaded local causal LM: %s", HF_ID)
        return {"tokenizer": tokenizer, "model": model}
    else:
        raise ValueError(f"Unknown provider: {provider}")

def call_llm(client, prompt):
    messages = [
        {"role": "system", "content": "Return only JSON."},
        {"role": "user",   "content": prompt},
    ]
    provider = MODEL_CFG["provider"]

    if provider == "local":
        tokenizer = client["tokenizer"]
        model     = client["model"]
        # Newer transformers return BatchEncoding (dict-like) instead of a tensor.
        # Normalize to a plain input_ids tensor.
        chat_out = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(chat_out, "input_ids"):
            input_ids = chat_out["input_ids"].to(model.device)
        elif isinstance(chat_out, dict):
            input_ids = chat_out["input_ids"].to(model.device)
        else:
            input_ids = chat_out.to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0, input_ids.shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    kwargs = dict(model=MODEL_NAME, temperature=0.0, messages=messages)
    if provider == "openai":
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs).choices[0].message.content

def run_llm(texts):
    client  = build_llm_client()
    retries = 3
    preds   = []

    for i in range(0, len(texts), BATCH):
        batch  = texts[i:i + BATCH]
        n      = len(batch)
        prompt = build_prompt(batch)
        labels = None

        for attempt in range(retries):
            try:
                content = call_llm(client, prompt)
                labels  = fix_length(safe_parse(content), n)
                break
            except Exception as e:
                log.warning("Batch %d attempt %d/%d failed: %s", i, attempt + 1, retries, e)
                time.sleep(5)

        if labels is None or len(labels) == 0:
            log.warning("Batch %d — empty response, defaulting to '%s' for %d reviews", i, LABELS[0], n)
            labels = [LABELS[0]] * n

        labels = labels[:n]
        preds.extend([norm_pred(x) for x in labels])
        log.info("%d/%d done", min(i + BATCH, len(texts)), len(texts))

    return preds

# ================================================================
# MAIN RUN
# ================================================================
log.info("Starting inference — Family: %s", FAMILY)
texts   = DF["text"].tolist()
t_start = time.perf_counter()

if FAMILY == "embedding":
    preds = run_embedding(texts)
elif FAMILY == "nli":
    preds = run_nli(texts)
elif FAMILY == "llm":
    preds = run_llm(texts)

total_latency  = time.perf_counter() - t_start
per_sample_lat = total_latency / len(DF)
DF["Predicted"] = preds

log.info("Total Inference Time:   %.2f s", total_latency)
log.info("Avg Latency per Sample: %.4f s", per_sample_lat)

# ================================================================
# METRICS
# ================================================================
macro = weighted = accuracy = macro_p = macro_r = None

if DF["gold"].str.len().sum() > 0:
    macro    = f1_score(DF["gold"], DF["Predicted"], average="macro",    zero_division=0)
    weighted = f1_score(DF["gold"], DF["Predicted"], average="weighted", zero_division=0)
    accuracy = accuracy_score(DF["gold"], DF["Predicted"])
    macro_p  = precision_score(DF["gold"], DF["Predicted"], average="macro", zero_division=0)
    macro_r  = recall_score(DF["gold"],    DF["Predicted"], average="macro", zero_division=0)

    log.info("Accuracy:        %.4f", accuracy)
    log.info("Macro-F1:        %.4f", macro)
    log.info("Weighted-F1:     %.4f", weighted)
    log.info("Macro-Precision: %.4f", macro_p)
    log.info("Macro-Recall:    %.4f", macro_r)
    print(classification_report(DF["gold"], DF["Predicted"], zero_division=0))

# ================================================================
# SAVE
# ================================================================
config_tag = "_".join(filter(None, [
    "persona" if PERSONA_ON else "",
    "fewshot" if FEWSHOT_ON else "zeroshot",
    "cot"     if COT_ON     else "",
]))
model_tag = re.sub(r'[\\/*?:"<>|]', "_", MODEL_NAME)
out       = f"RESULT_{DATASET}_{model_tag}_{config_tag}.xlsx"

perf_rows = []
if macro is not None:
    perf_rows = [
        ("Accuracy",                   accuracy),
        ("Macro-F1",                   macro),
        ("Weighted-F1",                weighted),
        ("Macro-Precision",            macro_p),
        ("Macro-Recall",               macro_r),
        ("Total Inference Time (s)",   total_latency),
        ("Avg Latency per Sample (s)", per_sample_lat),
    ]
else:
    perf_rows = [
        ("Total Inference Time (s)",   total_latency),
        ("Avg Latency per Sample (s)", per_sample_lat),
    ]

with pd.ExcelWriter(out, engine="openpyxl") as w:
    DF.to_excel(w, sheet_name="Predictions", index=False)
    pd.DataFrame(perf_rows, columns=["Metric", "Value"]).to_excel(
        w, sheet_name="Summary", index=False
    )

log.info("Saved: %s", out)
