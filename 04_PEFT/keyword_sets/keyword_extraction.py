import os, re
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
import spacy
from string import punctuation
from collections import Counter

# ========= 1) إعدادات =========
MODEL_PATH = "Computer_Science_D_2.bin"
TOP_K = 10
PROBE_MULTIPLIER = 5

# ========= 2) Seeds (القديمة) =========
SEEDS = {
    "clap": {
        "Bug": ["bug"],
        "Feature": ["feature"],
        "Performance": ["performance"],
        "Security": ["security"],
        "Energy": ["power"],
        "Usability": ["usability"],
        "Other": []
    },

    "pan": {
        "Feature Request": ["request", "feature", "add"],
        "Information Giving": ["information", "details", "description"],
        "Problem Discovery": ["error", "issue", "failure"],
        "Information Seeking": ["question", "ask"]
    },

    "f-droid": {
        "Crashes": ["crash", "failure"],
        "Feature & UI Bugs": ["ui", "interface", "layout"]
    }
}

# ========= 3) تحميل Word2Vec =========
def load_vectors(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    try:
        kv = KeyedVectors.load_word2vec_format(path, binary=path.endswith(".bin"))
        return kv
    except Exception:
        w2v = Word2Vec.load(path)
        return w2v.wv

kv = load_vectors(MODEL_PATH)
print("[i] vocab size:", len(kv.index_to_key))

# ========= 4) NLP =========
nlp = spacy.load("en_core_web_sm")

STOP = set(nlp.Defaults.stop_words) | set(punctuation)
POS_KEEP = {"NOUN", "ADJ", "VERB"}

BAD_WORDS = {
    "hardware", "transistor", "kernel", "socket",
    "protocol", "binary", "compiler", "thread"
}

def clean_token(w):
    w = w.lower().strip()
    w = re.sub(r"[^a-z0-9_\-]+", "", w)
    return w

def pos_ok(w):
    doc = nlp(w)
    return any(t.pos_ in POS_KEEP for t in doc)

def filter_words(words):
    out, seen = [], set()
    for w in words:
        w2 = clean_token(w)
        if not w2 or w2 in STOP or w2 in seen:
            continue
        if w2 in BAD_WORDS:
            continue
        if not pos_ok(w2):
            continue
        seen.add(w2)
        out.append(w2)
    return out

# ========= 5) استخراج كلمات =========
def extract_from_seed(seed):
    if seed not in kv:
        return []
    candidates = kv.most_similar(seed, topn=TOP_K * PROBE_MULTIPLIER)
    words = [w for w, _ in candidates]
    return filter_words(words)

# ========= 6) دمج multi-seeds =========
def get_keywords_multi_seed(seeds):
    counter = Counter()

    for s in seeds:
        words = extract_from_seed(s)
        counter.update(words)

    ranked = [w for w, _ in counter.most_common()]
    return ranked[:TOP_K]

# ========= 7) إزالة التكرار =========
def remove_cross_class_duplicates_balanced(class_dict):
    sorted_classes = sorted(class_dict.items(), key=lambda x: len(x[1]), reverse=True)

    used = set()
    cleaned = {}

    for label, words in sorted_classes:
        new_words = []
        for w in words:
            if w not in used:
                new_words.append(w)
                used.add(w)
        cleaned[label] = new_words

    return cleaned

# ========= 8) التنفيذ =========
results = []

for dataset_name, categories in SEEDS.items():

    print("\n==============================")
    print(f" DATASET: {dataset_name.upper()}")
    print("==============================")

    temp_keywords = {}

    for label, seed_list in categories.items():

        if not seed_list:
            print(f"\n=== {label} ===")
            print("No keywords (intentional)")
            temp_keywords[label] = []
            continue

        keywords = get_keywords_multi_seed(seed_list)

        print(f"\n=== {label} ===")
        print(f"Seeds: {seed_list}")
        print(f"Top-{TOP_K}: {', '.join(keywords) if keywords else 'None'}")

        temp_keywords[label] = keywords

    temp_keywords = remove_cross_class_duplicates_balanced(temp_keywords)

    print("\n--- After Deduplication ---")
    for label, words in temp_keywords.items():
        print(f"{label}: {', '.join(words) if words else 'None'}")

        results.append({
            "Dataset": dataset_name,
            "Class": label,
            "Keywords": ", ".join(words)
        })

# ========= 9) حفظ =========
df = pd.DataFrame(results)
df.to_csv("clean_keywords.csv", index=False, encoding="utf-8")

print("\n[OK] Saved: clean_keywords.csv")