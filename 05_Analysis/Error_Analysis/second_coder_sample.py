import pandas as pd
import numpy as np

INPUT_FILE = "ERROR_CODING_SHEETS-C1-C5.xlsx"
OUTPUT_FILE = "SECOND_CODER_30.xlsx"

SEED = 42
TARGET_N = 30

GROUP_SHEETS = [
    "F-Droid_ZeroShot",
    "F-Droid_Prompting",
    "F-Droid_PEFT",
    "CLAP_ZeroShot",
    "CLAP_Prompting",
    "CLAP_PEFT",
    "Pan_ZeroShot",
    "Pan_Prompting",
    "Pan_PEFT"
]

rng = np.random.default_rng(SEED)

all_rows = []

for sheet in GROUP_SHEETS:
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet)

    # Keep only cases coded up to saturation point
    coded = df[
        df["Error_Category"].notna() &
        df["Error_Category"].astype(str).str.startswith("C")
    ].copy()

    coded["Source_Sheet"] = sheet
    all_rows.append(coded)

combined = pd.concat(all_rows, ignore_index=True)

print("Total coded cases available:", len(combined))

# Should be 118
if len(combined) != 118:
    print("WARNING: Expected 118 coded cases, but found:", len(combined))

# Each sheet is an independent stratum
strata = GROUP_SHEETS.copy()

# 30 / 9:
# 3 cases per group = 27
# + 1 extra for three groups = 30
rng.shuffle(strata)

base_n = TARGET_N // len(strata)      # 3
remainder = TARGET_N % len(strata)    # 3

sampled_parts = []

for i, stratum in enumerate(strata):

    g = combined[
        combined["Source_Sheet"] == stratum
    ].copy()

    n_take = base_n + (1 if i < remainder else 0)

    selected_idx = rng.choice(
        g.index.to_numpy(),
        size=n_take,
        replace=False
    )

    sampled_parts.append(
        combined.loc[selected_idx].copy()
    )

sample30 = pd.concat(
    sampled_parts,
    ignore_index=True
)

# Final shuffle so the second coder does not see groups ordered
sample30 = sample30.iloc[
    rng.permutation(len(sample30))
].reset_index(drop=True)

sample30.insert(
    0,
    "Coder2_ID",
    [f"C2_{i+1:02d}" for i in range(len(sample30))]
)

# MASTER: keeps our coding for later comparison
master = sample30.copy()

# Second coder version
# Remove our codes so they do not bias the coder
coder2 = sample30.drop(
    columns=[
        c for c in [
            "Error_Category",
            "Error_Category_Name",
            "Coding_Status",
            "New_Category",
            "Source_Sheet"
        ]
        if c in sample30.columns
    ]
).copy()

# Empty column for the second coder to fill
coder2["Coder2_Code"] = ""

# Codebook from the original file
codebook = pd.read_excel(
    INPUT_FILE,
    sheet_name="Codebook"
)

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    coder2.to_excel(
        writer,
        sheet_name="Coder2_Sample",
        index=False
    )

    codebook.to_excel(
        writer,
        sheet_name="Codebook",
        index=False
    )

    master.to_excel(
        writer,
        sheet_name="Master_Do_Not_Share",
        index=False
    )

print("\nDONE ->", OUTPUT_FILE)

print("\nSample distribution:")
print(
    master["Source_Sheet"]
    .value_counts()
    .sort_index()
)

print("\nTotal sampled:", len(master))