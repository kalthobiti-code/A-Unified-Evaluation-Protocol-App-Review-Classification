import pandas as pd
import numpy as np
import os

BASE = '.'

M = 'meta-llama-3-8b-instruct'

DS = [
    ('f-droid', 'F-Droid'),
    ('clap', 'CLAP'),
    ('pan', 'Pan')
]

PARA = [
    ('Zero-Shot', 'Instruction-only'),
    ('Prompting', 'Few-Shot (pre-specified)'),
    ('PEFT', 'DoRA + Sparse ON (pre-specified)')
]

SEED = 42


def load(d, par):

    if par == 'Zero-Shot':
        path = os.path.join(
            BASE,
            f'RESULT_{d}_{M}_zeroshot.xlsx'
        )

        x = pd.read_excel(
            path,
            sheet_name='Predictions'
        )

        return x[['text', 'gold', 'Predicted']].rename(
            columns={
                'gold': 'true_label',
                'Predicted': 'predicted_label'
            }
        )

    if par == 'Prompting':
        path = os.path.join(
            BASE,
            f'RESULT_{d}_{M}_fewshot.xlsx'
        )

        x = pd.read_excel(
            path,
            sheet_name='Predictions'
        )

        return x[['text', 'gold', 'Predicted']].rename(
            columns={
                'gold': 'true_label',
                'Predicted': 'predicted_label'
            }
        )

    # PEFT
    path = os.path.join(
        BASE,
        f'all_folds_predictions{d}.xlsx'
    )

    x = pd.read_excel(path)

    return x[['text', 'true_label', 'pred_label']].rename(
        columns={
            'pred_label': 'predicted_label'
        }
    )


summary = []

with pd.ExcelWriter('ERROR_CODING_SHEETS.xlsx') as w:

    for d, dl in DS:

        for par, cfg in PARA:

            df = load(d, par)

            err = df[
                df['true_label'] != df['predicted_label']
            ].copy().reset_index(drop=True)

            rng = np.random.default_rng(SEED)

            groups = {}

            for lab, g in err.groupby('true_label'):

                idx = np.array(
                    g.index.tolist()
                )

                rng.shuffle(idx)

                groups[lab] = list(idx)

            # Rank classes by number of errors
            order = sorted(
                groups,
                key=lambda k: -len(groups[k])
            )

            # Stratified alternation across classes
            seq = []

            while any(groups[l] for l in order):

                for l in order:

                    if groups[l]:
                        seq.append(
                            groups[l].pop(0)
                        )

            out = err.loc[seq].reset_index(
                drop=True
            )

            out.insert(
                0,
                'id',
                [
                    f'{d}_{par.replace("-", "").lower()}_{i+1:04d}'
                    for i in range(len(out))
                ]
            )

            out['paradigm'] = par
            out['configuration'] = cfg
            out['coding_order'] = range(
                1,
                len(out) + 1
            )

            out = out[
                [
                    'id',
                    'text',
                    'true_label',
                    'predicted_label',
                    'paradigm',
                    'configuration',
                    'coding_order'
                ]
            ]

            sheet = f'{dl}_{par.replace("-", "")}'[:31]

            out.to_excel(
                w,
                sheet_name=sheet,
                index=False
            )

            summary.append(
                {
                    'Dataset': dl,
                    'Paradigm': par,
                    'Configuration': cfg,
                    'Total errors': len(out),
                    'Error-bearing classes':
                        out['true_label'].nunique(),
                    'Sheet': sheet
                }
            )

    pd.DataFrame(
        summary
    ).to_excel(
        w,
        sheet_name='00_Index',
        index=False
    )


print(
    pd.DataFrame(summary).to_string(
        index=False
    )
)

print(
    '\nDONE -> ERROR_CODING_SHEETS.xlsx'
)