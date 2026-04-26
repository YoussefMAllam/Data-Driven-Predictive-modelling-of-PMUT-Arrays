"""
06_progressive_training.py
══════════════════════════════════════════════════════════════════════════════
Cumulative progressive training: train on [1..N-1] → predict PMUT N, N=3..16.

Reads predictions from the pre-computed cache (data/progressive_cache.csv)
written by 01_train_cache.py — no model training happens here.

Two input/output approaches compared side-by-side:
  Vector    — Input: N_PMUT (scalar); Output: 1000-pt amplitude vector.
  Pointwise — Input: [N_PMUT, Freq_MHz]; Output: single amplitude value.

Both evaluated on Full Spectrum (1000 pts) and FWHM ROI (in-band pts).

Files written
─────────────
  outputs/06_progressive/all_combined.html  — interactive page
  outputs/06_progressive/progressive_metrics.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from helpers.cache_utils import load_progressive
from helpers.html_utils  import build_interactive_html

# ── Paths ─────────────────────────────────────────────────────────────────────
PROG_CACHE = os.path.join("data", "progressive_cache.csv")
OUT_DIR    = "outputs/06_progressive"
OUT_CSV    = os.path.join(OUT_DIR, "progressive_metrics.csv")
OUT_COMB   = os.path.join(OUT_DIR, "all_combined.html")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading progressive cache → {PROG_CACHE}")
    all_data, df_vec_metrics = load_progressive(PROG_CACHE)

    # 06 starts from PMUT 3 (need at least 2 training devices)
    all_data = {k: v for k, v in all_data.items() if int(k) >= 3}

    # ── Build per-PMUT metrics CSV (all approaches + ROI) ─────────────────────
    records = []
    for target_str, entry in all_data.items():
        target = int(target_str)
        for approach in ["Vector", "Pointwise"]:
            for model in ["RF", "GB", "MLP"]:
                d = entry[approach][model]
                records.append({
                    "Target_PMUT": target,
                    "N_train":     len(entry["train"]),
                    "Approach":    approach,
                    "Model":       model,
                    "Full_MAE":    d["mae_full"],
                    "Full_R2":     d["r2_full"],
                    "ROI_MAE":     d["mae_roi"],
                    "ROI_R2":      d["r2_roi"],
                })

    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(OUT_CSV, index=False)
    print(f"Saved CSV  → {OUT_CSV}")

    # ── Build interactive HTML ─────────────────────────────────────────────────
    html = build_interactive_html(
        title    = "Progressive Training — PMUTs 3–16",
        subtitle = (
            "Cumulative training: train on all preceding PMUTs [1..N-1], predict PMUT N. "
            "<b>Left:</b> Vector (multi-output). <b>Right:</b> Pointwise. "
            "Use controls above to filter models, approach, scope, or individual PMUT."
        ),
        all_data = all_data,
    )
    with open(OUT_COMB, "w") as f:
        f.write(html)
    print(f"Saved HTML → {OUT_COMB}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── Mean metrics by Approach × Model ─────────────────────────────")
    print(df_metrics.groupby(["Approach", "Model"])[
        ["Full_MAE", "Full_R2", "ROI_MAE", "ROI_R2"]
    ].mean().round(5).to_string())


if __name__ == "__main__":
    main()
