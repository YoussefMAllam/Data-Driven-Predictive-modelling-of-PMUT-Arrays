"""
07_regime_aware_training.py
══════════════════════════════════════════════════════════════════════════════
Regime-aware progressive training with three experimental regimes:

  Regime 1  PMUTs  1– 2  within: [1]→2
  Regime 2  PMUTs  3–11  within: [3]→4, [3,4]→5 … [3..10]→11
  Regime 3  PMUTs 12–16  within: [12]→13 … [12..15]→16

  ★ Cross-regime predictions (fresh model → first PMUT of next regime):
      End of Regime 1 → [1,2]   → Predict PMUT  3
      End of Regime 2 → [3..11] → Predict PMUT 12

Reads predictions from the pre-computed cache (data/regime_cache.csv)
written by 01_train_cache.py — no model training happens here.

Two input/output approaches compared side-by-side:
  Vector    — Input: N_PMUT (scalar); Output: 1000-pt amplitude vector.
  Pointwise — Input: [N_PMUT, Freq_MHz]; Output: single amplitude value.

Both evaluated on Full Spectrum and FWHM ROI.

Files written
─────────────
  outputs/07_regime/all_combined.html   — single interactive page
  outputs/07_regime/regime_metrics.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from helpers.cache_utils import load_regime
from helpers.html_utils  import build_interactive_html

# ── Paths ─────────────────────────────────────────────────────────────────────
REGI_CACHE = os.path.join("data", "regime_cache.csv")
OUT_DIR    = "outputs/07_regime"
OUT_CSV    = os.path.join(OUT_DIR, "regime_metrics.csv")
OUT_COMB   = os.path.join(OUT_DIR, "all_combined.html")

# Regime metadata for the HTML controls
REGIME_LABELS = [
    ("1", "Regime 1 (1–2)"),
    ("2", "Regime 2 (3–11)"),
    ("3", "Regime 3 (12–16)"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading regime cache → {REGI_CACHE}")
    all_data, regime_map, cross_pmuts, df_vec_metrics = load_regime(REGI_CACHE)

    # ── Build per-PMUT metrics CSV (all approaches + ROI) ─────────────────────
    records = []
    for target_str, entry in all_data.items():
        target = int(target_str)
        for approach in ["Vector", "Pointwise"]:
            for model in ["RF", "GB", "MLP", "Physics"]:
                d = entry[approach][model]
                records.append({
                    "Target_PMUT":  target,
                    "N_train":      len(entry["train"]),
                    "Regime":       regime_map.get(target),
                    "Regime_Name":  entry.get("regime_name", ""),
                    "Step_Type":    entry.get("step_type", "within"),
                    "Approach":     approach,
                    "Model":        model,
                    "Full_MAE":     d["mae_full"],
                    "Full_R2":      d["r2_full"],
                    "ROI_MAE":      d["mae_roi"],
                    "ROI_R2":       d["r2_roi"],
                })

    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(OUT_CSV, index=False)
    print(f"Saved CSV  → {OUT_CSV}")

    # ── Build interactive HTML ─────────────────────────────────────────────────
    html = build_interactive_html(
        title    = "Regime-Aware Training — PMUTs 2–16",
        subtitle = (
            "Fresh model at each regime boundary. "
            "<b>★ Cross-Regime</b>: all PMUTs of completed regime → predict first of next regime. "
            "Regimes: <b>1–2</b> | <b>3–11</b> | <b>12–16</b>. "
            "Use controls above to filter models, approach, scope, or regime."
        ),
        all_data      = all_data,
        regime_map    = regime_map,
        cross_pmuts   = cross_pmuts,
        regime_labels = REGIME_LABELS,
    )
    with open(OUT_COMB, "w") as f:
        f.write(html)
    print(f"Saved HTML → {OUT_COMB}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n── Mean metrics by Regime × Approach × Model ────────────────────")
    print(df_metrics.groupby(["Regime_Name", "Approach", "Model"])[
        ["Full_MAE", "Full_R2", "ROI_MAE", "ROI_R2"]
    ].mean().round(5).to_string())


if __name__ == "__main__":
    main()
