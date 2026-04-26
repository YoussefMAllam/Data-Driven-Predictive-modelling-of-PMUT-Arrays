"""
01_train_cache.py
══════════════════════════════════════════════════════════════════════════════
Run every model-training combination ONCE and write prediction caches to the
data/ folder.  All downstream scripts (04, 06, 07, 08) read from these CSVs
instead of retraining.

Output
──────
  data/progressive_cache.csv
      Scheme : cumulative [1..N-1] → predict PMUT N,  for N = 2..16
      Rows   : one per (target_pmut × approach × model × frequency_point)

  data/regime_cache.csv
      Scheme : regime-aware (fresh model per regime boundary)
               Regime 1  PMUTs  1–2   [1]→2
               Regime 2  PMUTs  3–11  [3]→4 … [3..10]→11
               Regime 3  PMUTs 12–16  [12]→13 … [12..15]→16
               + cross-regime [1,2]→3 and [3..11]→12
      Same row schema, plus regime_id / regime_name / step_type columns.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from helpers.data_loader import load_data
from helpers.fwhm_utils  import compute_fwhm_window
from helpers.model_utils import train_predict_vector, train_predict_pointwise

DATA_DIR = "data"
PROG_CSV = os.path.join(DATA_DIR, "progressive_cache.csv")
REGI_CSV = os.path.join(DATA_DIR, "regime_cache.csv")

APPROACHES = [("Vector", train_predict_vector), ("Pointwise", train_predict_pointwise)]
MODELS     = ["RF", "GB", "MLP"]

REGIMES = [
    {
        "id": 1, "name": "Regime 1 (1–2)",
        "within":        [{"train": [1], "target": 2}],
        "cross_to_next": {"train": [1, 2], "target": 3},
    },
    {
        "id": 2, "name": "Regime 2 (3–11)",
        "within":        [{"train": list(range(3, t)), "target": t} for t in range(4, 12)],
        "cross_to_next": {"train": list(range(3, 12)), "target": 12},
    },
    {
        "id": 3, "name": "Regime 3 (12–16)",
        "within":        [{"train": list(range(12, t)), "target": t} for t in range(13, 17)],
        "cross_to_next": None,
    },
]


def _predict_all(target, t_pmuts, fwhm_lo, fwhm_hi, df_all, amp_scaler,
                 extra: dict = None) -> list:
    """
    Train and predict for all approach × model combinations.
    Returns a flat list of row dicts — one per frequency point per combination.
    """
    base = {
        "target_pmut": target,
        "train_pmuts":  str(t_pmuts),
        "n_train":      len(t_pmuts),
        "fwhm_lo":      round(float(fwhm_lo), 5),
        "fwhm_hi":      round(float(fwhm_hi), 5),
        **(extra or {}),
    }
    rows = []
    for approach_name, train_fn in APPROACHES:
        for model in MODELS:
            fq, act, pr = train_fn(model, df_all, t_pmuts, target, amp_scaler)
            for f, a, p in zip(fq, act, pr):
                rows.append({
                    **base,
                    "approach":      approach_name,
                    "model":         model,
                    "frequency_mhz": round(float(f), 5),
                    "actual":        round(float(a), 6),
                    "predicted":     round(float(p), 6),
                })
            print(f"    PMUT {target:>2}  {model:<3}  {approach_name:<10}  ✓")
    return rows


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df_all, amp_scaler, unique_pmuts, _, _ = load_data()

    # ── Progressive (targets 2–16) ─────────────────────────────────────────────
    print("\n── Progressive Training ────────────────────────────────────────────")
    prog_rows = []
    for target in [int(p) for p in unique_pmuts if p >= 2]:
        t_pmuts = [int(p) for p in unique_pmuts if p < target]
        fwhm_lo, fwhm_hi = compute_fwhm_window(target, df_all)
        print(f"  PMUT {target:>2}  train={t_pmuts}")
        prog_rows.extend(
            _predict_all(target, t_pmuts, fwhm_lo, fwhm_hi, df_all, amp_scaler))

    df_prog = (pd.DataFrame(prog_rows)
               .sort_values(["target_pmut", "approach", "model", "frequency_mhz"]))
    df_prog.to_csv(PROG_CSV, index=False)
    print(f"\nSaved {len(df_prog):,} rows → {PROG_CSV}")

    # ── Regime-Aware ──────────────────────────────────────────────────────────
    print("\n── Regime-Aware Training ───────────────────────────────────────────")
    regi_rows = []
    for regime in REGIMES:
        rid, rname = regime["id"], regime["name"]
        for step in regime["within"]:
            target, t_pmuts = step["target"], step["train"]
            fwhm_lo, fwhm_hi = compute_fwhm_window(target, df_all)
            print(f"  R{rid}  PMUT {target:>2} (within)  train={t_pmuts}")
            regi_rows.extend(
                _predict_all(target, t_pmuts, fwhm_lo, fwhm_hi, df_all, amp_scaler,
                             extra={"regime_id": rid, "regime_name": rname,
                                    "step_type": "within"}))
        if regime["cross_to_next"]:
            cx = regime["cross_to_next"]
            target, t_pmuts = cx["target"], cx["train"]
            fwhm_lo, fwhm_hi = compute_fwhm_window(target, df_all)
            print(f"  R{rid}  PMUT {target:>2} ★ cross  train={t_pmuts}")
            regi_rows.extend(
                _predict_all(target, t_pmuts, fwhm_lo, fwhm_hi, df_all, amp_scaler,
                             extra={"regime_id": rid, "regime_name": rname,
                                    "step_type": "cross"}))

    df_regi = (pd.DataFrame(regi_rows)
               .sort_values(["target_pmut", "approach", "model", "frequency_mhz"]))
    df_regi.to_csv(REGI_CSV, index=False)
    print(f"\nSaved {len(df_regi):,} rows → {REGI_CSV}")

    print("\nCache complete — downstream scripts will read from data/ instead of retraining.")


if __name__ == "__main__":
    main()
