"""
00_export_data.py
Export both Excel sheets to a unified CSV file containing the averaged transfer
curve plus the raw sweep-level quantities needed for physics-informed fitting.

Outputs
-------
  data/pmuts_all.csv        — averaged transfer curve per PMUT and frequency
  data/physics_sweeps.csv  — raw sweep-level measurements for fitting
"""

import os
import numpy as np
import pandas as pd

EXCEL_FILES = [
    ("All Freq Response Data (1-8 PMUTs).xlsx", "All_Sweeps_Labeled"),
    ("All_Freq_Response_Data_PMUT9-16.xlsx", "All_Sweeps_Labeled"),
]
OUT_CSV = "data/pmuts_all.csv"
OUT_SWEEPS = "data/physics_sweeps.csv"


def _pick_column(df, options):
    for col in options:
        if col in df.columns:
            return col
    return None


def _read_sweep_sheet(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="All_Sweeps_Labeled")

    freq_col = _pick_column(df, ["Frequency_Hz", "Frequency_Hz_rounded"])
    amp_col = _pick_column(df, ["Amplitude_R", "Amplitude_R_mean"])
    phase_col = _pick_column(df, ["Phase_deg", "Phase_deg_mean"])
    voltage_col = _pick_column(df, ["auxin0", "auxin1", "V", "auxin0pwr", "auxin1pwr"])

    if freq_col is None or amp_col is None or phase_col is None:
        raise ValueError(f"Required sweep columns missing in {path}")

    def _safe_int(series):
        if series is None:
            return pd.Series(0, index=df.index)
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

    out = pd.DataFrame({
        "N_PMUT": df["N_PMUT"].astype(int),
        "PMUTSweep": _safe_int(df["Sweep"] if "Sweep" in df.columns else None),
        "DetectionN": _safe_int(df["Detection"] if "Detection" in df.columns else None),
        "Frequency_Hz": df[freq_col].astype(float),
        "Amplitude_R": df[amp_col].astype(float),
        "Phase_deg": df[phase_col].astype(float),
    })

    if voltage_col is not None:
        out["V"] = df[voltage_col].astype(float)
    else:
        out["V"] = 1.0

    out["Frequency_MHz"] = out["Frequency_Hz"] / 1e6
    out["w_rad_s"] = 2.0 * np.pi * out["Frequency_Hz"]
    out["phase_rad"] = np.radians(out["Phase_deg"])
    out["X_val"] = out["Amplitude_R"] * np.cos(out["phase_rad"])
    out["Y_val"] = out["Amplitude_R"] * np.sin(out["phase_rad"])
    out = out.dropna(subset=["Amplitude_R", "Phase_deg"]).reset_index(drop=True)
    return out


def main():
    os.makedirs("data", exist_ok=True)

    sweep_frames = [_read_sweep_sheet(path) for path, _ in EXCEL_FILES]
    df_sweeps = pd.concat(sweep_frames, ignore_index=True)

    # Keep the original averaged-transfer schema expected by the ML pipeline,
    # but add the raw sweep fields needed for physics-informed fitting.
    summary = (
        df_sweeps
        .groupby(["N_PMUT", "Frequency_Hz"], as_index=False)
        .agg(
            Frequency_MHz=("Frequency_MHz", "first"),
            Amplitude_R_mean=("Amplitude_R", "mean"),
            Amplitude_R_std=("Amplitude_R", "std"),
            Phase_deg_mean=("Phase_deg", "mean"),
            Phase_deg_std=("Phase_deg", "std"),
            V=("V", "mean"),
            PMUTSweep=("PMUTSweep", "first"),
            DetectionN=("DetectionN", "first"),
        )
    )
    summary["Amplitude_R_std"] = summary["Amplitude_R_std"].fillna(0.0)
    summary["Phase_deg_std"] = summary["Phase_deg_std"].fillna(0.0)

    # Add the physics helper columns to the averaged summary as well.
    summary["w_rad_s"] = 2.0 * np.pi * summary["Frequency_Hz"]
    summary["phase_rad"] = np.radians(summary["Phase_deg_mean"])
    summary["X_val"] = summary["Amplitude_R_mean"] * np.cos(summary["phase_rad"])
    summary["Y_val"] = summary["Amplitude_R_mean"] * np.sin(summary["phase_rad"])

    summary = summary.sort_values(["N_PMUT", "Frequency_Hz"]).reset_index(drop=True)
    summary["N_PMUT"] = summary["N_PMUT"].astype(int)

    summary.to_csv(OUT_CSV, index=False)
    df_sweeps.to_csv(OUT_SWEEPS, index=False)

    print(f"Exported averaged transfer rows to {OUT_CSV}")
    print(f"Exported sweep rows to {OUT_SWEEPS}")
    print(f"PMUTs: {sorted(summary['N_PMUT'].unique())}")
    print(f"Points per PMUT: {summary.groupby('N_PMUT').size().unique().tolist()}")


if __name__ == "__main__":
    main()
