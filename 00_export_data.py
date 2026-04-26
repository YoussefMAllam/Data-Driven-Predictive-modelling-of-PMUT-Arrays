"""
00_export_data.py
Export both Excel sheets to a single unified CSV file.
Run this once before any other script.
Output: data/pmuts_all.csv
"""

import pandas as pd
import os

EXCEL_1 = "All Freq Response Data (1-8 PMUTs).xlsx"
EXCEL_2 = "All_Freq_Response_Data_PMUT9-16.xlsx"
OUT_CSV  = "data/pmuts_all.csv"


def clean_frequency(val):
    if isinstance(val, str):
        val = val.replace(" x 10^", "e").replace("x10^", "e").replace(" ", "")
    return float(val)


def main():
    os.makedirs("data", exist_ok=True)

    df1 = pd.read_excel(EXCEL_1, sheet_name="Averaged_Transfer")
    df2 = pd.read_excel(EXCEL_2, sheet_name="Averaged_Transfer")

    df1["Frequency_MHz"] = df1["Frequency_Hz_rounded"].apply(clean_frequency) / 1e6
    df2["Frequency_MHz"] = df2["Frequency_Hz"].apply(clean_frequency) / 1e6

    cols = ["N_PMUT", "Frequency_MHz", "Amplitude_R_mean", "Amplitude_R_std",
            "Phase_deg_mean", "Phase_deg_std"]
    df_all = pd.concat([df1[cols].dropna(subset=["Amplitude_R_mean"]),
                        df2[cols].dropna(subset=["Amplitude_R_mean"])],
                       ignore_index=True)
    df_all = df_all.sort_values(["N_PMUT", "Frequency_MHz"]).reset_index(drop=True)
    df_all["N_PMUT"] = df_all["N_PMUT"].astype(int)

    df_all.to_csv(OUT_CSV, index=False)
    print(f"Exported {len(df_all)} rows to {OUT_CSV}")
    print(f"PMUTs: {sorted(df_all['N_PMUT'].unique())}")
    print(f"Points per PMUT: {df_all.groupby('N_PMUT').size().unique().tolist()}")


if __name__ == "__main__":
    main()
