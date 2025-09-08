import pandas as pd
import numpy as np
import os
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CryptoGraph import derive_codes_from_csv




def aggregate_windows(csv_path, out_csv, fps=30, frame_step=5, window_secs=7):
    # Load per-frame dataset
    df = pd.read_csv(csv_path)

    # Effective fps (after frame skipping)
    eff_fps = fps / frame_step
    frames_per_window = int(window_secs * eff_fps)

    # Assign each row to a window_id
    df["window_id"] = df["frame_idx"] // frames_per_window

    # Numeric columns (ignore frame_idx & window_id)
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(["frame_idx", "window_id"])

    # Aggregate per window
    agg_funcs = ["mean", "std", "min", "max"]
    df_win = df.groupby("window_id")[num_cols].agg(agg_funcs)

    # Flatten column MultiIndex (feature_stat)
    df_win.columns = [f"{c}_{stat}" for c, stat in df_win.columns]
    df_win.reset_index(inplace=True)

    # Save to CSV
    df_win.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv} ({len(df_win)} windows)")

    return df_win

if __name__ == "__main__":
    in_csv = "results/features.csv"
    out_csv = "results/window_features.csv"
    aggregate_windows(in_csv, out_csv, fps=30, frame_step=5, window_secs=7)
    derive_codes_from_csv(
        csv_path="results/window_features.csv",
        out_json_path="results/window_codes5.json",
        out_csv_path="results/window_codes5.csv"
    )
