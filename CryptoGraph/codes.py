import hashlib
import pandas as pd
import json

SECRET = "SmokeCrypt0042"  # Your private salt

def normalize_row(row, feature_cols):
    """Convert row values to 4-decimal rounded strings."""
    return [f"{row[col]:.4f}" for col in feature_cols]

def generate_hash_code(feature_str, salt=SECRET):
    """Hash the concatenated string with salt using SHA-256."""
    to_hash = f"{salt}:{feature_str}"
    return hashlib.sha256(to_hash.encode()).hexdigest()

def derive_codes_from_csv(csv_path, out_json_path, out_csv_path):
    df = pd.read_csv(csv_path)

    # Choose key feature columns (or keep all float columns)
    feature_cols = [
        col for col in df.columns if col not in ["window_id", "timestamp"] and df[col].dtype != "object"
    ]

    codes = []
    for _, row in df.iterrows():
        norm_values = normalize_row(row, feature_cols)
        feature_str = "|".join(norm_values)
        code = generate_hash_code(feature_str)

        codes.append({
            "window_id": int(row["window_id"]),
            "hash_code": code,
        })

    # Save JSON
    with open(out_json_path, "w") as f:
        json.dump(codes, f, indent=2)
    print(f"[saved] {out_json_path}")

    # Save CSV
    pd.DataFrame(codes).to_csv(out_csv_path, index=False)
    print(f"[saved] {out_csv_path}")
