# test_min_entropy.py

import pandas as pd
import numpy as np

def estimate_min_entropy(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['window_id'], errors='ignore')

    min_entropies = []

    for col in df.columns:
        values, counts = np.unique(df[col].round(3), return_counts=True)
        probs = counts / sum(counts)
        max_prob = max(probs)
        min_entropy = -np.log2(max_prob)
        min_entropies.append(min_entropy)
        print(f"{col}: Min-Entropy ≈ {min_entropy:.4f}")

    avg_entropy = np.mean(min_entropies)
    print(f"\nAverage Min-Entropy per feature: {avg_entropy:.4f}")

if __name__ == "__main__":
    estimate_min_entropy("../results/window_features.csv")
