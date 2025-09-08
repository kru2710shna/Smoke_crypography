# test_sensitivity.py

import pandas as pd
import numpy as np
import hashlib

def hash_vector(vec, salt="smoke"):
    feature_str = ','.join(map(str, vec))
    combined = (salt + feature_str).encode('utf-8')
    return hashlib.sha256(combined).hexdigest()

def test_sensitivity(csv_path, noise_std=0.01):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['window_id'], errors='ignore')

    original = df.iloc[0].values
    original_hash = hash_vector(original)

    perturbed = original + np.random.normal(0, noise_std, size=original.shape)
    perturbed_hash = hash_vector(perturbed)

    hamming_dist = sum(c1 != c2 for c1, c2 in zip(original_hash, perturbed_hash))
    print(f"Original Hash:  {original_hash}")
    print(f"Perturbed Hash: {perturbed_hash}")
    print(f"Hamming Distance: {hamming_dist} / {len(original_hash)}")

if __name__ == "__main__":
    test_sensitivity("../results/window_features.csv")
