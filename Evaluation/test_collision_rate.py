# test_collision_rate.py

import json
from collections import defaultdict

def load_hashes(filepaths):
    all_runs = []
    for path in filepaths:
        with open(path, 'r') as f:
            run = json.load(f)
            all_runs.append({entry['window_id']: entry['hash_code'] for entry in run})
    return all_runs

def compute_collision_rate(all_runs):
    collisions = defaultdict(set)
    num_runs = len(all_runs)
    num_windows = len(all_runs[0])

    for window_id in range(num_windows):
        codes = [run[window_id] for run in all_runs]
        unique_codes = set(codes)
        if len(unique_codes) < len(codes):
            collisions[window_id].update(codes)

    print(f"Collision windows: {len(collisions)} out of {num_windows}")
    for wid, hashes in collisions.items():
        print(f"  Window {wid} → Collision in codes: {hashes}")

if __name__ == "__main__":
    filepaths = [
        "../results/window_codes.json",
        "../results/window_codes2.json",
        "../results/window_codes3.json",
        "../results/window_codes4.json",
        "../results/window_codes5.json",
    ]
    all_runs = load_hashes(filepaths)
    compute_collision_rate(all_runs)
