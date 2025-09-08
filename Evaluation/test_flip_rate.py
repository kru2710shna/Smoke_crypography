# test_flip_rate.py

import json

def compute_flip_rate(file_path):
    with open(file_path, 'r') as f:
        hashes = json.load(f)

    flips = 0
    total = len(hashes) - 1

    for i in range(total):
        if hashes[i]['hash_code'] != hashes[i + 1]['hash_code']:
            flips += 1

    flip_rate = flips / total
    print(f"Flip Rate: {flip_rate:.4f} ({flips}/{total})")

if __name__ == "__main__":
    compute_flip_rate("../results/window_codes.json")
