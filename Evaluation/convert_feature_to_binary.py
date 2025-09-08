# convert_feature_to_binary.py

import pandas as pd
import struct

def convert_to_binary(input_csv, output_file):
    df = pd.read_csv(input_csv)
    df = df.drop(columns=['window_id'], errors='ignore')

    with open(output_file, 'wb') as f:
        for _, row in df.iterrows():
            for val in row:
                b = struct.pack('f', val)
                f.write(b)

if __name__ == "__main__":
    convert_to_binary("../results/window_features.csv", "../results/feature_stream.bin")
