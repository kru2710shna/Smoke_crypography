# main.py
import os
import json
import cv2
import numpy as np

from Preprocess.preprocess_frames import preprocess_frames
from Preprocess import smoke_mask
from Preprocess import contour_features
from utils.save_image import save_image
from utils.basic import ensure_dir
from Preprocess import density_feature
from Preprocess import mask_entropy_bits


VIDEO = "data/Thin_Smoke_8___30s___4k_res.mp4"
RESULTS_DIR = "results"


if __name__ == "__main__":
    ensure_dir(RESULTS_DIR)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"Error: could not open {VIDEO}")
        raise SystemExit(1)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: could not read first frame")
        raise SystemExit(1)

    # 1) Preprocess: resize → gray (CLAHE) → blurred
    frame_resized, gray, blurred = preprocess_frames(frame, reseize_frame=960, show=False)

    # 2) Smoke mask (use blurred for smoother mask)
    mask = smoke_mask(blurred, method="canny+thresh")

    # 3) Contour features (set base_y=None unless you know the incense base y)
    feats = contour_features(mask, area_min=50, base_y=None)
    
    # 4) Add density to the dict
    feats["smoke_density"] = density_feature(mask)
    
    # 5) Add entropy (uncertainty) to the dict
    feats["mask_entropy_bits"] = mask_entropy_bits(mask)


    # 4) Save images
    save_image(frame_resized, os.path.join(RESULTS_DIR, "original.jpg"))
    save_image(gray,           os.path.join(RESULTS_DIR, "gray.jpg"))
    save_image(mask,           os.path.join(RESULTS_DIR, "smoke_mask.jpg"))

    # 5) Save features dict as JSON
    feats_path = os.path.join(RESULTS_DIR, "contour_features.json")
    # Convert any numpy types to plain Python for json
    clean_feats = {k: (float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else int(v) if isinstance(v, (np.integer,)) else (None if (isinstance(v, float) and np.isnan(v)) else v))
                   for k, v in feats.items()}
    with open(feats_path, "w") as f:
        json.dump(clean_feats, f, indent=2)
    print(f"[saved] {feats_path}")

    # Optional quick preview window (press any key to close)
    # cv2.imshow("Original", frame_resized)
    # cv2.imshow("Gray (CLAHE)", gray)
    # cv2.imshow("Smoke Mask", mask)
    # cv2.waitKey(0); cv2.destroyAllWindows()
