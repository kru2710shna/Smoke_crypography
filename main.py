# main.py (annotated video + live HUD + real-time HMAC)
import os
import json
import cv2
import numpy as np
import pandas as pd
# import sys
# import hmac
# import hashlib
# from collections import deque

from utils.basic import (
    ensure_dir,
    save_image,
    to_py,
    union_bbox_from_contours,
    draw_flow_field,
)
from Preprocess.preprocess_frames import preprocess_frames
from Preprocess import (
    smoke_mask,
    contour_features,
    density_feature,
    mask_entropy_bits,
    optical_flow_extras,
)
from CryptoGraph.codes import (
    generate_hash_code,
)  # You already exposed this in __init__.py

VIDEO = "data/Thin_Smoke_8___30s___4k_res.mp4"
RESULTS_DIR = "results"
FRAME_STEP = 5
RESIZE_W = 960
AREA_MIN = 50
BASE_Y = None
DRAW_FLOW_FIELD = False

SECRET_KEY = b"smoke2025_secret"
WINDOW_DURATION_SEC = 2  # Produce hash every 2 seconds
WINDOW_FEATURES = []
HASH_CODES = []
WINDOW_FRAME_COUNT = 0

if __name__ == "__main__":
    ensure_dir(RESULTS_DIR)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"Error: could not open {VIDEO}")

    ok, probe = cap.read()
    if not ok:
        raise SystemExit("Error: video has no frames")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    probe_r, _, _ = preprocess_frames(probe, reseize_frame=RESIZE_W, show=False)
    out_h, out_w = probe_r.shape[:2]
    raw_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = max(1.0, raw_fps / max(1, FRAME_STEP))

    frames_per_window = int(raw_fps * WINDOW_DURATION_SEC / FRAME_STEP)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        os.path.join(RESULTS_DIR, "annotated.mp4"), fourcc, out_fps, (out_w, out_h)
    )

    rows, prev_gray, first_saved = [], None, False
    frame_idx, window_id, current_code = 0, 0, None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % FRAME_STEP != 0:
            frame_idx += 1
            continue

        frame_r, gray, blurred = preprocess_frames(
            frame, reseize_frame=RESIZE_W, show=False
        )
        overlay = frame_r.copy()

        mask = smoke_mask(blurred, method="canny+thresh")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if cv2.contourArea(c) >= AREA_MIN]

        feats_c = contour_features(mask, area_min=AREA_MIN, base_y=BASE_Y)
        dens = density_feature(mask)
        ent_bits = mask_entropy_bits(mask)
        flow = (
            optical_flow_extras(prev_gray, blurred, mask)
            if prev_gray is not None
            else {
                "optical_flow_x": 0.0,
                "optical_flow_y": 0.0,
                "flow_mag_mean": 0.0,
                "flow_mag_std": 0.0,
                "flow_ang_std": 0.0,
            }
        )

        # Collect window feature
        WINDOW_FEATURES.append(
            [
                feats_c["bounding_box_height"],
                feats_c["avg_contour_area"],
                feats_c["num_contours"],
                feats_c["centroid_y"],
                feats_c["curvature_score"],
                dens,
                ent_bits,
                flow["optical_flow_x"],
                flow["optical_flow_y"],
                flow["flow_mag_mean"],
                flow["flow_mag_std"],
                flow["flow_ang_std"],
            ]
        )
        WINDOW_FRAME_COUNT += 1

        if WINDOW_FRAME_COUNT == frames_per_window:
            # Generate HMAC-based hash code for this window
            window_array = np.mean(WINDOW_FEATURES, axis=0)
            current_code = generate_hash_code(window_array.tolist(), salt=SECRET_KEY)
            HASH_CODES.append({"window_id": window_id, "hash_code": current_code})
            WINDOW_FEATURES = []
            WINDOW_FRAME_COUNT = 0
            window_id += 1

        # Draw contours
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
        if (ub := union_bbox_from_contours(contours)) is not None:
            x, y, w, h = ub
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 140, 255), 2)
        if contours:
            c_big = contours[np.argmax([cv2.contourArea(c) for c in contours])]
            M = cv2.moments(c_big)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.circle(overlay, (cx, cy), 4, (255, 50, 50), -1)

        if DRAW_FLOW_FIELD and prev_gray is not None:
            draw_flow_field(overlay, prev_gray, blurred, mask)

        # HUD Info
        hud = [
            f"frame: {frame_idx}",
            f"contours: {feats_c['num_contours']}",
            f"bbox_h: {feats_c['bounding_box_height']:.1f}",
            f"area_avg: {feats_c['avg_contour_area']:.1f}",
            f"centroid_y: {feats_c['centroid_y']:.1f}",
            f"curvature: {feats_c['curvature_score']:.1f}",
            f"density: {dens:.4f}",
            f"entropy(bits): {ent_bits:.3f}",
            f"flow(x,y): ({flow['optical_flow_x']:.3f}, {flow['optical_flow_y']:.3f})",
            f"flow|mag|: μ={flow['flow_mag_mean']:.3f} σ={flow['flow_mag_std']:.3f}",
        ]
        y0 = 22
        for i, line in enumerate(hud):
            cv2.putText(
                overlay,
                line,
                (10, y0 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                overlay,
                line,
                (10, y0 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (30, 30, 30),
                1,
            )

        # Draw Code
        if current_code:
            chars_per_line = 50
            wrapped_lines = [
                current_code[i : i + chars_per_line]
                for i in range(0, len(current_code), chars_per_line)
            ]
            y_text = out_h - (22 * len(wrapped_lines)) - 10
            for line in wrapped_lines:
                cv2.putText(
                    overlay,
                    f"CODE: {line}",
                    (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                y_text += 22

        writer.write(overlay)
        cv2.imshow("Smoke Annotated (q quits)", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if not first_saved:
            save_image(frame_r, os.path.join(RESULTS_DIR, "original.jpg"))
            save_image(gray, os.path.join(RESULTS_DIR, "gray.jpg"))
            save_image(mask, os.path.join(RESULTS_DIR, "smoke_mask.jpg"))
            first_saved = True

        rows.append(
            dict(
                frame_idx=frame_idx,
                bounding_box_height=feats_c["bounding_box_height"],
                avg_contour_area=feats_c["avg_contour_area"],
                num_contours=feats_c["num_contours"],
                centroid_y=feats_c["centroid_y"],
                curvature_score=feats_c["curvature_score"],
                smoke_density=dens,
                mask_entropy_bits=ent_bits,
                optical_flow_x=flow["optical_flow_x"],
                optical_flow_y=flow["optical_flow_y"],
                flow_mag_mean=flow["flow_mag_mean"],
                flow_mag_std=flow["flow_mag_std"],
                flow_ang_std=flow["flow_ang_std"],
            )
        )

        prev_gray = blurred
        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "features.csv"), index=False)
    print(f"[saved] features.csv with {len(df)} rows")

    with open(os.path.join(RESULTS_DIR, "features.json"), "w") as f:
        json.dump([{k: to_py(v) for k, v in r.items()} for r in rows], f, indent=2)

    with open(os.path.join(RESULTS_DIR, "window_codes.json"), "w") as f:
        json.dump(HASH_CODES, f, indent=2)
    print(f"[saved] window_codes.json with {len(HASH_CODES)} windows")
