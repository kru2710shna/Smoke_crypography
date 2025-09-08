# main.py (annotated video + live HUD)
import os
import json
import cv2
import numpy as np
import pandas as pd

from utils.basic import ensure_dir
from utils.basic import save_image
from utils.basic import to_py
from utils.basic import union_bbox_from_contours
from utils.basic import draw_flow_field

from Preprocess.preprocess_frames import preprocess_frames
from Preprocess import smoke_mask
from Preprocess import contour_features
from Preprocess import density_feature
from Preprocess import mask_entropy_bits
from Preprocess import optical_flow_extras



VIDEO = "data/Thin_Smoke_8___30s___4k_res.mp4"
RESULTS_DIR = "results"
FRAME_STEP = 5        # process every Nth frame for speed
RESIZE_W = 960
AREA_MIN = 50
BASE_Y = None           # set if you know the incense base y (after resize)
DRAW_FLOW_FIELD = False # set True to draw a sparse flow field arrows (slower)


if __name__ == "__main__":
    ensure_dir(RESULTS_DIR)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"Error: could not open {VIDEO}")
        raise SystemExit(1)

    # Determine output size & fps from first frame
    ok, probe = cap.read()
    if not ok:
        print("Error: video has no frames")
        raise SystemExit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Pre-size by our pipeline so writer matches the annotated frame size
    probe_r, _, _ = preprocess_frames(probe, reseize_frame=RESIZE_W, show=False)
    out_h, out_w = probe_r.shape[:2]

    raw_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = max(1.0, raw_fps / max(1, FRAME_STEP))  # approximate actual processed FPS

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(RESULTS_DIR, "annotated.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))

    rows = []
    prev_gray = None
    first_saved = False
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % FRAME_STEP != 0:
            frame_idx += 1
            continue

        # Preprocess
        frame_r, gray, blurred = preprocess_frames(frame, reseize_frame=RESIZE_W, show=False)
        overlay = frame_r.copy()

        # Mask & contours (for drawing)
        mask = smoke_mask(blurred, method="canny+thresh")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if cv2.contourArea(c) >= AREA_MIN]

        # Features
        feats_c = contour_features(mask, area_min=AREA_MIN, base_y=BASE_Y)
        dens = density_feature(mask)
        ent_bits = mask_entropy_bits(mask)

        if prev_gray is None:
            flow = dict(optical_flow_x=0.0, optical_flow_y=0.0,
                        flow_mag_mean=0.0, flow_mag_std=0.0, flow_ang_std=0.0)
        else:
            flow = optical_flow_extras(prev_gray, blurred, mask=mask)

        # Draw contours
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

        # Draw union bbox
        ub = union_bbox_from_contours(contours)
        if ub is not None:
            x, y, w, h = ub
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 140, 255), 2)

        # Draw centroid (for largest contour)
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            c_big = contours[int(np.argmax(areas))]
            M = cv2.moments(c_big)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
                cv2.circle(overlay, (cx, cy), 4, (255, 50, 50), -1)

        # Optional: draw sparse flow field
        if DRAW_FLOW_FIELD and prev_gray is not None:
            draw_flow_field(overlay, prev_gray, blurred, mask)

        # HUD text (top-left)
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
            cv2.putText(overlay, line, (10, y0 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(overlay, line, (10, y0 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1)

        # Write frame to video
        writer.write(overlay)

        # Show live
        cv2.imshow("Smoke Annotated (q quits)", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save preview images once
        if not first_saved:
            save_image(frame_r, os.path.join(RESULTS_DIR, "original.jpg"))
            save_image(gray,    os.path.join(RESULTS_DIR, "gray.jpg"))
            save_image(mask,    os.path.join(RESULTS_DIR, "smoke_mask.jpg"))
            first_saved = True

        # Collect row for CSV
        rows.append(dict(
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
        ))

        prev_gray = blurred
        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Save CSV & JSON
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "features.csv")
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path} ({len(df)} rows)")

    json_path = os.path.join(RESULTS_DIR, "features.json")
    clean_rows = [{k: to_py(v) for k, v in r.items()} for r in rows]
    with open(json_path, "w") as f:
        json.dump(clean_rows, f, indent=2)
    print(f"[saved] {json_path}")
    print(f"[saved] {out_path}")
