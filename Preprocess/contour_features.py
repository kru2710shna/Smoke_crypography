import cv2
import numpy as np  
import pandas as pd

def density_feature(mask):
    return float(np.count_nonzero(mask) / mask.size)

def contour_features(mask, area_min=50, base_y=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= area_min]
    num_contours = len(contours)

    if num_contours == 0:
        return dict(
            bounding_box_height=0.0,
            avg_contour_area=0.0,
            centroid_y=np.nan,
            curvature_score=0.0,
            num_contours=0
        )

    ys = np.concatenate([c[:,0,1] for c in contours])
    y_min, y_max = ys.min(), ys.max()
    bbox_h_abs = float(y_max - y_min)

    # relative height above incense base if provided
    if base_y is not None:
        bbox_h = float(max(0.0, base_y - y_min))
    else:
        bbox_h = bbox_h_abs

    areas = np.array([cv2.contourArea(c) for c in contours], dtype=np.float32)
    avg_contour_area = float(areas.mean())

    c_big = contours[int(areas.argmax())]
    M = cv2.moments(c_big)
    centroid_y = float(M["m01"]/M["m00"]) if M["m00"] != 0 else np.nan

    perims = np.array([cv2.arcLength(c, True) for c in contours], dtype=np.float32)
    curvature = (perims**2) / (areas + 1e-6)
    curvature_score = float(np.clip(np.mean(curvature), 0, 1e6))
    
    return dict(
        bounding_box_height=bbox_h,
        avg_contour_area=avg_contour_area,
        centroid_y=centroid_y,
        curvature_score=curvature_score,
        num_contours=num_contours
    )

    


## uncertainty or randomness in the pixel classification (smoke vs. not smoke).

def mask_entropy_bits(mask):
    p = np.count_nonzero(mask) / mask.size
    if p <= 0.0 or p >= 1.0:
        return 0.0
    import math
    return float(-(p*math.log2(p) + (1-p)*math.log2(1-p)))