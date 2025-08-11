import os 
import numpy as np
import pandas as pd
import cv2



def save_image(image, filename, folder="results"):
    """
    Saves the given image to the specified folder.
    Creates the folder if it doesn't exist.
    """
    os.makedirs(folder, exist_ok=True)  # create folder if not exists
    path = os.path.join(folder, filename)
    cv2.imwrite(path, image)
    print(f"[INFO] Saved: {path}")
    
    
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    
    
def to_py(v):
    if isinstance(v, (np.floating, np.float32, np.float64)): return float(v)
    if isinstance(v, (np.integer,)):                         return int(v)
    if isinstance(v, float) and np.isnan(v):                 return None
    return v


def union_bbox_from_contours(contours):
    if not contours:
        return None
    xs, ys = [], []
    for c in contours:
        xs.append(c[:,0,0])
        ys.append(c[:,0,1])
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return (x0, y0, x1 - x0, y1 - y0)



def draw_flow_field(frame_bgr, prev_gray, gray, mask=None, step=20):
    # Draw sparse arrows showing motion
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    h, w = gray.shape
    for y in range(step//2, h, step):
        for x in range(step//2, w, step):
            if mask is not None and mask[y, x] == 0:
                continue
            dx, dy = flow[y, x, 0], flow[y, x, 1]
            end_pt = (int(x + dx*2), int(y + dy*2))
            cv2.arrowedLine(frame_bgr, (x, y), end_pt, (0, 255, 255), 1, tipLength=0.3)