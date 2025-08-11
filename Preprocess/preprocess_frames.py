# Preprocess.py
import cv2
import numpy as np

def preprocess_frames(frame, reseize_frame=960, show=False):
    height, width = frame.shape[:2]
    if width != reseize_frame:
        scale = reseize_frame / width
        frame = cv2.resize(frame, (reseize_frame, int(height*scale)), interpolation=cv2.INTER_AREA)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(cl1, (5, 5), 0)

    if show:
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Gray", gray)
        cv2.imshow("CLAHE", cl1)
        cv2.imshow("Blurred", blurred)
        
        print("Press any key to close windows...")
        cv2.waitKey(0)  
        cv2.destroyAllWindows()

    return frame, gray, blurred


def smoke_mask(gray, method="canny+thresh"):
    if method == "canny+thresh":
        edges = cv2.Canny(gray, 30, 120)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, -10)
        mask = cv2.bitwise_or(edges, th)
    else:
        mask = cv2.Canny(gray, 40, 130)
    k = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=1)
    return mask
