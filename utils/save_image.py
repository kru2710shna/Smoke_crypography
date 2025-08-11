import os
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
