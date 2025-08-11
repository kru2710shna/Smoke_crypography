import os 

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)