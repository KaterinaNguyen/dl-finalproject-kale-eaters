import os
import cv2
import numpy as np
from PIL import Image

input_dir = "DID-MDN-all"
output_dir = "DID-MDN-split"

input_out_dir = os.path.join(output_dir, "input")
gt_out_dir = os.path.join(output_dir, "gt")
os.makedirs(input_out_dir, exist_ok=True)
os.makedirs(gt_out_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"Skipping unreadable image: {filename}")
            continue

        h, w, _ = img.shape
        if w < 2:
            print(f"Image too narrow to split: {filename}")
            continue

        left = img[:, :w // 2, :]
        right = img[:, w // 2:, :]

        # Convert back to PIL format and save as .png
        left_pil = Image.fromarray(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
        right_pil = Image.fromarray(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))

        base = os.path.splitext(filename)[0]
        left_pil.save(os.path.join(input_out_dir, base + ".png"))
        right_pil.save(os.path.join(gt_out_dir, base + ".png"))

print("Done! Split images saved to 'input/' and 'gt/' folders.")
