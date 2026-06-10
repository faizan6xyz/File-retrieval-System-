from ultralytics import YOLO
from PIL import Image
import pyautogui, time

detect_model = YOLO("weights/icon_detect/model.pt")

image_path = r"dataset/001_github_com_faizan6xyz.png"
results = detect_model(image_path)
img = Image.open(image_path).convert("RGB")
IMG_W, IMG_H = img.size

# Screen and browser info
BROWSER_W  = 1536
BROWSER_H  = 816
TOP_OFFSET = 110

print(f"Image size: {IMG_W}x{IMG_H}")
print(f"Detected {len(results[0].boxes)} elements\n")

elements = []
for i, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    scale_x = BROWSER_W / IMG_W
    scale_y = BROWSER_H / IMG_H
    actual_x = int(cx * scale_x)
    actual_y = int(cy * scale_y + TOP_OFFSET)

    elements.append({"actual_x": actual_x, "actual_y": actual_y})
    print(f"Element {i+1}: screen({actual_x}, {actual_y})")
