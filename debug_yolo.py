"""
Debug script: Run YOLO directly on image without any filters to see raw detections.
Also tests color validation separately.
"""
import cv2
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

img_path = sys.argv[1] if len(sys.argv) > 1 else 'test_real_annotated.jpg'
img = cv2.imread(img_path)
if img is None:
    print(f"Cannot read: {img_path}")
    sys.exit(1)

ih, iw = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
brightness = gray.mean()
print(f"Image: {img_path}  |  {iw}x{ih}  |  brightness={brightness:.1f}")

# ── Load YOLO directly ──────────────────────────────────────────────────
from ultralytics import YOLO
model_path = Path('models/indian_license_plate.pt')
if not model_path.exists():
    print("Model not found!")
    sys.exit(1)

model = YOLO(str(model_path))

# Test on original + brightened
def run(src, label):
    results = model(src, verbose=False, conf=0.10, iou=0.45)
    print(f"\n--- YOLO on {label} ---")
    found = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            w, h = x2-x1, y2-y1
            asp  = w/h if h>0 else 0
            print(f"  conf={conf:.3f}  ({x1},{y1}) {w}x{h}  asp={asp:.2f}")
            found += 1
    if found == 0:
        print("  (no detections)")
    return found

run(img, "original")

# Brighten for night
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
bright = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
run(bright, "CLAHE brightened")

# Also run full pipeline
print("\n\n=== Full PlateDetector pipeline ===")
from app.services.plate_detector import plate_detector
plates = plate_detector.detect_plates(img)
print(f"Final: {len(plates)} plates")
for i, p in enumerate(plates):
    print(f"  [{i+1}] conf={p['confidence']:.2f}  ({p['x']},{p['y']}) {p['w']}x{p['h']}")

if plates:
    out = plate_detector.draw_plates(img, plates)
    out_path = img_path.replace('.jpg','_DEBUG.jpg').replace('.png','_DEBUG.png')
    cv2.imwrite(out_path, out)
    print(f"\nSaved: {out_path}")
