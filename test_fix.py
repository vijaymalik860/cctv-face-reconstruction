"""Test plate detection on a specific image path."""
import cv2
import sys
import numpy as np

sys.path.insert(0, '.')
from app.services.plate_detector import plate_detector

# Check command line arg for image path
img_path = sys.argv[1] if len(sys.argv) > 1 else 'test_real_annotated.jpg'
img = cv2.imread(img_path)
if img is None:
    print(f"Could not read image: {img_path}")
    sys.exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Image: {img_path}")
print(f"Shape: {img.shape}")
print(f"Mean brightness: {gray.mean():.1f} (night if < 80)")

plates = plate_detector.detect_plates(img)
print(f"\nRESULT: {len(plates)} plate(s) detected")
for i, p in enumerate(plates):
    print(f"  Plate {i+1}: conf={p['confidence']:.2f}  "
          f"pos=({p['x']},{p['y']}) size={p['w']}x{p['h']}")

if plates:
    out = plate_detector.draw_plates(img, plates)
    out_path = img_path.replace('.jpg', '_plate_detected.jpg').replace('.png', '_plate_detected.png')
    cv2.imwrite(out_path, out)
    print(f"\nSaved: {out_path}")
else:
    print("\n[WARN] No plates detected — check image quality / model")
