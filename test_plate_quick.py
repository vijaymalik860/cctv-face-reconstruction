"""Quick test for plate_detector on a synthetic car plate image."""
import cv2
import numpy as np
import sys
sys.path.insert(0, r"d:\cctv face reconstruction")

# Build synthetic car + plate image
img = np.full((480, 640, 3), 80, dtype=np.uint8)
cv2.rectangle(img, (0, 0), (640, 350), (120, 120, 120), -1)
cv2.ellipse(img, (150, 320), (80, 40), 0, 0, 360, (220, 220, 180), -1)
cv2.ellipse(img, (490, 320), (80, 40), 0, 0, 360, (220, 220, 180), -1)
# White plate rectangle
cv2.rectangle(img, (210, 355), (430, 405), (255, 255, 255), -1)
cv2.rectangle(img, (210, 355), (430, 405), (0, 0, 0), 2)
cv2.putText(img, "HR 13N 2950", (220, 392), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 150), 2)
cv2.imwrite(r"d:\cctv face reconstruction\test_plate_synth.jpg", img)
print("Saved test_plate_synth.jpg")

# Run the new detector
from app.services.plate_detector import plate_detector

for label, test_img in [("synthetic", img)]:
    plates = plate_detector.detect_plates(test_img)
    print(f"\n[{label}] Detected {len(plates)} plate(s):")
    for p in plates:
        print(f"  x={p['x']} y={p['y']} w={p['w']} h={p['h']} conf={p['confidence']:.2f}")

# Also test on real test_input.jpg
import os
real = r"d:\cctv face reconstruction\test_input.jpg"
if os.path.exists(real):
    real_img = cv2.imread(real)
    plates2 = plate_detector.detect_plates(real_img)
    print(f"\n[test_input.jpg] Detected {len(plates2)} plate(s):")
    for p in plates2:
        print(f"  x={p['x']} y={p['y']} w={p['w']} h={p['h']} conf={p['confidence']:.2f}")
