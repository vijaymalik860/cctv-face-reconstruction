"""Test plate detector on the actual uploaded car image."""
import cv2
import sys
sys.path.insert(0, r"d:\cctv face reconstruction")

img_path = r"d:\cctv face reconstruction\uploads\plates\5b320c6a-db22-452d-8db2-ad00d8a22295.jpeg"
img = cv2.imread(img_path)
if img is None:
    print("ERROR: Could not read image!")
    sys.exit(1)

print(f"Image shape: {img.shape}")
print(f"Mean brightness: {img.mean():.1f}")

import numpy as np
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Gray mean: {gray.mean():.1f}")

# Test white region detection directly
_, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
white_pixels = cv2.countNonZero(white_mask)
print(f"White pixels (>180): {white_pixels} out of {gray.size}")

from app.services.plate_detector import plate_detector

plates = plate_detector.detect_plates(img)
print(f"\nTotal plates detected: {len(plates)}")
for i, p in enumerate(plates):
    print(f"  Plate {i+1}: x={p['x']} y={p['y']} w={p['w']} h={p['h']} conf={p['confidence']:.2f}")

if plates:
    # Save annotated result
    annotated = plate_detector.draw_plates(img, plates)
    cv2.imwrite(r"d:\cctv face reconstruction\test_real_annotated.jpg", annotated)
    print("\nSaved: test_real_annotated.jpg")
