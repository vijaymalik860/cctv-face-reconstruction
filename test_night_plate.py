"""
Quick test — night-time plate detection on a dark CCTV image.
Usage: python test_night_plate.py path/to/image.jpg
"""
import sys, cv2, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from app.services.plate_detector import plate_detector
from app.services.enhancer       import face_enhancer

IMG = sys.argv[1] if len(sys.argv) > 1 else "test_input.jpg"
print(f"\n=== Testing night plate detection on: {IMG} ===\n")

image = cv2.imread(IMG)
if image is None:
    print(f"ERROR: cannot read {IMG}")
    sys.exit(1)

print(f"Image size: {image.shape[1]}×{image.shape[0]}")
is_night = plate_detector._is_night(image)
print(f"Night detected: {is_night}")

if is_night:
    print("\n--- Night preprocessing preview ---")
    nv = plate_detector.preprocess_night_vision(image)
    cv2.imwrite("night_preprocessed.jpg", nv)
    print("Saved: night_preprocessed.jpg")

print("\n--- Detecting plates ---")
plates = plate_detector.detect_plates(image, conf_threshold=0.10)
print(f"Plates found: {len(plates)}")

if plates:
    annotated = plate_detector.draw_plates(image, plates)
    cv2.imwrite("night_detected.jpg", annotated)
    print("Saved: night_detected.jpg")

    crops = plate_detector.crop_plates(image, plates)
    for i, (meta, crop) in enumerate(zip(plates, crops)):
        raw_path = f"night_plate_{i}_raw.jpg"
        cv2.imwrite(raw_path, crop)
        print(f"\nPlate {i}: conf={meta['confidence']:.3f}  night={meta['is_night_vision']}")
        print(f"  Raw saved: {raw_path}")

        enhanced = face_enhancer.enhance_plate(
            crop, upscale=4, is_night=meta['is_night_vision']
        )
        enh_path = f"night_plate_{i}_enhanced.jpg"
        cv2.imwrite(enh_path, enhanced)
        print(f"  Enhanced saved: {enh_path}  size={enhanced.shape[1]}×{enhanced.shape[0]}")
else:
    print("\nNo plates detected.")
    print("Tip: Try with --force_night or check if YOLO model exists in models/")

print("\n=== Done ===")
