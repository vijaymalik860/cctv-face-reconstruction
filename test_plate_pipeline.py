import cv2, sys

# Load sample car image
img = cv2.imread('sample_data/car_plate_1.jpg')
if img is None:
    print('ERROR: Could not load sample image')
    sys.exit(1)

print(f'Image loaded: {img.shape[1]}x{img.shape[0]} px')

# Test Stage 1: plate detector
from app.services.plate_detector import plate_detector
print('PlateDetector initialized OK')

# Test detection
print('\nRunning detect_plates()...')
plates = plate_detector.detect_plates(img, conf_threshold=0.25)
print(f'  Plates found: {len(plates)}')
for i, p in enumerate(plates):
    print(f'  Plate {i+1}: x={p["x"]} y={p["y"]} w={p["w"]} h={p["h"]} conf={p["confidence"]}')

# Test rectifier load
from app.services.plate_rectifier import plate_rectifier
print('\nPlateRectifier imported OK')

# Test OCR helpers
from app.services.plate_ocr import plate_ocr, _pick_best_candidate
print('PlateOCR imported OK')

test_parts = ['HR', '26', 'AB', '1234']
result = _pick_best_candidate(test_parts)
print(f'\n_pick_best_candidate test: {test_parts} -> "{result}"')

# If plates found, run a quick rectify+OCR on first plate
if plates:
    p = plates[0]
    crop = img[p['y']:p['y']+p['h'], p['x']:p['x']+p['w']]
    if crop.size > 0:
        rectified = plate_rectifier.rectify(crop)
        print(f'Rectified shape: {rectified.shape}')
        ocr_out = plate_ocr.read_text(rectified)
        print(f'OCR result: "{ocr_out["text"]}" (conf={ocr_out["confidence"]:.2f})')

print('\nAll services verified successfully!')
