import cv2
from app.services.plate_detector import plate_detector
from app.services.plate_rectifier import plate_rectifier
from app.services.plate_ocr import plate_ocr

img = cv2.imread('sample_data/car_plate_1.jpg')
print(f'Image: {img.shape}')

plates = plate_detector.detect_plates(img)
print(f'Plates found: {len(plates)}')
for p in plates:
    print(' ', p)

if plates:
    crops = plate_detector.crop_plates(img, plates)
    for i, crop in enumerate(crops):
        rect = plate_rectifier.rectify(crop)
        result = plate_ocr.read_text(rect)
        print(f'Plate {i+1} OCR: "{result["text"]}" (conf={result["confidence"]:.2f})')

print('Done.')
