"""
Diagnostic: Test plate models on a synthetic car-plate image.
Run: venv\Scripts\python test_plate_model.py
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

MODEL_DIR = Path(r"d:\cctv face reconstruction\models")

# ── Create synthetic test image with a white plate ──────────────────────────
img = np.full((480, 640, 3), 100, dtype=np.uint8)
cv2.rectangle(img, (150, 300), (490, 370), (255, 255, 255), -1)
cv2.rectangle(img, (150, 300), (490, 370), (0, 0, 0), 2)
cv2.putText(img, "HR 13N 2950", (165, 348), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# ── Also test on real test_input.jpg if it exists ───────────────────────────
real_img_path = Path(r"d:\cctv face reconstruction\test_input.jpg")
real_img = cv2.imread(str(real_img_path)) if real_img_path.exists() else None

models_to_test = [
    "indian_license_plate.pt",
    "yolov8_license_plate.pt",
]

for model_name in models_to_test:
    path = MODEL_DIR / model_name
    if not path.exists():
        print(f"[SKIP] {model_name} not found")
        continue

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Classes: {YOLO(str(path)).names}")
    print(f"{'='*60}")

    model = YOLO(str(path))

    for conf in [0.25, 0.10, 0.05]:
        for label, test_img in [("synthetic", img), ("real", real_img)]:
            if test_img is None:
                continue
            results = model(test_img, conf=conf, verbose=False)
            dets = []
            for r in results:
                if r.boxes:
                    for b in r.boxes:
                        dets.append(
                            f"cls={model.names[int(b.cls[0])]} "
                            f"conf={float(b.conf[0]):.3f} "
                            f"box={[round(v) for v in b.xyxy[0].tolist()]}"
                        )
            status = ", ".join(dets) if dets else "NO DETECTIONS"
            print(f"  [{label}] conf>={conf}: {status}")

print("\nDone.")
