# CCTV Face & License Plate Enhancer

An AI-powered web application designed to reconstruct and enhance low-quality, blurry, or low-light faces as well as detect and recognize vehicle license plates typically captured from CCTV footage. This project uses advanced deep learning models to upscale, restore details, and perform accurate fast object detection entirely locally.

## Features

### 👤 Face Enhancement
- Upscales and clears blurry CCTV facial crops using models like **GFPGAN** and **Real-ESRGAN**.
- Provides side-by-side comparisons of raw vs. enhanced images.

### 🚗 License Plate Detection & Recognition (ALPR)
- **Indian License Plate Detection:** Utilizes specialized YOLOv8 models (`indian_license_plate.pt`) for accurate vehicle and plate localization.
- **Perspective Correction:** Employs WPOD-NET and MSER-based fallback logic to rectify skewed plates from various angles.
- **Night / Low-Light Processing:** Dedicated processing pipeline including CLAHE, dehazing, advanced deblurring, and super-resolution to extract text from dark or deeply blurred plates.
- **OCR & Text Cleanup:** Integrates robust OCR engines and customized post-processing to clean up Indian plate formats and remove false positives.
- **Video Analysis:** Upload videos directly with a custom Frame Selection UI to selectively process sequence intervals (skipping frames) to save computing time while tracking moving plates.

### 🔒 Fast Local Processing
- Processes images and videos natively on your machine without sending sensitive data over the internet.

## Prerequisites
- **Python 3.8+**
- Git

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/vijaymalik860/cctv-face-reconstruction.git
   cd cctv-face-reconstruction
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download AI Models (Important!)**
   Because AI models are very large, they are not included in this repository. We have provided a script to automate the download process for standard models.
   
   Run the following command to download the models:
   ```bash
   python download_models.py
   ```
   *Note for Custom Models:* For your custom trained models (like `indian_license_plate.pt` and `wpod_net.h5`), you will need to upload them to cloud storage (like Google Drive, HuggingFace, or GitHub Releases), get a direct download link, and update the placeholder URLs inside `download_models.py`, or simply copy those files manually into the `models/` folder.

5. **Setup Environment Variables**
   - Create a file named `.env` in the root folder.
   - Add any necessary configurations, ports, or secret keys required by the app inside this `.env` file.

## Running the Application

**Option 1 (Using Batch Script for Windows):**
Simply double-click the `Start_App.bat` file in the root folder.

**Option 2 (Using Terminal):**
```bash
python run.py
```
Open your web browser and navigate to `http://127.0.0.1:5000` (or whatever port is specified).

## Security & Privacy Note
This repository is configured to exclude sensitive files such as `.env` and heavy local models/caches (like `venv/` or `.pth` weight files) for security and storage best practices. Do not commit these files to GitHub.
