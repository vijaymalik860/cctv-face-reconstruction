# CCTV Face Reconstruction

An AI-powered web application designed to reconstruct and enhance low-quality, blurry, or low-light faces typically captured from CCTV footage. This project uses advanced deep learning models to upscale and restore facial details securely locally.

## Features
- **Face Enhancement:** Upscales and clears blurry CCTV facial crops using models like GFPGAN and Real-ESRGAN.
- **Fast Local Processing:** Processes images natively on the machine without sending sensitive data over the internet.
- **Web Interface:** Easy-to-use aesthetic frontend to easily upload raw images and view enhanced results side-by-side.

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
   Because AI models are very large, they are not included in this repository. You must download them and place them in the correct directories:
   - Download GFPGAN/Real-ESRGAN `.pth` model weights.
   - Place them inside the newly created `models/` folder.
   - Example path: `models/GFPGANv1.4.pth` 
   - (Create the `models/` or `gfpgan/weights/` directories if they do not exist).

5. **Setup Environment Variables**
   - Create a file named `.env` in the root folder.
   - Add any necessary database credentials, ports, or secret keys required by the app inside this `.env` file.

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
