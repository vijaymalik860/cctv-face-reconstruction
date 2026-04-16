import os
import requests

# Define the models and their download URLs
# NOTE: For custom trained models, upload them to HuggingFace, GitHub Releases, or Google Drive 
# with direct link capabilities and replace the 'TODO_YOUR_...' URLs.
MODELS = {
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    
    # Custom or specific models - Placeholders (Replace with actual direct download links)
    "yolov8n_license_plate.pt": "TODO_YOUR_LINK_HERE",
    "indian_license_plate.pt": "TODO_YOUR_LINK_HERE",
    "yolov8_license_plate.pt": "TODO_YOUR_LINK_HERE",
    "FSRCNN_x2.pb": "TODO_YOUR_LINK_HERE",
    "FSRCNN_x4.pb": "TODO_YOUR_LINK_HERE",
    "wpod_net.h5": "TODO_YOUR_LINK_HERE",
    "wpod_net.json": "TODO_YOUR_LINK_HERE"
}

def download_file(url, dest_path):
    if url.startswith("TODO"):
        print(f"Skipping {os.path.basename(dest_path)}: URL not configured. Upload your model and update the script.")
        return

    if os.path.exists(dest_path):
        print(f"Already exists: {dest_path}")
        return

    print(f"Downloading {os.path.basename(dest_path)}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Successfully downloaded: {dest_path}")
    except Exception as e:
        print(f"Failed to download {dest_path}: {e}")
        # Clean up partial file
        if os.path.exists(dest_path):
            os.remove(dest_path)

if __name__ == "__main__":
    print("Starting model downloads...")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    for model_name, url in MODELS.items():
        dest_path = os.path.join(models_dir, model_name)
        download_file(url, dest_path)
    
    print("\nDownload process completed.")
