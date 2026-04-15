import os
import urllib.request

SAMPLES = {
    "car_plate_1.jpg": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "face_zidane.jpg": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
    
    # Face Tests (Historical / Needs Enhancement)
    "blurry_face_1.png": "https://raw.githubusercontent.com/TencentARC/GFPGAN/master/inputs/whole_imgs/10045.png",
    "blurry_face_2.jpg": "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/inputs/0014.jpg"
}

def main():
    out_dir = "sample_data"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Downloading samples into P.C. folder: '{os.path.abspath(out_dir)}' ...")
    for name, url in SAMPLES.items():
        try:
            out_path = os.path.join(out_dir, name)
            print(f"Fetching {name} ...")
            req = urllib.request.Request(
                url, 
                data=None, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"  [+] Saved {out_path}")
        except Exception as e:
            print(f"  [-] Failed to download {name}: {e}")

if __name__ == "__main__":
    main()
