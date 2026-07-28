import torch
import os
from ultralytics import YOLO

def main():
    # Clear GPU memory cache
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 1. Load the model
    model = YOLO("yolo11m.pt")

    # 2. Start the training
    results = model.train(
        data="C:/IITM/Human_detection_360/CrowdHuman.v3-blurhumanfinal.yolov11/data.yaml",
        epochs=50,
        imgsz=640,       # Start with 640 to ensure stability on 4GB VRAM
        batch=4,         
        fraction=0.26,   # <--- NEW: Uses ~5,000 images for training
        rect=True,       # Saves memory
        device=0,
        workers=2,       
        project="C:/IITM/Human_detection_360/CrowdHuman_Train",
        name="pedestrian_model_subset"
    )

if __name__ == '__main__':
    main()