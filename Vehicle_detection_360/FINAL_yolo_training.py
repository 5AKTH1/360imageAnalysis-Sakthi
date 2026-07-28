from ultralytics import YOLO

def train_custom_model():
    # 1. Load the YOLOv11 model (starting from pretrained weights)
    model = YOLO("yolo11n.pt") # 'n' for nano is best for RTX 3050 speed/stability

    # 2. Start training
    results = model.train(
        data=r"C:\IITM\Vehicle_detection_360\Balanced_vehiclesyolov11_Dataset\data.yaml", # Path to your balanced dataset
        epochs=100,            # Balanced datasets often converge faster
        imgsz=640,             # Standard resolution
        batch=16,              # Adjust to 8 if you get 'Out of Memory'
        device=0,              # Uses your RTX 3050
        workers=4,             # Parallel data loading
        project="runs/detect", # Where results are saved
        name="balanced_v11_6classes",
        
        # --- ENHANCED AUGMENTATION FOR BALANCED DATA ---
        mosaic=1.0,            # Combines 4 images to help with small objects (Autos/Bikes)
        mixup=0.1,             # Blends two images to improve occlusion handling
        perspective=0.0005,    # Helps the model understand 360-degree lens distortion
        
        # --- TRAINING STABILITY ---
        patience=20,           # Early stopping if model stops improving
        save=True,             # Save checkpoints
        exist_ok=True          # Overwrite if folder exists
    )

if __name__ == "__main__":
    train_custom_model()