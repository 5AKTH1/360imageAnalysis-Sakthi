import os
import cv2

def create_resized_copies(base_dir, lens_folder="lens 1"):
    lens_path = os.path.join(base_dir, lens_folder)
    
    # Find the source 4K video
    video_files = [f for f in os.listdir(lens_path) if f.startswith("undistorted_") and f.endswith(".mp4")]
    if not video_files:
        print("No source video found!")
        return

    source_path = os.path.join(lens_path, video_files[0])
    cap = cv2.VideoCapture(source_path)
    
    # Get Original Specs
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Source: {source_path} | {orig_w}x{orig_h} @ {fps} FPS")
    
    # Target Heights
    targets = [1080]
    
    for h in targets:
        aspect_ratio = orig_w / orig_h
        w = int(h * aspect_ratio)
        
        output_filename = f"source_{h}p.mp4"
        output_path = os.path.join(lens_path, output_filename)
        
        if os.path.exists(output_path):
            print(f"Skipping {output_filename} (Already exists)")
            continue
            
        print(f"Generating {output_filename} ({w}x{h})...")
        
        # Reset Source Reader
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            out.write(resized)
            
            count += 1
            if count % 500 == 0:
                print(f"  Encoded {count}/{total_frames} frames...", end="\r")
        
        out.release()
        print(f"\nSaved: {output_path}")

    cap.release()
    print("Pre-processing Complete.")

if __name__ == "__main__":
    BASE_DIR = r"C:\IITM\CAMERA_Short_Clips"
    create_resized_copies(BASE_DIR)