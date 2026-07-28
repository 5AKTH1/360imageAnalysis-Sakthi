import cv2
import json

# Define the exact order of your 6 seams
seam_names = [
    "Lens 3 to Lens 2",
    "Lens 2 to Lens 1",
    "Lens 1 to Lens 6",
    "Lens 6 to Lens 5",
    "Lens 5 to Lens 4",
    "Lens 4 to Lens 3"
]

current_seam_idx = 0
points = []
results = {}

# Global variables for image states
original_img = None
display_img = None

def click_event(event, x, y, flags, param):
    global current_seam_idx, points, display_img, results

    # Only register clicks if we haven't finished all 6 seams
    if event == cv2.EVENT_LBUTTONDOWN and current_seam_idx < len(seam_names):
        points.append(x)
        
        # Draw a red vertical line where the user clicked
        cv2.line(display_img, (x, 0), (x, display_img.shape[0]), (0, 0, 255), 4)
        cv2.imshow("Mark Seam Zones", display_img)

        # If 2 points are collected, the seam zone is complete
        if len(points) == 2:
            x1, x2 = points[0], points[1]
            left_x = min(x1, x2)
            right_x = max(x1, x2)
            width = right_x - left_x
            
            seam = seam_names[current_seam_idx]
            results[seam] = {"left_bound": left_x, "right_bound": right_x, "width_pixels": width}
            
            print(f"[+] {seam} Logged! | Width: {width}px | Bounds: {left_x} to {right_x}")

            # Draw a semi-transparent green box over the completed seam zone
            overlay = display_img.copy()
            cv2.rectangle(overlay, (left_x, 0), (right_x, display_img.shape[0]), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)
            
            # Reset for the next seam
            points = []
            current_seam_idx += 1

            if current_seam_idx < len(seam_names):
                print(f"\n---> Next up: Click the Left and Right boundaries for {seam_names[current_seam_idx]}")
            else:
                print("\n🎉 All 6 seams marked! Press 's' to save and quit, or 'q' to quit without saving.")
            
            cv2.imshow("Mark Seam Zones", display_img)

def main(image_path):
    global original_img, display_img, current_seam_idx, points, results
    
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("Error: Could not load image.")
        return

    display_img = original_img.copy()

    # Create a resizable window (Crucial for massive 360/stitched images)
    cv2.namedWindow("Mark Seam Zones", cv2.WINDOW_NORMAL)
    # Start the window at a reasonable 1080p size so it doesn't break your monitor
    cv2.resizeWindow("Mark Seam Zones", 1920, 1080) 
    
    cv2.setMouseCallback("Mark Seam Zones", click_event)

    print("==================================================")
    print("              SEAM ZONE MEASUREMENT               ")
    print("==================================================")
    print(f"---> Start by clicking the Left and Right boundaries for {seam_names[current_seam_idx]}")
    print("Controls:")
    print(" - Click to mark boundaries.")
    print(" - Press 'r' to restart if you make a mistake.")
    print(" - Press 'q' to quit.")
    print("==================================================\n")

    cv2.imshow("Mark Seam Zones", display_img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' to quit
        if key == ord('q'):
            break
            
        # 'r' to reset everything
        elif key == ord('r'):
            print("\n[!] Resetting all progress...")
            display_img = original_img.copy()
            current_seam_idx = 0
            points = []
            results.clear()
            cv2.imshow("Mark Seam Zones", display_img)
            print(f"---> Start by clicking the Left and Right boundaries for {seam_names[current_seam_idx]}")
            
        # 's' to save
        elif key == ord('s') and current_seam_idx == len(seam_names):
            with open("seam_measurements.json", "w") as f:
                json.dump(results, f, indent=4)
            print("\n[SUCCESS] Measurements saved to 'seam_measurements.json'")
            break

    cv2.destroyAllWindows()

# --- RUN SCRIPT ---
main(r"C:\IITM\DATA_RETRIEVAL_TASK\Extraction_2026-02-01_14-31-02\Final_Static_Stitch.jpg")