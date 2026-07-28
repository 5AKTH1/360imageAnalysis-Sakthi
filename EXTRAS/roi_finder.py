import cv2

# Load your stitched image
image_path = "C:\IITM\DATA_RETRIEVAL_TASK\Extraction_2026-01-22_16-29-28\Final_Static_Stitch.jpg" 
img = cv2.imread(image_path)

if img is None:
    print(f"Could not load {image_path}. Make sure the file name is correct.")
    exit()

# Create a resizable window because your panorama is huge (3328x1664)
cv2.namedWindow("Select Crop Region", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select Crop Region", 1600, 800)

print("--- INSTRUCTIONS ---")
print("1. Click and drag to draw a box around the area you want to KEEP (ignore the curved black edges).")
print("2. Press SPACE or ENTER to confirm your selection.")
print("3. Press 'c' to cancel and try again.")

# Open the interactive selector
roi = cv2.selectROI("Select Crop Region", img, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

# Unpack the bounding box values
x, y, w, h = roi

# If a valid box was drawn
if w > 0 and h > 0:
    total_height = img.shape[0]
    
    # Calculate how many pixels to cut from the top and bottom
    crop_top = y
    crop_bottom = total_height - (y + h)
    
    print("\n" + "="*40)
    print("   YOUR EXACT CROP VALUES")
    print("="*40)
    print(f"self.CROP_TOP = {crop_top}")
    print(f"self.CROP_BOTTOM = {crop_bottom}")
    print("="*40 + "\n")
    print("Copy and paste these numbers directly into your ClipVideoStitcher class!")
else:
    print("Selection cancelled. No box was drawn.")