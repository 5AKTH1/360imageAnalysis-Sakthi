import cv2

# --- CHANGE THIS TO YOUR IMAGE PATH ---
image_path = r"C:\IITM\DATA_RETRIEVAL_TASK\Extraction_2026-02-02_13-53-32\Final_Static_Stitch.jpg"

img = cv2.imread(image_path)
points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # x and y will be the TRUE pixel coordinates of the original large image!
        points.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.line(img, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
        cv2.imshow('Draw Polygon (Press Q when done)', img)

print("1. Click along the edge of your vehicle/rig from LEFT to RIGHT.")
print("2. Click all the way down into the bottom corners to close the shape.")
print("3. Press 'q' when you are finished.")

# --- THE FIX ---
window_name = 'Draw Polygon (Press Q when done)'
# cv2.WINDOW_NORMAL allows the window to be resized manually or via code
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# Resize the display window to fit easily on a standard laptop screen (e.g., 1600x800)
cv2.resizeWindow(window_name, 1600, 800) 

cv2.imshow(window_name, img)
cv2.setMouseCallback(window_name, click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n--- COPY AND PASTE THIS INTO YOUR MAIN SCRIPT ---")
print(f"self.EGO_POLYGON = np.array({points}, np.int32)")