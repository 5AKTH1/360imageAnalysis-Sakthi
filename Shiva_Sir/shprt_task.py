from pysift import PySIFT
import cv2

sift = PySIFT()
gray = cv2.imread(r"C:\IITM\DATA_RETRIEVAL_TASK\Extraction_2026-01-22_16-29-28\Final_Static_Stitch.jpg", cv2.IMREAD_GRAYSCALE)
kp, desc = sift.detectAndCompute(gray, None)
print(f"Detected {len(kp)} keypoints, descriptor shape: {desc.shape}")