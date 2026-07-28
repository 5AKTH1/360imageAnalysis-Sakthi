# MUMMAS 360 Camera - Data Collection and AI Processing Repository

> **IITM Research Project** | Multi-sensor Urban Mobility Monitoring and Analysis System
> 6-Lens 360 Camera - LiDAR - IMU/GPS - AI Detection Pipeline

---

## File Naming Convention

> **Important:** Files and folders prefixed with **FINAL_** are the **canonical, production-ready, and actively maintained** scripts. All other files without this prefix are **previous versions, experimental drafts, or auxiliary extras** and should not be used for active processing unless specifically noted.

---

## Repository Structure

`
IITM/
|-- FINAL_Intrinsics/              <- Camera calibration JSON files (all 6 lenses)
|-- PIPELINE/                      <- [MAIN] End-to-end data retrieval and AI pipeline
|-- Data_Preparation/              <- Video trimming, undistortion, and stitching scripts
|-- Human_detection_360/           <- Pedestrian detection on 360 panoramic video
|-- Vehicle_detection_360/         <- Vehicle detection and tracking on 360 video
|-- Parked_vs_Moving_Detection/    <- LiDAR-fused parked vs moving vehicle classifier
|-- CAMERA_Short_Clips/            <- Short raw clip samples for testing
|-- IMU-GPS/                       <- Raw IMU/GPS CSV sensor data
|-- DATA_RETRIEVAL_TASK/           <- Output artefacts from batch pipeline runs
|-- Results/                       <- Final output videos, images, and detection results
|-- RUMMAIYA/                      <- Spatiotemporal trip manifests and LiDAR-fisheye projection
|-- Shiva_Sir/                     <- GPS extraction utilities and S2 building data
|-- princeton_team/                <- Princeton collaboration batch processing CSVs
|-- Object_detection_test/         <- Quick YOLO test on a single Street View image
|-- Pano360_Banus_Stitching_test/  <- Early OpenCV feature-matching stitcher experiments
|-- EXTRAS/                        <- Miscellaneous utility scripts (non-final)
|-- presentations/                 <- Project reports, slide decks, and research documents
|-- yolov8n.pt                     <- Pre-trained YOLOv8 Nano weights
|-- yolo11n.pt                     <- Pre-trained YOLO11 Nano weights
|-- Insta360Stitcher_Winx64_4.0.0.exe <- Official Insta360 stitcher software (Windows)
|-- ffmpeg-8.0.1-essentials_build.zip <- FFmpeg binaries (required dependency)
|-- python-3.11.9-amd64.exe        <- Python 3.11 installer
\\\

---

## PIPELINE - Main End-to-End Hub

> This is the **primary entry point** for all data retrieval and AI processing.

### FINAL Files (Active and Executable)

---

#### \FINAL_sample.py\ - Master Pipeline Script

| | |
|---|---|
| **Input** | Interactive CLI prompts: (1) path to raw data drive root (e.g. \D:\\MUMMAS DATA COLLECTION\), (2) master output directory path, (3) a timestamp / frame number / CSV file of timestamps, (4) menu selections for processing level |
| **Output** | Undistorted lens frames or videos, stitched 360 panorama image/video, vehicle and pedestrian detection JSON files, annotated output video, and an updated \_processed.csv\ with per-timestamp detection counts |
| **Purpose** | Full-featured menu-driven pipeline that takes a timestamp or batch CSV of timestamps, searches the raw data drive, undistorts all 6 lens videos, stitches them into a 360 panorama, runs YOLO vehicle and pedestrian detection, and writes all results including vehicle counts back into the user spreadsheet |

---

#### \FINAL_metadata_extraction.py\ - Session Metadata Extractor

| | |
|---|---|
| **Input** | Configured path to the hard-drive root (\ROOT_DIR\) containing date-organised session folders with \CAMERA/\, \IMU/\, \LIDAR/\ sub-folders |
| **Output** | A \metadata.json\ file per date folder summarising trip/run start-end times, frame counts per lens, IMU file paths, LiDAR file paths, and a \	imestamps_metadata_DDMMYYYY.csv\ per date |
| **Purpose** | Recursively walks the data drive and generates structured JSON metadata for every recording session, enabling the pipeline fast timestamp-to-clip lookup |

---

#### \FINAL_availability_check.py\ - Timestamp Availability Checker

| | |
|---|---|
| **Input** | A root drive path to index \metadata.json\ files and a target datetime object representing a timestamp to query |
| **Output** | A tuple \(yes/no, reason_string)\ indicating whether camera and IMU data is available for that exact timestamp |
| **Purpose** | Library module used by the pipeline to rapidly verify data availability for any given timestamp before attempting extraction, reporting missing lenses or missing IMU data clearly |

---

#### \FINAL_4k_to_1080p.py\ - Batch 4K to 1080p Video Compressor

| | |
|---|---|
| **Input** | \SOURCE_FOLDER\ path configured at bottom of script: root folder of raw 4K .mp4 files (e.g. \G:\\MUMMAS DATA COLLECTION\) |
| **Output** | A mirrored directory tree at \DESTINATION_FOLDER\ with all .mp4 files re-encoded to 1080p using CUDA-accelerated H.264, and all non-video files copied as-is |
| **Purpose** | Mirrors an entire data drive while downscaling all 4K camera videos to 1080p using NVIDIA GPU acceleration via FFmpeg to save storage space for downstream processing |

---

### Non-Final Files (Previous Versions / Extras)

| File | Purpose |
|------|---------|
| \sample.py\ | Older version of \FINAL_sample.py\; retained for reference |
| \vailability_scanner_from_pipeline.py\ | Earlier standalone availability scanner; superseded by \FINAL_availability_check.py\ |
| \detection_from_pipeline.py\ | Standalone detection module extracted from pipeline; for reference only |
| \extract_undistort_from_pipeline.py\ | Standalone undistortion module; for reference only |
| \stitcher_from_pipeline.py\ | Standalone stitcher module; for reference only |
| \ReadMe.Md\ | Detailed user-facing operation guide for the PIPELINE (step-by-step instructions) |
| \ReadMe.txt\ | Plain-text version of the operation guide |

---

## Data_Preparation - Video Pre-processing

> Scripts to trim raw recordings into short clips, undistort fish-eye lenses, and stitch frames into panoramas.

### FINAL Files (Active and Executable)

---

#### \FINAL_video_trimmer.py\ - Raw Video Clipper

| | |
|---|---|
| **Input** | \BASE_DIR\: path to a camera run folder containing \LENS1/\ through \LENS6/\ sub-folders with \ideo_lens{N}.mp4\ files; \START_TIME\ and \END_TIME\ in HH:MM:SS format configured in the script |
| **Output** | 6 trimmed .mp4 clips in \OUTPUT_DIR\ named \origin_0_SESSION001.mp4\ through \origin_5_SESSION001.mp4\ in Insta360-compatible naming format |
| **Purpose** | Trims all 6 lens videos to a common time window using FFmpeg stream-copy (lossless, no re-encoding) and renames them to the format expected by the Insta360 stitcher and downstream pipeline scripts |

---

#### \FINAL_undistort_omni_videos.py\ - Omnidirectional Lens Undistortion

| | |
|---|---|
| **Input** | \ase_dir\: folder containing \lens 1/\ through \lens 6/\ sub-folders with \processed_*.mp4\ videos; \intrinsics_file\: path to a \calibration_omni.json\ omnidirectional camera model file |
| **Output** | \undistorted_*.mp4\ videos in each lens folder, plus \calibration_pinhole_lens{N}.json\ files written to the \FINAL_Intrinsics/\ folder |
| **Purpose** | Applies omnidirectional Scaramuzza lens undistortion to each of the 6 fish-eye camera feeds, converting them to standard pinhole-model rectilinear videos and saving the derived pinhole intrinsic parameters for stitching |

---

#### \FINAL_clean_stitcher.py\ - SIFT-Based 360 Video Stitcher

| | |
|---|---|
| **Input** | \ase_dir\: folder containing \lens 1/\ through \lens 6/\ sub-folders with \undistorted_*.mp4\ videos; \intrinsics_dir\: folder containing \calibration_pinhole_lens{N}.json\ files from \FINAL_Intrinsics/\ |
| **Output** | A single stitched equirectangular panoramic video at 3328x1664 resolution written to \ase_dir\; filename prompted at runtime |
| **Purpose** | Stitches 6 undistorted lens streams into a seamless 360 equirectangular panoramic video using pre-optimised SIFT-based geometry, per-lens zoom calibration, feathered weight blending, and configurable crop margins |

---

#### \FINAL_prepare_videos.py\ - Resolution Downscaler for Test Clips

| | |
|---|---|
| **Input** | \BASE_DIR\: directory containing a \lens 1/\ sub-folder with an \undistorted_*.mp4\ source video |
| **Output** | A resized \source_1080p.mp4\ saved in the same \lens 1/\ folder |
| **Purpose** | Creates a 1080p copy of an undistorted lens video (maintaining original aspect ratio) for lightweight testing and preview without the full stitching step |

---

### Non-Final Files

| File | Purpose |
|------|---------|
| \
ew_stitcher.py\ | Older stitcher using LoFTR/SIFT; superseded by \FINAL_clean_stitcher.py\ |
| \
ew_undistort_omni_videos.py\ | Earlier undistortion script; superseded by \FINAL_undistort_omni_videos.py\ |
| \	trimmer.py\ | Original trimmer script; superseded by \FINAL_video_trimmer.py\ |
| \undistort_omni_images.py\ | Applies undistortion to static images (not video); exploratory/reference |

---

## FINAL_Intrinsics - Camera Calibration Parameters

> Data files (not scripts) consumed by the pipeline. All lens calibration JSONs live here.

| File | Purpose |
|------|---------|
| \calibration_omni.json\ | Master omnidirectional Scaramuzza-model calibration shared across all 6 lenses |
| \calibration_pinhole_lens{1-6}.json\ | Derived pinhole intrinsic parameters for each lens after undistortion (auto-generated by \FINAL_undistort_omni_videos.py\) |
| \lens{1-6}_calibration_omni.json\ | Individual per-lens omnidirectional calibration files with full K, D, and xi parameters |

---

## Human_detection_360 - Pedestrian Detection

> AI-powered pedestrian detection and tracking on stitched 360 panoramic video.

### FINAL Files (Active and Executable)

---

#### \FINAL_panoramic_detection.py\ - 360 Pedestrian Detector and Tracker

| | |
|---|---|
| **Input** | \CUSTOM_MODEL_PATH\: fine-tuned YOLO11m .pt model; \SOURCE_VIDEO_PATH\: stitched 360 panoramic video; \VEHICLE_JSON_PATH\: vehicle detection JSON from \Vehicle_detection_360/FINAL_panoramic_detection.py\ (used to suppress false positives) |
| **Output** | Annotated output video with tracked pedestrian bounding boxes and IDs; JSON file with frame-by-frame pedestrian detections and track data |
| **Purpose** | Runs SAHI-sliced YOLO inference on each frame of a 360 panoramic video to detect pedestrians; applies ByteTrack tracking, ego-vehicle masking, vehicle-region suppression, and motion/speed filters to produce clean deduplicated pedestrian counts |

---

#### \FINAL_yolo_training.py\ - Pedestrian Model Fine-tuner

| | |
|---|---|
| **Input** | \CrowdHuman.v3-blurhumanfinal.yolov11/data.yaml\ dataset YAML; pre-trained \yolo11m.pt\ weights; inline training config (50 epochs, batch 4, 26% data fraction) |
| **Output** | Fine-tuned YOLO model checkpoints saved to \CrowdHuman_Train/pedestrian_model_subset/\ |
| **Purpose** | Fine-tunes a YOLO11m model on the CrowdHuman dataset using a memory-optimised subset to improve pedestrian detection accuracy for 360 fisheye-like panoramic imagery |

---

### Non-Final Files

| File | Purpose |
|------|---------|
| \seperate_lens_detection.py\ | Earlier per-lens pedestrian detector before stitching; superseded |
| \panoramic_detection.json\ | Cached output JSON from a previous panoramic detection run |
| \yolo11m.pt\ / \yolo11n.pt\ | YOLO model weights used during detection |
| \Datasets/\ | CrowdHuman training dataset (zip + extracted) |

---

## Vehicle_detection_360 - Vehicle Detection and Tracking

> YOLO-based vehicle detection with multi-lens strategies, track healing, and late-fusion deduplication.

### FINAL Files (Active and Executable)

---

#### \FINAL_panoramic_detection.py\ - 360 Vehicle Detector and Tracker

| | |
|---|---|
| **Input** | Fine-tuned YOLO .pt model path and stitched 360 panoramic video path (configured at bottom of script); optional \FINAL_custom_track.yaml\ for tracker settings |
| **Output** | Annotated tracking video with per-vehicle IDs and class labels; \FINAL_panoramic_detection2.json\ and \FINAL_panoramic_detection2_coco.json\ with full frame-by-frame detection data |
| **Purpose** | Performs YOLO vehicle detection on the full 360 panoramic video with dynamic confidence thresholds (distance-adaptive), ByteTrack/BotSORT tracking, broken-track healing, and sky/ground masking to produce clean vehicle counts |

---

#### \FINAL_seperate_lens_detection.py\ - Per-Lens Vehicle Detector with Late Fusion

| | |
|---|---|
| **Input** | Individual per-lens undistorted video files from \lens 1/\ through \lens 6/\ directories and a fine-tuned YOLO .pt model (path configured in script) |
| **Output** | Per-lens detection JSON files and a fused \FINAL_CV_Hybrid_detection.json\; optional annotated videos per lens |
| **Purpose** | Detects vehicles independently on each of the 6 lens videos, then applies late-fusion deduplication using IoU and seam-aware logic to merge overlapping cross-lens detections into a unified deduplicated vehicle count |

---

#### \FINAL_yolo_training.py\ - Vehicle Model Trainer

| | |
|---|---|
| **Input** | \data.yaml\ for the balanced 6-class vehicle dataset; pre-trained \yolo11n.pt\ weights; inline training config (100 epochs, mosaic/mixup augmentation) |
| **Output** | Trained model checkpoints saved to \
uns/detect/balanced_v11_6classes/\ |
| **Purpose** | Trains a YOLO11n model on a custom balanced dataset of 6 Indian street vehicle classes (Car, Auto, Bus, Truck, Motorbike) with augmentation tuned for 360 lens perspective distortion |

---

#### \FINAL_custom_track.yaml\ - BotSORT Tracker Configuration

> YAML configuration file for the BotSORT multi-object tracker used by \FINAL_panoramic_detection.py\. Sets thresholds for track confidence, buffer duration, appearance matching, and ORB-based global motion compensation.

---

### Non-Final / Output Files

| File | Purpose |
|------|---------|
| \FINAL_CV_Hybrid_detection.json\ | Fused vehicle detection output from \FINAL_seperate_lens_detection.py\ |
| \FINAL_panoramic_detection2.json\ / \_coco.json\ | Detection outputs in raw and COCO format |
| \FINAL_master_id_map.txt\ | Track ID remapping table generated after track healing |
| \panoramic_detection_SIFT_final.json\ | Detection output from a SIFT-stitched video run (reference) |
| \blation_seperate_lens_detection.py\ | Ablation study variant of the per-lens detector |
| \hybrid_seperate_lens_detection.py\ | Hybrid CV + YOLO detection experiment |
| \seperate_lens_detection.py\ | Older version; superseded by \FINAL_seperate_lens_detection.py\ |
| \panoramic_detection_CVAT.py\ | CVAT annotation export variant |
| \manual_stitch_detection_deduplication.py\ | Manual deduplication experiment |
| \blation_study_gpu.json\ | GPU ablation study benchmark results |
| \ehicles_detected_*.json\ | Various detection count summary JSONs |
| \datasets/\ \plots/\ \
uns/\ | Training dataset, plots, and YOLO run outputs |

---

## Parked_vs_Moving_Detection - LiDAR-Fused Parking Analysis

> Combines camera YOLO detections with synchronized LiDAR point clouds to classify vehicles as parked or moving and estimate distances.

### FINAL Files (Active and Executable)

---

#### \FINAL_lens1_detection_parked.py\ - LiDAR-Camera Fusion Parked Vehicle Classifier

| | |
|---|---|
| **Input** | IMU CSV file (\IMU_CSV\), two LiDAR PCAP files (\L1_PCAP\, \L2_PCAP\), lens 1 video (\VIDEO_PATH\), session metadata JSON (\META_JSON\), and pre-computed detections JSON (\DETECTIONS_JSON\) - all paths configured at top of script |
| **Output** | Annotated output video with bounding boxes colour-coded as parked (red) or moving (green), overlaid with LiDAR depth point cloud and distance estimates |
| **Purpose** | Fuses synchronised LiDAR point clouds with YOLO bounding box detections on lens 1 video to determine each detected vehicle real-world distance and classify it as parked or actively moving using telemetry-derived ego-speed |

---

#### \FINAL_lidar_pcap.py\ - LiDAR PCAP Parser and Loader

| | |
|---|---|
| **Input** | Path(s) to one or two Velodyne LiDAR .pcap files and optional time-window filters (\	_min\, \	_max\ as Unix timestamps) |
| **Output** | NumPy arrays of (timestamp, x, y, z) 3D point cloud data for the specified time window |
| **Purpose** | Parses raw Velodyne 16-beam LiDAR PCAP packets, converts azimuth/elevation angles and distances to Cartesian XYZ coordinates, and returns a filtered time-windowed point cloud with in-memory caching for efficiency |

---

#### \FINAL_perceive.py\ - LiDAR-to-Camera Projection and Depth Estimator

| | |
|---|---|
| **Input** | Camera intrinsics (fx, fy, cx, cy), a 4x4 LiDAR-to-camera extrinsic transformation matrix, and a NumPy XYZ point cloud array |
| **Output** | 2D projected pixel coordinates (u, v) and depth values (z) for each LiDAR point visible in the camera frame; median depth per bounding box |
| **Purpose** | Projects 3D LiDAR points into 2D camera image space using calibrated extrinsic transform, filters out-of-frame and behind-camera points, and estimates per-vehicle depth using histogram peak detection to lock onto the vehicle chassis and ignore road/background noise |

---

#### \FINAL_sync.py\ - IMU Telemetry Synchroniser

| | |
|---|---|
| **Input** | Path to an IMU/GPS CSV file (\imu_all_topics_*.csv\) containing \	_unix\, \ilter_lla_lat\, \ilter_lla_lon\, \yaw_deg\, \speed_mps\ columns |
| **Output** | Interpolated latitude, longitude, heading (yaw), and speed values for any given Unix timestamp via the \get_telemetry(t_unix)\ method |
| **Purpose** | Loads IMU/GPS telemetry data, builds cubic-interpolation models for position, heading, and speed over time, and provides a synchronisation API to look up exact vehicle ego-state at any video frame timestamp |

---

### Non-Final / Output Files

| File | Purpose |
|------|---------|
| \panoramic_detection_parked.py\ | Older panoramic parked vehicle detector without LiDAR fusion |
| \panoramic_detection_parked_optical_flow.py\ | Optical flow + YOLO parked vehicle experiment |
| \opticalFlow_YOLO_parked.json\ | Output from the optical flow detection run |
| \panoramic_detection_parked.json\ | Cached detection output from previous run |
| \parked_analytics_output.json\ | Analytics summary from parked vehicle detection |
| \output_tracked_lens1.mp4\ | Output video from a previous LiDAR-camera fusion run |

---

## RUMMAIYA - Spatiotemporal Data and LiDAR-Fisheye Projection

| File | Purpose |
|------|---------|
| \Lidar_fisheye_projection.py\ | **Input:** fisheye images (fisheye{1-6}.png) and a paired .pcd point cloud in \input_path\, plus extrinsic/intrinsic parameters in \ExParam_path\. **Output:** \projected_{i}.png\ images with colourised LiDAR depth points overlaid on each fisheye lens. **Purpose:** Projects LiDAR point cloud onto 6-lens fisheye images using pre-calibrated extrinsic transforms and Turbo colormap for depth visualisation |
| \Chennai_Daily_Grid_Time_Manifest_updated.csv\ | Raw time-grid manifest mapping Chennai spatial grid cells to data collection timestamps |
| \Chennai_Daily_Grid_Time_Manifest_updated_validated.csv\ | Validated version of the manifest after availability checking |
| \Expanded_Chennai_Manifest.csv\ | Full expanded trip manifest with per-timestamp entries |
| \Expanded_Chennai_Manifest_processed.csv\ | Processed subset of the expanded manifest |
| \Extracted_Raw_Peak_Trips_Data_processed.xlsx\ | Peak-hour trip data extracted and processed for analysis |
| \Master_Mapping.csv\ | Master lookup table mapping sessions to file paths |
| \pm_spatiotemporal_all_day_unique_data.csv\ | Unique spatiotemporal observation records for the full day |
| \unique_timestamps_200.csv\ | Sample of 200 unique timestamps used for pipeline testing |
| \Batch_Splits/\ | Chunked sub-batches of the manifest for parallel pipeline runs |

---

## Shiva_Sir - GPS Extraction and Building Data

| File | Purpose |
|------|---------|
| \gps_info_extractor.py\ | **Input:** IMU CSV path (\imu_all_topics_*.csv\). **Output:** A .gpx GPS track file with lat/lon/altitude/timestamp waypoints sampled at every 5th unique GPS position change. **Purpose:** Extracts a clean GPX route from the IMU/GPS CSV by filtering stationary duplicate readings, creating a file viewable in mapping tools |
| \extracted_gps_data.gpx\ | GPX route file generated by \gps_info_extractor.py\ for the Jan 17, 2026 trip |
| \VID_20260110_122108_file4.gpx\ | Raw GPX file from a mobile phone GPS recording |
| \gpx.fmt\ | ExifTool GPX format template used for GPX file generation |
| \Siva_sir_test.xlsx\ / \_processed.csv\ | Test spreadsheet and its processed output |
| \mock_timestamps.csv\ / \_processed.csv\ | Mock timestamp dataset for pipeline testing |
| \sample_imu.csv\ | Sample IMU/GPS data file for development testing |
| \mummas_stitcher_1080GPU_V4.py\ | Legacy large stitcher script (GPU-accelerated, v4); reference only |
| \shprt_task.py\ | Short utility task script; reference only |
| \S2_Building_extractor/\ | Tools for extracting building footprints using the S2 geometry library |
| \Results/\ | Output results from Shiva Sir-related processing runs |

---

## princeton_team - Collaboration Batch Processing

> CSVs for batch image processing tasks shared with the Princeton research team across three urban categories.

| File | Purpose |
|------|---------|
| \atch_imageproc_autostand.csv\ | Input batch manifest for auto-stand (rickshaw stand) locations |
| \atch_imageproc_autostand_matched.csv\ | Matched results where timestamps were found in the data drive |
| \atch_imageproc_autostand_processed.csv\ | Final processed output with detection counts |
| \atch_imageproc_autostand_unmatched.*\ | Unmatched entries and their exported analysis |
| \atch_imageproc_landfill.csv\ | Input batch manifest for landfill site locations |
| \atch_imageproc_landfill_matched.csv\ / \_processed.csv\ / \_unmatched.*\ | Landfill batch results (matched, processed, unmatched) |
| \atch_imageproc_transportation.csv\ | Input batch manifest for transportation hub locations |
| \atch_imageproc_transportation_matched.csv\ / \_processed.csv\ / \_unmatched.*\ | Transportation hub batch results |

---

## CAMERA_Short_Clips - Raw Test Clips

| File | Purpose |
|------|---------|
| \origin_0_SESSION001.mp4\ through \origin_5_SESSION001.mp4\ | Short raw clips from lenses 0 through 5 in Insta360 naming format used for development and stitching tests |
| \March26 Clips/\ | Additional short clip samples from March 2026 sessions |

---

## IMU-GPS - Raw Sensor Data

| File | Purpose |
|------|---------|
| \imu_all_topics_2026-01-17_15-21-18.csv\ | Full IMU/GPS sensor log from the January 17, 2026 data collection session (~500 MB); contains timestamps, GPS coordinates, yaw, speed, and accelerometer/gyroscope readings |

---

## Object_detection_test - Quick YOLO Test

| File | Purpose |
|------|---------|
| \yolo_test.py\ | Standalone YOLO inference test on a single Street View image; used for quick model validation |
| \Sample Street View 360.jpg\ | Sample 360 street view image used as input for the test |
| \inal_output_boosted.jpg\ | Output image from the YOLO test with detection bounding boxes |
| \Street View Download 360.exe\ | Tool for downloading 360 street view imagery |

---

## Pano360_Banus_Stitching_test - Early Stitcher Experiments

> Previous experimental work on OpenCV feature-matching stitching. Not actively maintained.

| File | Purpose |
|------|---------|
| \main_stitcher.py\ | Early main stitcher script using OpenCV Stitcher API |
| \eatures.py\ | Feature extraction and matching utilities (SIFT/ORB) |
| \undle_adj.py\ | Bundle adjustment experiment for camera pose refinement |

---

## EXTRAS - Miscellaneous Utility Scripts

> Auxiliary scripts for specific sub-tasks. Not part of the main pipeline.

| File | Purpose |
|------|---------|
| \draw.py\ | Utility to draw bounding boxes or annotations on images |
| \json_2_coco.py\ | Converts custom detection JSON format to COCO annotation format |
| \
esults_save_automated.py\ | Automates saving detection results to disk |
| \
oi_finder.py\ | Identifies region-of-interest zones in images |
| \seam_zone_detector.py\ | Detects seam zones between lens boundaries in panoramic images |
| \short_task.py\ | Short one-off task script |
| \seam_measurements.json\ | Measured seam pixel positions for each lens boundary |

---

## DATA_RETRIEVAL_TASK - Pipeline Run Outputs

> Output artefacts from pipeline batch runs. Each sub-folder corresponds to a specific extraction or pipeline execution.

| Type | Purpose |
|------|---------|
| \Extraction_YYYY-MM-DD_HH-MM-SS/\ | Folder containing undistorted frames/video extracted at a specific run time |
| \ClipPipeline_*/\ | Output from a clip-based pipeline run (video + detections) |
| \Pipeline_*/\ | Full pipeline output (stitched video + AI results) |
| \Mapping_*.csv\ | Timestamp-to-file mapping CSV generated during a batch run |

---

## Results - Final Outputs

| File | Purpose |
|------|---------|
| \Final_Dynamic_Stitch.mp4\ | Full-length dynamically stitched 360 panoramic video |
| \Final_Dynamic_Stitch_shortened.mp4\ | Shortened version used for detection runs |
| \Final_SIFT_Shifted_1080p.mp4\ | SIFT-based static stitch with horizontal shift correction at 1080p |
| \Final_Static_Stitch_LoFTR.mp4\ | Static stitch produced using the LoFTR deep-learning feature matcher |
| \LateFusion_ReID_LargeFont.mp4\ | Vehicle detection video with late-fusion ReID tracking and large annotation font |
| \SIFT_Shifted_1080p_DynamicDetect.mp4\ | SIFT stitch with dynamic vehicle detection overlay |
| \Test_Stitch_Frame_2.jpg\ / \	est_stitch_frame_1.jpeg\ | Sample stitched panorama frames |
| \Human_detection_360/\ | Pedestrian detection output videos and JSONs |
| \Parked_vs_Moving/\ | Parked vs moving vehicle detection output |
| \Vehicle_detection_360/\ | Vehicle detection output videos |
| \lens 1/\ through \lens 6/\ | Per-lens undistorted video outputs |

---

## presentations - Reports and Documentation

| File | Purpose |
|------|---------|
| \360 Camera Pipeline.docx\ | Technical overview document of the 360 camera processing pipeline |
| \4 different crop formats.docx\ | Analysis document comparing 4 panoramic crop/output formats |
| \Ablation_Study_Insights.pptx\ | Presentation summarising YOLO detection ablation study findings |
| \Annotations Required.docx\ | Specification document for required dataset annotation classes |
| \Intern_Progress_Tracking_Template.pptx\ | Project progress tracking slide deck |
| \MUMMAS DATA DIRECTORY STRUCTURE.docx\ | Reference document describing the expected data folder structure on the collection drive |
| \Multi-Lens Extrinsic Calibration Using Video Data.docx\ | Technical document on estimating inter-lens extrinsic calibration from video |
| \Parked vs Moving.docx\ | Report on the parked vs moving vehicle detection methodology |
| \Pedestrian_Detection_report.docx\ | Report on pedestrian detection approach and results |
| \Vehicle_Detection_09022026.docx\ | Vehicle detection results report dated Feb 9, 2026 |
| \Vehicle_detection_360_Report.docx\ | Comprehensive vehicle detection report for the 360 system |
| \PAIR360_...pdf\ | Reference paper: PAIR360 - A Paired Dataset of High-Resolution 360 Panoramic Images and LiDAR Scans |
| \comprehensive_stitching_report.txt\ | Plain-text stitching quality report |
| \detection_report.txt\ | Plain-text detection summary report |

---

## Root-Level Model and Tool Files

| File | Purpose |
|------|---------|
| \yolov8n.pt\ | Pre-trained YOLOv8 Nano model weights (general object detection baseline) |
| \yolo11n.pt\ | Pre-trained YOLO11 Nano model weights (used for fine-tuning) |
| \Insta360Stitcher_Winx64_4.0.0.exe\ | Official Insta360 Studio stitcher application for Windows |
| \fmpeg-8.0.1-essentials_build.zip\ | FFmpeg binaries archive; must be extracted and added to PATH |
| \python-3.11.9-amd64.exe\ | Python 3.11.9 Windows installer |
| \rchive.zip\ | Miscellaneous archived files |

---

## Setup and Dependencies

### Prerequisites

\\\ash
# Core ML and Vision
pip install opencv-python numpy torch torchvision ultralytics supervision

# Object Detection and Tracking
pip install sahi deep-sort-realtime

# Data and Utilities
pip install pandas scipy pytz dpkt

# Optional: 3D / LiDAR
pip install open3d
\\\

### Required External Tools
- **FFmpeg 8.0+** - Extract and add to system PATH
- **NVIDIA CUDA** - Required for GPU-accelerated inference and video encoding
- **Python 3.11** - Recommended version

### Required Calibration Files (not in repo - store in FINAL_Intrinsics/)
- \calibration_omni.json\ - Master omnidirectional calibration
- \lens{1-6}_calibration_omni.json\ - Per-lens omnidirectional calibrations
- \calibration_pinhole_lens{1-6}.json\ - Derived pinhole calibrations (auto-generated by \FINAL_undistort_omni_videos.py\)

### Required Model Weights (download separately)
- \est.pt\ - Custom-trained 6-class Indian vehicle detector
- \yolo11m.pt\ - Fine-tuned pedestrian detector (trained by \FINAL_yolo_training.py\)

---

## Quick Start

\\\ash
# 1. Run the full pipeline (interactive menu)
python PIPELINE/FINAL_sample.py

# 2. Trim raw clips from a hard drive recording
python Data_Preparation/FINAL_video_trimmer.py

# 3. Undistort lens videos
python Data_Preparation/FINAL_undistort_omni_videos.py

# 4. Stitch into 360 panorama
python Data_Preparation/FINAL_clean_stitcher.py

# 5. Detect vehicles on panorama
python Vehicle_detection_360/FINAL_panoramic_detection.py

# 6. Detect pedestrians on panorama
python Human_detection_360/FINAL_panoramic_detection.py
\\\

---

*Last updated: July 2026*
