import json
import os

def convert_to_coco(json_path, output_coco):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Define Categories (Update this list if you have more labels)
    label_map = {"bus": 1, "motorbike": 2, "car": 3, "auto": 4, "truck": 5}
    categories = [{"id": v, "name": k, "supercategory": "vehicle"} for k, v in label_map.items()]

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    ann_id = 1
    processed_frames = set()

    for entry in data['detections']:
        frame_idx = entry['frame']
        
        # Add image info (one per frame)
        if frame_idx not in processed_frames:
            coco_data["images"].append({
                "id": frame_idx,
                "file_name": f"frame_{frame_idx:06d}.jpg", # Standard CVAT naming
                "width": 1920,  # Replace with actual width
                "height": 1080  # Replace with actual height
            })
            processed_frames.add(frame_idx)

        for obj in entry['objects']:
            # Your box: [xtl, ytl, xbr, ybr]
            # COCO box: [x_min, y_min, width, height]
            xtl, ytl, xbr, ybr = obj['bbox']
            width = xbr - xtl
            height = ybr - ytl

            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": frame_idx,
                "category_id": label_map.get(obj['label'], 0),
                "bbox": [xtl, ytl, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            ann_id += 1

    with open(output_coco, 'w') as f:
        json.dump(coco_data, f, indent=4)
    
    print(f"Successfully converted to {output_coco}")

convert_to_coco(r'C:\IITM\Vehicle_detection_360\FINAL_panoramic_detection2.json', r'C:\IITM\Vehicle_detection_360\FINAL_panoramic_detection2_coco.json')