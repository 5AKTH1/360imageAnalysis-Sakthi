import json
import zipfile
import os
import tempfile
import re
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def get_json_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        json_files = [f for f in z.namelist() if f.endswith('.json')]
        if not json_files:
            raise FileNotFoundError("No JSON file found inside the ZIP.")
        target_file = json_files[0]
        temp_dir = tempfile.mkdtemp()
        return z.extract(target_file, temp_dir)

def print_formatted_report(coco_eval, f1_score, id_to_cat_name):
    # Extract global stats
    map_50_95 = coco_eval.stats[0]
    map_50 = coco_eval.stats[1]
    map_75 = coco_eval.stats[2]
    mar = coco_eval.stats[8]

    report = []
    report.append("="*65)
    report.append("           VEHICLE DETECTION PERFORMANCE REPORT")
    report.append("="*65)
    report.append(f"{'METRIC':<30} | {'SCORE':<10}")
    report.append("-" * 65)
    report.append(f"{'mAP @ .50:.95 (Primary)':<30} | {map_50_95:<10.4f}")
    report.append(f"{'mAP @ .50 (Standard)':<30} | {map_50:<10.4f}")
    report.append(f"{'mAP @ .75 (Strict)':<30} | {map_75:<10.4f}")
    report.append(f"{'mAR (Average Recall)':<30} | {mar:<10.4f}")
    report.append(f"{'Overall F1-Score':<30} | {f1_score:<10.4f}")
    report.append("-" * 65)
    report.append("")
    report.append("CATEGORY-WISE BREAKDOWN")
    report.append("-" * 65)
    report.append(f"{'VEHICLE TYPE':<20} | {'mAP @.50':<15} | {'mAP @.75':<15}")
    report.append("-" * 65)

    prec_matrix = coco_eval.eval['precision']
    eval_cat_ids = list(coco_eval.params.catIds)

    for cat_id in sorted(eval_cat_ids):
        name = id_to_cat_name.get(cat_id, f"ID {cat_id}").upper()
        cat_idx = eval_cat_ids.index(cat_id)
        
        p_50 = prec_matrix[0, :, cat_idx, 0, 2] 
        p_75 = prec_matrix[5, :, cat_idx, 0, 2]
        
        ap_50 = np.mean(p_50[p_50 > -1]) if np.any(p_50 > -1) else 0.0
        ap_75 = np.mean(p_75[p_75 > -1]) if np.any(p_75 > -1) else 0.0
        
        report.append(f"{name:<20} | {ap_50:<15.4f} | {ap_75:<15.4f}")

    report.append("="*65)
    
    final_output = "\n".join(report)
    print(final_output)
    
    with open("detection_report.txt", "w") as f:
        f.write(final_output)

def evaluate_with_zip(ground_truth_path, prediction_path):
    extracted_gt_json = get_json_from_zip(ground_truth_path)
    coco_gt = COCO(extracted_gt_json)
    
    # Map labels to IDs
    cat_name_to_id = {cat['name'].lower().strip(): cat['id'] for cat in coco_gt.dataset['categories']}
    id_to_cat_name = {v: k for k, v in cat_name_to_id.items()}
    
    # Map frames (Limit to 3600 and apply Every 5th starting from 4)
    frame_to_image_id = {}
    valid_img_ids = []
    MAX_FRAMES = 3600
    
    for img in coco_gt.dataset['images']:
        match = re.search(r'\d+', img['file_name'])
        if match:
            frame_num = int(match.group())
            # CONDITION: Must be within 3600 AND follow every 5th frame starting at 4 (4, 9, 14...)
            if 0 <= frame_num < MAX_FRAMES and (frame_num % 10 == 4):
                frame_to_image_id[frame_num] = img['id']
                valid_img_ids.append(img['id'])
    
    print(f"[INFO] Evaluating on {len(valid_img_ids)} selected frames (Every 5th frame starting from 4).")

    with open(prediction_path, 'r') as f:
        pred_data = json.load(f)
        
    coco_predictions = []
    for frame_data in pred_data.get('detections', []):
        frame_num = frame_data['frame'] 
        # Only use predictions for the selected frames
        if frame_num not in frame_to_image_id:
            continue
            
        image_id = frame_to_image_id[frame_num]
        for obj in frame_data.get('objects', []):
            label = obj['label'].lower().strip()
            if label not in cat_name_to_id:
                continue 
                
            category_id = cat_name_to_id[label]
            coords = obj.get('box') or obj.get('bbox')
            xmin, ymin, xmax, ymax = coords
            
            coco_predictions.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "score": obj.get('current_conf', 1.0)
            })
            
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump(coco_predictions, tmp_file)
        tmp_pred_path = tmp_file.name

    coco_dt = coco_gt.loadRes(tmp_pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = valid_img_ids
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Calculate F1
    prec = coco_eval.stats[0]
    rec = coco_eval.stats[8]
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    # Output visually appealing report
    print_formatted_report(coco_eval, f1, id_to_cat_name)

if __name__ == "__main__":
    zip_input = r"C:\IITM\Results\Vehicle_detection_360\Ground_Truth_Labels_Panoramic.zip"
    predictions = r"C:\IITM\Vehicle_detection_360\FINAL_panoramic_detection2.json"
    evaluate_with_zip(zip_input, predictions)