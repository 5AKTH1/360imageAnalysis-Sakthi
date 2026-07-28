import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

# --- CONFIGURATION ---
BASE_DIR = r"C:\IITM\Vehicle_detection_360"
JSON_PATH = os.path.join(BASE_DIR, "ablation_study_gpu.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. LOAD DATA ---
if not os.path.exists(JSON_PATH):
    print(f"Error: Could not find {JSON_PATH}")
    # Fallback for demonstration if file doesn't exist in immediate run environment
    exit()

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# --- 2. PROCESS DATA FOR HEATMAP ---
rows = []
for config, stats in data.items():
    try:
        res, fps_str = config.split('_')
        fps = int(float(fps_str.replace('fps', '')))
        counts = stats.get('counts', {})
        total_vehicles = sum(counts.values())
        
        rows.append({
            "Resolution": res,
            "FPS": fps,
            "Total_Vehicles": total_vehicles
        })
    except:
        continue

df = pd.DataFrame(rows)

# Define logical sorting for the axis
res_order = ["4k", "1080p", "720p", "480p"]
df['Resolution'] = pd.Categorical(df['Resolution'], categories=res_order, ordered=True)

# Create Pivot Table: Rows=Resolution, Cols=FPS, Values=Count
heatmap_data = df.pivot(index="Resolution", columns="FPS", values="Total_Vehicles")

# Sort index/columns to ensure 4K is at top and 30fps is at right/left as preferred
heatmap_data = heatmap_data.sort_index()
heatmap_data = heatmap_data.sort_index(axis=1, ascending=False) # 30 -> 5

# --- 3. GENERATE HEATMAP ---
plt.figure(figsize=(10, 8))

# Use a color map that highlights 'high' (good) as green and 'low' (bad) as red
# 'RdYlGn' is Red-Yellow-Green
ax = sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".0f", 
    cmap="RdYlGn", 
    linewidths=1, 
    linecolor='white',
    cbar_kws={'label': 'Total Vehicles Detected'},
    vmin=heatmap_data.min().min(),
    vmax=heatmap_data.max().max()
)

plt.title("Total Vehicle Count Heatmap\n(Sensitivity Analysis)", fontsize=16, pad=20)
plt.xlabel("Input Frame Rate (FPS)", fontsize=12)
plt.ylabel("Input Resolution", fontsize=12)

# Move X-axis ticks to top for easier reading if desired (Matrix style)
# plt.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
# ax.set_xlabel('Input Frame Rate (FPS)', labelpad=15)
# ax.xaxis.set_label_position('top') 

save_path = os.path.join(OUTPUT_DIR, "plot5_total_count_heatmap.png")
plt.tight_layout()
plt.savefig(save_path)
print(f"Heatmap saved to: {save_path}")
plt.show()