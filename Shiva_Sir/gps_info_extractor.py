import os
import pandas as pd

def generate_gpx_from_imu_csv(input_csv, output_gpx, max_points=10000):
    """
    Extracts GPS data from a CSV with comments and creates a .gpx file.
    Triggers a new entry only when 'gnss_latitude' changes.
    """
    try:
        # 'comment=#' tells pandas to ignore lines starting with the # symbol
        # 'sep=None' auto-detects if your file uses commas or semicolons
        df = pd.read_csv(input_csv, comment='#', sep=None, engine='python')
        
        # Clean column names just in case there are hidden spaces
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Error reading csv file: {e}")
        return

    # Column names verified from your provided header list
    lat_col = 'gnss_latitude'
    lon_col = 'gnss_longitude'
    time_col = 'timestamp_utc'
    alt_col = 'gnss_altitude'

    # Safety check for required columns
    required = [lat_col, lon_col, time_col, alt_col]
    if not all(col in df.columns for col in required):
        missing = [col for col in required if col not in df.columns]
        print(f"Error: Could not find columns: {missing}")
        print(f"Detected columns: {list(df.columns[:5])}...")
        return
    #  Filter for:
    #    a) The very first row (where change_count is 1)
    #    b) Every 50th change thereafter
    lat_changed = df[lat_col] != df[lat_col].shift()


    change_count = lat_changed.cumsum()

    df_filtered = df[(change_count % 5 == 0) & lat_changed].copy()

    # Limit to the first 100 generated values 
    df_result = df_filtered.head(max_points)

    # Build the GPX structure matching your gpx.fmt template 
    filename = os.path.basename(input_csv)
    gpx_content = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="ExifTool" xmlns="http://www.topografix.com/GPX/1/1">',
        '  <trk>',
        f'    <name>{filename}</name>',
        '    <trkseg>'
    ]

    for _, row in df_result.iterrows():
        # The BODY format from your gpx.fmt 
        lat = row[lat_col]
        lon = row[lon_col]
        ele = row[alt_col]
        time = row[time_col]

        gpx_content.append(f'      <trkpt lat="{lat}" lon="{lon}">')
        gpx_content.append(f'        <ele>{ele}</ele>')
        gpx_content.append(f'        <time>{time}</time>')
        gpx_content.append('      </trkpt>')

    # Closing tags based on the TAIL format 
    gpx_content.extend([
        '    </trkseg>',
        '  </trk>',
        '</gpx>'
    ])

    # Write the file to disk
    with open(output_gpx, 'w', encoding='utf-8') as f:
        f.write('\n'.join(gpx_content))

    print(f"Success! Generated '{output_gpx}' with {len(df_result)} points.")

# --- Execution ---
# Using the exact path and filename from your system
input_file = r'C:\IITM\IMU-GPS\imu_all_topics_2026-01-17_15-21-18.csv'
output_file = r'C:\IITM\Shiva_Sir\extracted_gps_data.gpx'

generate_gpx_from_imu_csv(input_file, output_file)