import shutil
from pathlib import Path

def gather_stitched_images():
    #  Define your base directory and the new results directory
    base_dir = Path(r"C:\Results\geetika")
    results_dir = base_dir / "results"
    
    
    target_filename = "Final_Static_Stitch.jpg"

    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning through: {base_dir}")
    print("-" * 50)

    
    copied_count = 0
    for file_path in base_dir.rglob(target_filename):
        
        if results_dir in file_path.parents:
            continue

        parent_folder_name = file_path.parent.name

        
        if parent_folder_name.startswith("Pipeline_"):
            unique_id = parent_folder_name.replace("Pipeline_", "")
        else:
            unique_id = parent_folder_name

        
        new_name = f"Final_Static_Stitch_{unique_id}{file_path.suffix}"
        
       
        destination_path = results_dir / new_name

        
        shutil.copy2(file_path, destination_path)
        copied_count += 1
        
        print(f"Copied and renamed to: {new_name}")

    print("-" * 50)
    print(f"Task complete! Successfully copied {copied_count} files to {results_dir}")

if __name__ == "__main__":
    gather_stitched_images()