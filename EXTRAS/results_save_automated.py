import shutil
from pathlib import Path
import os   

def gather_and_rename_undistorted():
    # Set the root directory to scan
    root_dir = Path(r"E:\ViswakMUMMASRESULTS\RUMMAIYA")
    
    # Define the destination directory
    final_undistorted_dir = Path(r"E:\ViswakMUMMASRESULTS\RUMMAIYA\Final_Undistorted_Images")
    
    if not final_undistorted_dir.exists():
        final_undistorted_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning through: {root_dir}")
    print(f"Saving to: {final_undistorted_dir}")
    print("-" * 50)
    
    copied_count = 0
    
    # Scan for all folders named "undistorted"
    for undistorted_folder in root_dir.rglob("undistorted"):
        
        if final_undistorted_dir in undistorted_folder.parents:
            continue
            
        parent_folder_name = undistorted_folder.parent.name
        unique_id = parent_folder_name.replace("Pipeline_", "")
        
        new_folder_name = f"undistorted_{unique_id}"
        destination_path = final_undistorted_dir / new_folder_name
        
        # 1. Copy the entire directory
        if not destination_path.exists():
            shutil.copytree(undistorted_folder, destination_path)
            
            # 2. Rename files within the copied directory
            # We look for files containing 'LENS' and rename them to 1.jpg, 2.jpg, etc.
            files = sorted(list(destination_path.glob("*LENS*")))
            for i, file_path in enumerate(files, start=1):
                new_filename = f"{i}{file_path.suffix}"
                file_path.rename(destination_path / new_filename)
                
            copied_count += 1
            print(f"Copied and renamed files in: {new_folder_name}")
        else:
            print(f"Skipping: {new_folder_name} (already exists)")

    print("-" * 50)
    print(f"Task complete! Successfully processed {copied_count} undistorted folders.")

if __name__ == "__main__":
    gather_and_rename_undistorted()

#for copying the final stitched images to a separate folder with new names based on their parent pipeline folders, you can use the following code:

def gather_stitched_images():
    #  Define your base directory and the new results directory 
    # define the base directory in E/viswakMUMMASRESULTS and the results directory as Geetika within it. You can change these paths as needed.

    #directly rad from this folder E:\ViswakMUMMASRESULTS\Geetika
    directory_to_scan = Path(r"E:\ViswakMUMMASRESULTS\RUMMAIYA")

    # all the folders within the folder named like Pipeline_2026-01-19_10-52-58_F1028
    #   

    # results_dir = base_dir / "Geetika"
    

    
    
    target_filename = "Final_Static_Stitch.jpg"
    final_folder_name = Path(r"E:\ViswakMUMMASRESULTS\RUMMAIYA\Final_Stitched_Images")
    if not final_folder_name.exists():
        final_folder_name.mkdir(parents=True, exist_ok=True)

    
    # results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning through: {directory_to_scan}")
    print("-" * 50)

    
    copied_count = 0
    for file_path in directory_to_scan.rglob(target_filename):
        
        # if directory_to_scan in file_path.parents:
        #     continue
        if final_folder_name in file_path.parts:
            continue

        parent_folder_name = file_path.parent.name

        
        if parent_folder_name.startswith("Pipeline_"):
            unique_id = parent_folder_name.replace("Pipeline_", "")
        else:
            unique_id = parent_folder_name

        
        new_name = f"Final_Static_Stitch_{unique_id}{file_path.suffix}"
        
       
        destination_path = directory_to_scan / new_name

        
        # shutil.copy2(file_path, destination_path)
        #save each file with the new name within the final folder
        
        shutil.copy2(file_path, Path(final_folder_name) / new_name)
        copied_count += 1
        
        print(f"Copied and renamed to: {new_name}")

    print("-" * 50)
    print(f"Task complete! Successfully copied {copied_count} files to {final_folder_name}")

if __name__ == "__main__":
    gather_and_rename_undistorted()
    gather_stitched_images()