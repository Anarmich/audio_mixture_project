import os
import shutil

def move_files(src_dir, dest_dir_map):
    if not os.path.exists(src_dir):
        print(f"Source directory {src_dir} does not exist.")
        return
    
    for file in os.listdir(src_dir):
        if file.endswith(".wav"):
            # Determine the target subfolder based on the filename
            if 'road' in file:
                target_subfolder = 'road'
            elif 'wind' in file:
                target_subfolder = 'wind'
            elif 'powertrain' in file:
                target_subfolder = 'powertrain'
            else:
                print(f"Skipping file {file} as it does not match any target subfolder.")
                continue
            
            # Construct paths
            src_path = os.path.join(src_dir, file)
            dest_dir = dest_dir_map[target_subfolder]
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            dest_path = os.path.join(dest_dir, file)
            
            # Move file
            shutil.move(src_path, dest_path)
            print(f"Moved {src_path} to {dest_path}")

# Base directory for your augmented clips
base_dir = "/Users/argy/audio_mixture_project/augmented_sources"

# Define target directories
target_dirs = {
    "road": "/Users/argy/audio_mixture_project/data/train/road",
    "wind": "/Users/argy/audio_mixture_project/data/train/wind",
    "powertrain": "/Users/argy/audio_mixture_project/data/train/powertrain"
}

# Move files to the appropriate directories
move_files(base_dir, target_dirs)

print("Augmented files moved and renamed successfully!")
