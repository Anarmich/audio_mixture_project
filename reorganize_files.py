import os
import shutil

def move_files(src_dir, dest_dir, prefix):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for root, _, files in os.walk(src_dir):
        for idx, file in enumerate(files):
            if file.endswith(".wav"):
                # Construct a new file name
                new_name = f"{prefix}_{idx+1:02d}.wav"
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, new_name)
                shutil.move(src_path, dest_path)
                print(f"Moved {src_path} to {dest_path}")

# Base directory for your clips
base_dir = "Vehicle B clips"

# Define source and target directories
source_dirs = {
    "road_left": os.path.join(base_dir, "Road", "left"),
    "road_right": os.path.join(base_dir, "Road", "right"),
    "wind_left": os.path.join(base_dir, "Wind", "left"),
    "wind_right": os.path.join(base_dir, "Wind", "right"),
    "powertrain_left": os.path.join(base_dir, "PT", "left"),
    "powertrain_right": os.path.join(base_dir, "PT", "right"),
}

target_dirs = {
    "road": "data/train/road",
    "wind": "data/train/wind",
    "powertrain": "data/train/powertrain"
}

# Move files to the appropriate directories
move_files(source_dirs["road_left"], target_dirs["road"], "road_left")
move_files(source_dirs["road_right"], target_dirs["road"], "road_right")
move_files(source_dirs["wind_left"], target_dirs["wind"], "wind_left")
move_files(source_dirs["wind_right"], target_dirs["wind"], "wind_right")
move_files(source_dirs["powertrain_left"], target_dirs["powertrain"], "powertrain_left")
move_files(source_dirs["powertrain_right"], target_dirs["powertrain"], "powertrain_right")

print("Files moved and renamed successfully!")
