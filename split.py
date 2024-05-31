import os
import shutil
import random

def split_data(source_dir, train_dir, valid_dir, test_dir, train_ratio=0.7, valid_ratio=0.15):
    # Create directories if they do not exist
    for dir in [train_dir, valid_dir, test_dir]:
        os.makedirs(os.path.join(dir, 'road'), exist_ok=True)
        os.makedirs(os.path.join(dir, 'wind'), exist_ok=True)
        os.makedirs(os.path.join(dir, 'powertrain'), exist_ok=True)

    # List all files in the source directory
    road_files = [f for f in os.listdir(os.path.join(source_dir, 'road')) if f.endswith('.wav')]
    wind_files = [f for f in os.listdir(os.path.join(source_dir, 'wind')) if f.endswith('.wav')]
    powertrain_files = [f for f in os.listdir(os.path.join(source_dir, 'powertrain')) if f.endswith('.wav')]

    # Shuffle the files
    random.shuffle(road_files)
    random.shuffle(wind_files)
    random.shuffle(powertrain_files)

    # Split files into train, valid, and test
    def split_and_move(files, category):
        total_files = len(files)
        train_end = int(total_files * train_ratio)
        valid_end = train_end + int(total_files * valid_ratio)

        for i, file in enumerate(files):
            if i < train_end:
                dest_dir = train_dir
            elif i < valid_end:
                dest_dir = valid_dir
            else:
                dest_dir = test_dir
            shutil.move(os.path.join(source_dir, category, file), os.path.join(dest_dir, category, file))

    split_and_move(road_files, 'road')
    split_and_move(wind_files, 'wind')
    split_and_move(powertrain_files, 'powertrain')

    print("Data split completed successfully.")

if __name__ == "__main__":
    source_dir = 'data/train'
    train_dir = 'data/train_split'
    valid_dir = 'data/valid'
    test_dir = 'data/test'

    split_data(source_dir, train_dir, valid_dir, test_dir)
