# import librosa
# import numpy as np
# import json
# import os

# def extract_spectrogram(audio_path, n_fft=2048, hop_length=512):
#     """Extract the spectrogram from an audio file."""
#     y, sr = librosa.load(audio_path, sr=None)
#     S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
#     S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
#     return S_db, sr

# def load_manifest(file_path):
#     """Load the JSON manifest file."""
#     with open(file_path, 'r') as f:
#         return json.load(f)

# def prepare_datasets(manifest, data_dir, output_dir):
#     """Prepare the dataset by extracting spectrograms and saving them as NumPy arrays."""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     num_processed = 0

#     for mixture_id in range(len(manifest['mixtures'])):
#         try:
#             mixture_file = manifest['mixtures'][mixture_id]
#             road_file = manifest['sources']['road'][mixture_id]
#             wind_file = manifest['sources']['wind'][mixture_id]
#             powertrain_file = manifest['sources']['powertrain'][mixture_id]

#             # Construct the full paths correctly
#             mixture_path = os.path.join(data_dir, mixture_file)
#             road_path = os.path.join(data_dir, road_file)
#             wind_path = os.path.join(data_dir, wind_file)
#             powertrain_path = os.path.join(data_dir, powertrain_file)
            
#             if not os.path.exists(mixture_path):
#                 print(f"File not found: {mixture_path}")
#                 continue
#             if not os.path.exists(road_path):
#                 print(f"File not found: {road_path}")
#                 continue
#             if not os.path.exists(wind_path):
#                 print(f"File not found: {wind_path}")
#                 continue
#             if not os.path.exists(powertrain_path):
#                 print(f"File not found: {powertrain_path}")
#                 continue
            
#             mixture_spectrogram, _ = extract_spectrogram(mixture_path)
#             road_spectrogram, _ = extract_spectrogram(road_path)
#             wind_spectrogram, _ = extract_spectrogram(wind_path)
#             powertrain_spectrogram, _ = extract_spectrogram(powertrain_path)


#             # Avoid divide-by-zero errors by checking if max value is not zero
#             def safe_normalize(spectrogram):
#                 max_val = np.min(spectrogram)
#                 # if max_val != 0:
#                 return spectrogram / max_val
#                 # return spectrogram

#             mixture_spectrogram = safe_normalize(mixture_spectrogram)
#             road_spectrogram = safe_normalize(road_spectrogram)
#             wind_spectrogram = safe_normalize(wind_spectrogram)
#             powertrain_spectrogram = safe_normalize(powertrain_spectrogram)


#             # Save the spectrograms as .npy files
#             mixture_output_path = os.path.join(output_dir, f'mixture_{mixture_id}.npy')
#             road_output_path = os.path.join(output_dir, f'road_{mixture_id}.npy')
#             wind_output_path = os.path.join(output_dir, f'wind_{mixture_id}.npy')
#             powertrain_output_path = os.path.join(output_dir, f'powertrain_{mixture_id}.npy')

#             np.save(mixture_output_path, mixture_spectrogram)
#             np.save(road_output_path, road_spectrogram)
#             np.save(wind_output_path, wind_spectrogram)
#             np.save(powertrain_output_path, powertrain_spectrogram)

#             num_processed += 1
        
#         except Exception as e:
#             print(f"Error processing mixture ID {mixture_id}: {e}")
#             continue

#     print(f"Successfully processed {num_processed} mixtures.")

# if __name__ == "__main__":
#     data_dir = ''  # Change this to 'data/valid' or 'data/test' as needed
#     manifest_file = os.path.join(data_dir, '/Users/argy/audio_mixture_project/data/train/data_manifest.json')
#     output_dir = '/Users/argy/audio_mixture_project/data_features'  # Directory to save the processed features

#     # Load the manifest
#     manifest = load_manifest(manifest_file)

#     # Prepare the dataset
#     prepare_datasets(manifest, data_dir, output_dir)

import librosa
import numpy as np
import json
import os

def extract_spectrogram(audio_path, n_fft=2048, hop_length=512):
    """Extract the spectrogram from an audio file."""
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db, sr

def load_manifest(file_path):
    """Load the JSON manifest file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_datasets(manifest, data_dir, output_dir):
    """Prepare the dataset by extracting spectrograms and saving them as NumPy arrays."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_processed = 0
    new_manifest = {
        "mixtures": [],
        "sources": {
            "road": [],
            "wind": [],
            "powertrain": []
        }
    }

    for mixture_id in range(len(manifest['mixtures'])):
        try:
            mixture_file = manifest['mixtures'][mixture_id]
            road_file = manifest['sources']['road'][mixture_id]
            wind_file = manifest['sources']['wind'][mixture_id]
            powertrain_file = manifest['sources']['powertrain'][mixture_id]

            # Construct the full paths correctly
            mixture_path = os.path.join(data_dir, mixture_file)
            road_path = os.path.join(data_dir, road_file)
            wind_path = os.path.join(data_dir, wind_file)
            powertrain_path = os.path.join(data_dir, powertrain_file)
            
            if not os.path.exists(mixture_path):
                print(f"File not found: {mixture_path}")
                continue
            if not os.path.exists(road_path):
                print(f"File not found: {road_path}")
                continue
            if not os.path.exists(wind_path):
                print(f"File not found: {wind_path}")
                continue
            if not os.path.exists(powertrain_path):
                print(f"File not found: {powertrain_path}")
                continue
            
            mixture_spectrogram, _ = extract_spectrogram(mixture_path)
            road_spectrogram, _ = extract_spectrogram(road_path)
            wind_spectrogram, _ = extract_spectrogram(wind_path)
            powertrain_spectrogram, _ = extract_spectrogram(powertrain_path)

            # Avoid divide-by-zero errors by checking if max value is not zero
            def safe_normalize(spectrogram):
                max_val = np.min(spectrogram)
                max_val = 1
                # if max_val != 0:
                #     return spectrogram / max_val
                return spectrogram / max_val

            mixture_spectrogram = safe_normalize(mixture_spectrogram)
            road_spectrogram = safe_normalize(road_spectrogram)
            wind_spectrogram = safe_normalize(wind_spectrogram)
            powertrain_spectrogram = safe_normalize(powertrain_spectrogram)

            # Save the spectrograms as .npy files
            mixture_output_path = os.path.join(output_dir, f'mixture_{mixture_id}.npy')
            road_output_path = os.path.join(output_dir, f'road_{mixture_id}.npy')
            wind_output_path = os.path.join(output_dir, f'wind_{mixture_id}.npy')
            powertrain_output_path = os.path.join(output_dir, f'powertrain_{mixture_id}.npy')

            np.save(mixture_output_path, mixture_spectrogram)
            np.save(road_output_path, road_spectrogram)
            np.save(wind_output_path, wind_spectrogram)
            np.save(powertrain_output_path, powertrain_spectrogram)

            # Update new manifest
            new_manifest['mixtures'].append(mixture_output_path)
            new_manifest['sources']['road'].append(road_output_path)
            new_manifest['sources']['wind'].append(wind_output_path)
            new_manifest['sources']['powertrain'].append(powertrain_output_path)

            num_processed += 1
        
        except Exception as e:
            print(f"Error processing mixture ID {mixture_id}: {e}")
            continue

    print(f"Successfully processed {num_processed} mixtures.")

    # Save the new manifest file
    new_manifest_file = os.path.join(output_dir, 'data_manifest.json')
    with open(new_manifest_file, 'w') as f:
        json.dump(new_manifest, f, indent=4)

if __name__ == "__main__":
    data_dir = ''  # Change this to 'data/valid' or 'data/test' as needed
    # manifest_file = os.path.join(data_dir, '/home/ub14693/audio_mixture_project/data/train/data_manifest.json')
    manifest_file = os.path.join(data_dir, '/home/ub14693/audio_mixture_project/data/train/data_manifest.json')

    # output_dir = '/home/ub14693/audio_mixture_project/data_features'  # Directory to save the processed features
    output_dir = '/home/ub14693/audio_mixture_project/data_features'  # Directory to save the processed features

    # Load the manifest
    manifest = load_manifest(manifest_file)

    # Prepare the dataset
    prepare_datasets(manifest, data_dir, output_dir)
