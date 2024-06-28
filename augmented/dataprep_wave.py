import librosa
import numpy as np
import pywt
import json
import os

def extract_spectrogram(audio_path, n_fft=2048, hop_length=512):
    """Extract the spectrogram from an audio file."""
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db, sr

def extract_wavelet(audio_path, wavelet='db1'):
    """Extract the wavelet transform from an audio file."""
    y, sr = librosa.load(audio_path, sr=None)
    coeffs = pywt.wavedec(y, wavelet, level=6)
    wavelet_coeffs = np.hstack(coeffs)
    return wavelet_coeffs, sr

def normalize(data):
    """Normalize the data to the range [0, 1]."""
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

def load_manifest(file_path):
    """Load the JSON manifest file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_datasets(manifest, data_dir, output_dir):
    """Prepare the dataset by extracting spectrograms and wavelet transforms, saving them as NumPy arrays."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_processed = 0
    new_manifest = {
        "mixtures": [],
        "sources": {
            "road": [],
            "wind": [],
            "powertrain": []
        },
        "mixtures_wavelet": []
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
            
            # Extract spectrograms
            mixture_spectrogram, _ = extract_spectrogram(mixture_path)
            road_spectrogram, _ = extract_spectrogram(road_path)
            wind_spectrogram, _ = extract_spectrogram(wind_path)
            powertrain_spectrogram, _ = extract_spectrogram(powertrain_path)

            # Extract wavelet transforms
            mixture_wavelet, _ = extract_wavelet(mixture_path)

            # Normalize the spectrograms and wavelets
            mixture_spectrogram = normalize(mixture_spectrogram)
            road_spectrogram = normalize(road_spectrogram)
            wind_spectrogram = normalize(wind_spectrogram)
            powertrain_spectrogram = normalize(powertrain_spectrogram)
            
            mixture_wavelet = normalize(mixture_wavelet)

            # Save the spectrograms and wavelet transforms as .npy files
            mixture_output_path = os.path.join(output_dir, f'mixture_{mixture_id}.npy')
            road_output_path = os.path.join(output_dir, f'road_{mixture_id}.npy')
            wind_output_path = os.path.join(output_dir, f'wind_{mixture_id}.npy')
            powertrain_output_path = os.path.join(output_dir, f'powertrain_{mixture_id}.npy')
            
            mixture_wavelet_output_path = os.path.join(output_dir, f'mixture_wavelet_{mixture_id}.npy')

            np.save(mixture_output_path, mixture_spectrogram)
            np.save(road_output_path, road_spectrogram)
            np.save(wind_output_path, wind_spectrogram)
            np.save(powertrain_output_path, powertrain_spectrogram)
            
            np.save(mixture_wavelet_output_path, mixture_wavelet)

            # Update new manifest
            new_manifest['mixtures'].append(mixture_output_path)
            new_manifest['sources']['road'].append(road_output_path)
            new_manifest['sources']['wind'].append(wind_output_path)
            new_manifest['sources']['powertrain'].append(powertrain_output_path)
            
            new_manifest['mixtures_wavelet'].append(mixture_wavelet_output_path)

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
    data_dir = '/Users/argy/audio_mixture_project/data/train'  # Change this to 'data/valid' or 'data/test' as needed
    manifest_file = os.path.join(data_dir, 'data_manifest.json')
    output_dir = '/Users/argy/audio_mixture_project/data_features'  # Directory to save the processed features

    # Load the manifest
    manifest = load_manifest(manifest_file)

    # Prepare the dataset
    prepare_datasets(manifest, data_dir, output_dir)
