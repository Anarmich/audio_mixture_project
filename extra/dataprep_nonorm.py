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

    for mixture_id in range(len(manifest['mixtures'])):
        try:
            mixture_file = manifest['mixtures'][mixture_id]
            road_file = manifest['sources']['road'][mixture_id]
            wind_file = manifest['sources']['wind'][mixture_id]
            powertrain_file = manifest['sources']['powertrain'][mixture_id]

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
            
            np.save(os.path.join(output_dir, f"mixture_{mixture_id}.npy"), mixture_spectrogram)
            np.save(os.path.join(output_dir, f"road_{mixture_id}.npy"), road_spectrogram)
            np.save(os.path.join(output_dir, f"wind_{mixture_id}.npy"), wind_spectrogram)
            np.save(os.path.join(output_dir, f"powertrain_{mixture_id}.npy"), powertrain_spectrogram)
            
            print(f"Processed mixture {mixture_id}")
        
        except Exception as e:
            print(f"Error processing mixture {mixture_id}: {e}")

if __name__ == "__main__":
    manifest_file = '/path/to/manifest.json'
    data_dir = '/path/to/audio/files'
    output_dir = '/path/to/data_features'
    
    manifest = load_manifest(manifest_file)
    prepare_datasets(manifest, data_dir, output_dir)
