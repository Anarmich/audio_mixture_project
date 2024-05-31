import librosa
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def extract_spectrogram(audio_path, n_fft=2048, hop_length=512):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db, sr

def load_manifest(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_mixture_details(manifest, mixture_id):
    try:
        mixture_file = manifest['mixtures'][mixture_id]
        road_file = manifest['sources']['road'][mixture_id]
        wind_file = manifest['sources']['wind'][mixture_id]
        powertrain_file = manifest['sources']['powertrain'][mixture_id]

        print(f"Mixture: {mixture_file}")
        print(f"Road sound: {road_file if road_file else 'Not included'}")
        print(f"Wind sound: {wind_file if wind_file else 'Not included'}")
        print(f"Powertrain sound: {powertrain_file if powertrain_file else 'Not included'}")

        if road_file:
            road_path = os.path.join(data_dir, 'sources', 'road', road_file)
            road_spectrogram, sr = extract_spectrogram(road_file)
            print(road_spectrogram.shape)
            plt.figure(figsize=(10, 6))
            librosa.display.specshow(road_spectrogram, sr=sr, hop_length=512, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Road Spectrogram')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')

        if wind_file:
            wind_path = os.path.join(data_dir, 'sources', 'wind', wind_file)
            wind_spectrogram, sr = extract_spectrogram(wind_file)
            print(wind_spectrogram.shape)
            plt.figure(figsize=(10, 6))

            librosa.display.specshow(wind_spectrogram, sr=sr, hop_length=512, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Wind Spectrogram')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')

        if powertrain_file:
            powertrain_path = os.path.join(data_dir, 'sources', 'powertrain', powertrain_file)
            powertrain_spectrogram, sr = extract_spectrogram(powertrain_file)
            print(powertrain_spectrogram.shape)
            plt.figure(figsize=(10, 6))
            librosa.display.specshow(powertrain_spectrogram, sr=sr, hop_length=512, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Powertrain Spectrogram')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')

        plt.show()

    except IndexError:
        print("Invalid mixture ID. Please provide a valid ID.")

if __name__ == "__main__":
    data_dir = '/Users/argy/audio_mixture_project/data/train'  # Change this to 'data/valid' or 'data/test' as needed
    manifest_file = os.path.join(data_dir, 'data_manifest.json')

    # Load the manifest
    manifest = load_manifest(manifest_file)

    # Prompt user to enter mixture ID
    mixture_id = int(input("Enter the mixture ID to check: "))
    
    # Get mixture details
    get_mixture_details(manifest, mixture_id)

    # Extract and plot spectrogram
    mixture_path = os.path.join(data_dir, 'mixture', f'mixture_{mixture_id}.wav')
    spectrogram, sr = extract_spectrogram(mixture_path)
    print(spectrogram.shape)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(spectrogram, sr=sr, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()
