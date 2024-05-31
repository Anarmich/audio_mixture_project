import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

def inspect_npy_file(file_path, sr=44100, n_fft=2048, hop_length=512, original_max_value=None):
    """Load and inspect an .npy file, optionally recovering the original spectrogram."""
    data = np.load(file_path)
    print(f"Shape of the loaded data: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Max value: {np.max(data)}")
    print(f"Min value: {np.min(data)}")

    # Recover the original spectrogram if original_max_value is provided
    if original_max_value:
        data = data * original_max_value
        print(f"Recovered Max value: {np.max(data)}")
        print(f"Recovered Min value: {np.min(data)}")

    # Display the spectrogram with correct axes
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(data, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

if __name__ == "__main__":
    data_dir = '/Users/argy/audio_mixture_project/data_features'
    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    if npy_files:
        print("Available .npy files:")
        for i, file in enumerate(npy_files):
            print(f"{i}: {file}")

        # Ask the user to select a file
        file_index = int(input(f"Enter the index of the file to inspect (0-{len(npy_files) - 1}): "))
        if 0 <= file_index < len(npy_files):
            file_path = os.path.join(data_dir, npy_files[file_index])
            print(f"Inspecting file: {file_path}")
            
            # Provide the original max value used for normalization if known
            original_max_value = 1.0  # Replace with the actual max value used during normalization
            
            # Sample rate, FFT window size, and hop length
            sr = 44100  # Sample rate
            n_fft = 2048  # FFT window size
            hop_length = 512  # Hop length
            
            inspect_npy_file(file_path, sr=sr, n_fft=n_fft, hop_length=hop_length, original_max_value=original_max_value)
        else:
            print("Invalid index selected.")
    else:
        print("No .npy files found in the specified directory.")
