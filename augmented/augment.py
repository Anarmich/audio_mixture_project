import os
import numpy as np
import librosa
import soundfile as sf
import random
from tqdm import tqdm

def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def shift_time(data, shift_max=0.2, shift_direction='both'):
    shift = np.random.randint(int(len(data) * shift_max))
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = random.choice([-1, 1])
        shift = shift * direction
    augmented_data = np.roll(data, shift)
    return augmented_data

def change_pitch(data, sampling_rate, pitch_factor=2):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def change_volume(data, db_factor=5.0):
    """ Randomly amplify or attenuate the volume """
    factor = np.random.uniform(-db_factor, db_factor)
    augmented_data = librosa.db_to_amplitude(factor) * data
    return augmented_data

def augment_data(source_dir, output_dir, augment_count=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for source_name in ['road', 'wind', 'powertrain']:
        source_path = os.path.join(source_dir, source_name)
        if not os.path.isdir(source_path):
            continue

        files = os.listdir(source_path)
        
        for file_name in tqdm(files, desc=f"Augmenting {source_name}"):
            file_path = os.path.join(source_path, file_name)
            if not file_path.endswith(('.wav', '.mp3', '.flac')):  # Ensure only audio files are processed
                continue

            data, sr = librosa.load(file_path, sr=None)

            for j in range(augment_count):
                augmented_data = data
                if random.random() < 0.5:
                    augmented_data = add_noise(augmented_data)
                if random.random() < 0.5:
                    augmented_data = shift_time(augmented_data)
                if random.random() < 0.5:
                    augmented_data = change_pitch(augmented_data, sr)
                if random.random() < 0.5:
                    augmented_data = change_volume(augmented_data)

                base_name = os.path.splitext(file_name)[0]
                augmented_file_path = os.path.join(output_dir, f"{base_name}_{source_name}_aug_{j}.wav")
                sf.write(augmented_file_path, augmented_data, sr)

if __name__ == "__main__":
    source_dir = '/Users/argy/audio_mixture_project/data/train'
    output_dir = '/Users/argy/audio_mixture_project/augmented_sources'
    augment_data(source_dir, output_dir, augment_count=5)
