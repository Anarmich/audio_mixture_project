# scripts/create_mixtures.py

import os
import numpy as np
import soundfile as sf
from itertools import product
import random
import json

def load_audio(file_path):
    audio, samplerate = sf.read(file_path)
    return audio, samplerate

def save_audio(file_path, audio, samplerate):
    sf.write(file_path, audio, samplerate)

def create_mixture(road_audio, wind_audio, powertrain_audio):
    min_length = min(len(road_audio), len(wind_audio), len(powertrain_audio))
    road_audio = road_audio[:min_length]
    wind_audio = wind_audio[:min_length]
    powertrain_audio = powertrain_audio[:min_length]

    mixture = road_audio + wind_audio + powertrain_audio

    max_val = np.max(np.abs(mixture))
    if max_val > 1.0:
        mixture = mixture / max_val

    return mixture

def main(data_dir, output_json, num_augmentations=5):
    source_dirs = {
        'road': os.path.join(data_dir, 'road'),
        'wind': os.path.join(data_dir, 'wind'),
        'powertrain': os.path.join(data_dir, 'powertrain')
    }
    output_dir = os.path.join(data_dir, 'mixture')

    os.makedirs(output_dir, exist_ok=True)

    dataset = {'mixtures': [], 'sources': {'road': [], 'wind': [], 'powertrain': []}}

    road_files = [os.path.join(source_dirs['road'], f) for f in os.listdir(source_dirs['road']) if f.endswith('.wav')]
    wind_files = [os.path.join(source_dirs['wind'], f) for f in os.listdir(source_dirs['wind']) if f.endswith('.wav')]
    powertrain_files = [os.path.join(source_dirs['powertrain'], f) for f in os.listdir(source_dirs['powertrain']) if f.endswith('.wav')]

    combinations = list(product(road_files, wind_files, powertrain_files))
    random.shuffle(combinations)

    for i, (road_file, wind_file, powertrain_file) in enumerate(combinations):
        road_audio, sr = load_audio(road_file)
        wind_audio, _ = load_audio(wind_file)
        powertrain_audio, _ = load_audio(powertrain_file)

        mixture = create_mixture(road_audio, wind_audio, powertrain_audio)
        output_file = os.path.join(output_dir, f'mixture_{i}.wav')
        save_audio(output_file, mixture, sr)

        base_filename = f'mixture_{i}'
        dataset['mixtures'].append(output_file)
        dataset['sources']['road'].append(road_file)
        dataset['sources']['wind'].append(wind_file)
        dataset['sources']['powertrain'].append(powertrain_file)

    with open(output_json, 'w') as f:
        json.dump(dataset, f)

if __name__ == "__main__":
    data_dirs = ['data/train', 'data/valid', 'data/test']
    for data_dir in data_dirs:
        output_json = os.path.join(data_dir, 'data_manifest.json')
        main(data_dir, output_json)
