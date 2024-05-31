import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.startswith('mixture_')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mixture_file = self.file_list[idx]
        mixture_id = mixture_file.split('_')[1].split('.')[0]
        mixture_path = os.path.join(self.data_dir, mixture_file)
        road_path = os.path.join(self.data_dir, f'road_{mixture_id}.npy')
        wind_path = os.path.join(self.data_dir, f'wind_{mixture_id}.npy')
        powertrain_path = os.path.join(self.data_dir, f'powertrain_{mixture_id}.npy')

        mixture = np.load(mixture_path)
        road = np.load(road_path)
        wind = np.load(wind_path)
        powertrain = np.load(powertrain_path)

        # mixture = mixture / np.min(mixture)
        # road = road / np.min(road)
        # wind = wind / np.min(wind)
        # powertrain = powertrain / np.min(powertrain)

        mixture = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0)
        road = torch.tensor(road, dtype=torch.float32)
        wind = torch.tensor(wind, dtype=torch.float32)
        powertrain = torch.tensor(powertrain, dtype=torch.float32)

        sources = torch.stack([road, wind, powertrain], dim=0)

        return mixture, sources

if __name__ == "__main__":
    data_dir = '/Users/argy/audio_mixture_project/data_features'
    dataset = AudioDataset(data_dir)
    print(len(dataset))
    print(dataset[0])
