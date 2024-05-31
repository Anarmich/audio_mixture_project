import os
import torchaudio
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, split, input_file, output_files, sample_rate=44100):
        self.root = os.path.join(root, split)
        self.input_file = input_file
        self.output_files = output_files.split(',')
        self.tracks = sorted(os.listdir(self.root))
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        input_path = os.path.join(self.root, track, self.input_file)
        input_audio, sr = torchaudio.load(input_path)

        target_audios = []
        for output_file in self.output_files:
            output_path = os.path.join(self.root, track, output_file)
            target_audio, _ = torchaudio.load(output_path)
            target_audios.append(target_audio)

        targets = torch.stack(target_audios)
        return {'mixture': input_audio, 'targets': targets}
