import os
import torch
import json
from torch.utils.data import DataLoader, Dataset
import torchaudio
from openunmix import OpenUnmix

class CustomDataset(Dataset):
    def __init__(self, manifest):
        self.manifest = manifest

    def __len__(self):
        return len(self.manifest['mixtures'])

    def __getitem__(self, idx):
        mixture_path = self.manifest['mixtures'][idx]
        road_path = self.manifest['sources']['road'][idx]
        wind_path = self.manifest['sources']['wind'][idx]
        powertrain_path = self.manifest['sources']['powertrain'][idx]

        mixture, sr = torchaudio.load(mixture_path)
        sources = []
        for path in [road_path, wind_path, powertrain_path]:
            if path:
                source, _ = torchaudio.load(path)
                sources.append(source)
            else:
                sources.append(torch.zeros_like(mixture))
        return mixture, torch.stack(sources)

def load_manifest(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def train(model, train_loader, valid_loader, epochs, learning_rate, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for mixture, sources in train_loader:
            mixture, sources = mixture.to(device), sources.to(device)
            optimizer.zero_grad()
            estimates = model(mixture)
            loss = loss_fn(estimates, sources)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for mixture, sources in valid_loader:
                mixture, sources = mixture.to(device), sources.to(device)
                estimates = model(mixture)
                loss = loss_fn(estimates, sources)
                valid_loss += loss.item()

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Valid Loss: {valid_loss / len(valid_loader)}')

    return model

if __name__ == "__main__":
    train_manifest_path = 'data/train/data_manifest.json'
    valid_manifest_path = 'data/valid/data_manifest.json'

    train_manifest = load_manifest(train_manifest_path)
    valid_manifest = load_manifest(valid_manifest_path)

    train_dataset = CustomDataset(train_manifest)
    valid_dataset = CustomDataset(valid_manifest)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OpenUnmix(n_fft=4096, n_hop=1024, nb_channels=2).to(device)

    trained_model = train(model, train_loader, valid_loader, epochs=100, learning_rate=1e-3, device=device)

    torch.save(trained_model.state_dict(), 'openunmix_model.pth')
