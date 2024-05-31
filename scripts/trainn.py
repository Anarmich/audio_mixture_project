import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

class AudioDataset(Dataset):
    def __init__(self, manifest_file):
        with open(manifest_file, 'r') as f:
            self.manifest = json.load(f)
        self.mixtures = self.manifest['mixtures']
        self.road = self.manifest['sources']['road']
        self.wind = self.manifest['sources']['wind']
        self.powertrain = self.manifest['sources']['powertrain']

    def __len__(self):
        return len(self.mixtures)

    def __getitem__(self, idx):
        mixture_path = self.mixtures[idx]
        road_path = self.road[idx]
        wind_path = self.wind[idx]
        powertrain_path = self.powertrain[idx]

        mixture = np.load(mixture_path)
        road = np.load(road_path)
        wind = np.load(wind_path)
        powertrain = np.load(powertrain_path)

        mixture = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        road = torch.tensor(road, dtype=torch.float32)
        wind = torch.tensor(wind, dtype=torch.float32)
        powertrain = torch.tensor(powertrain, dtype=torch.float32)
        sources = torch.stack([road, wind, powertrain], dim=0)

        return mixture, sources

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        # Bottleneck
        bottleneck = self.bottleneck(enc3)

        # Decoder
        dec3 = self.upconv3(bottleneck)
        if dec3.size() != enc3.size():
            diffY = enc3.size()[2] - dec3.size()[2]
            diffX = enc3.size()[3] - dec3.size()[3]
            dec3 = F.pad(dec3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        if dec2.size() != enc2.size():
            diffY = enc2.size()[2] - dec2.size()[2]
            diffX = enc2.size()[3] - dec2.size()[3]
            dec2 = F.pad(dec2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        if dec1.size() != enc1.size():
            diffY = enc1.size()[2] - dec1.size()[2]
            diffX = enc1.size()[3] - dec1.size()[3]
            dec1 = F.pad(dec1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        out = self.out_conv(dec1)
        return out

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cuda'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Training Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                if batch_idx % 10 == 0:
                    print(f"Validation Batch {batch_idx}/{len(val_loader)}, Loss: {loss.item():.4f}")

        val_loss /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), 'unmix_model.pth')

if __name__ == "__main__":
    manifest_file = '/Users/argy/audio_mixture_project/data_features/data_manifest.json'

    dataset = AudioDataset(manifest_file)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Reduced batch size
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = UNet(in_channels=1, out_channels=3)
    train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu')
