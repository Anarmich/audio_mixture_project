import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

def normalize_spectrogram(spectrogram):
    mean = spectrogram.mean()
    std = spectrogram.std()
    return spectrogram, mean, std

def denormalize_spectrogram(normalized_spectrogram, mean, std):
    return normalized_spectrogram * std + mean

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

        # mixture, mixture_mean, mixture_std = normalize_spectrogram(mixture)
        # road, road_mean, road_std = normalize_spectrogram(road)
        # wind, wind_mean, wind_std = normalize_spectrogram(wind)
        # powertrain, powertrain_mean, powertrain_std = normalize_spectrogram(powertrain)

        mixture = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        road = torch.tensor(road, dtype=torch.float32)
        wind = torch.tensor(wind, dtype=torch.float32)
        powertrain = torch.tensor(powertrain, dtype=torch.float32)
        sources = torch.stack([road, wind, powertrain], dim=0)

        # means = torch.tensor([road_mean, wind_mean, powertrain_mean], dtype=torch.float32)
        # stds = torch.tensor([road_std, wind_std, powertrain_std], dtype=torch.float32)

        return mixture, sources#, means, stds

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)  # Added layer
        self.enc6 = self.conv_block(1024, 2048)  # Added layer

        # Bottleneck
        self.bottleneck = self.conv_block(2048, 4096)  # Increased depth

        # Decoder
        self.upconv6 = nn.ConvTranspose2d(4096, 2048, kernel_size=2, stride=2)  # Added layer
        self.dec6 = self.conv_block(4096, 2048)  # Added layer
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)  # Added layer
        self.dec5 = self.conv_block(2048, 1024)  # Added layer
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)  # Dropout added
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2, stride=2))
        enc5 = self.enc5(F.max_pool2d(enc4, kernel_size=2, stride=2))  # Added layer
        enc6 = self.enc6(F.max_pool2d(enc5, kernel_size=2, stride=2))  # Added layer

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc6, kernel_size=2, stride=2))

        # Decoder with skip connections
        dec6 = self.upconv6(bottleneck)  # Added layer
        if dec6.size() != enc6.size():
            diffY = enc6.size()[2] - dec6.size()[2]
            diffX = enc6.size()[3] - dec6.size()[3]
            dec6 = F.pad(dec6, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec6 = torch.cat((dec6, enc6), dim=1)
        dec6 = self.dec6(dec6)

        dec5 = self.upconv5(dec6)  # Added layer
        if dec5.size() != enc5.size():
            diffY = enc5.size()[2] - dec5.size()[2]
            diffX = enc5.size()[3] - dec5.size()[3]
            dec5 = F.pad(dec5, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.dec5(dec5)

        dec4 = self.upconv4(dec5)
        if dec4.size() != enc4.size():
            diffY = enc4.size()[2] - dec4.size()[2]
            diffX = enc4.size()[3] - dec4.size()[3]
            dec4 = F.pad(dec4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
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

def weighted_mse_loss(predictions, targets, weights):
    weights = weights.view(1, -1, 1, 1)
    loss = weights * (predictions - targets) ** 2
    return loss.mean()

def train_model(model, train_loader, num_epochs=20, learning_rate=0.0001, device='cuda'):
    weights = torch.tensor([1.0, 1.0, 3.0]).to(device)  # Assign higher weight to powertrain component
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    writer = SummaryWriter(log_dir = 'runs/experiment_3')

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
                loss = weighted_mse_loss(outputs, targets, weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            writer.add_scalar('Batch Loss', loss.item(),epoch*len(train_loader)+batch_idx)
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Training Loss: {epoch_loss:.4f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        # for name,param in model.named_parameters():
        #     writer.add_histogram(name,param,epoch)
        #     if param.grad is not None:
        #         writer.add_historam(f'{name}_grad',param.grad, epoch)

    torch.save(model.state_dict(), 'unmix_model.pth')
    writer.close()
if __name__ == "__main__":
    manifest_file = '/home/ub14693/audio_mixture_project1/data_features/data_manifest.json'

    dataset = AudioDataset(manifest_file)

    train_size = int(1 * len(dataset))
    train_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model = UNet(in_channels=1, out_channels=3)
    train_model(model, train_loader, num_epochs=15, learning_rate=0.0001, device='cuda' if torch.cuda.is_available() else 'cpu')
