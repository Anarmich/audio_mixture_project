import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from torch.cuda.amp import GradScaler, autocast

def normalize_spectrogram(spectrogram):
    mean = spectrogram.mean()
    std = spectrogram.std()
    return (spectrogram - mean) / std, mean, std

def denormalize_spectrogram(normalized_spectrogram, mean, std):
    return normalized_spectrogram * std + mean

class AudioDataset(Dataset):
    def __init__(self, manifest_file):
        with open(manifest_file, 'r') as f:
            self.manifest = json.load(f)
        self.mixtures = self.manifest['mixtures']
        self.wind = self.manifest['sources']['wind']
        self.road = self.manifest['sources']['road']
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
        
        mixture, mixture_mean, mixture_std = normalize_spectrogram(mixture)
        road, road_mean, road_std = normalize_spectrogram(road)
        wind, wind_mean, wind_std = normalize_spectrogram(wind)
        powertrain, powertrain_mean, powertrain_std = normalize_spectrogram(powertrain)
        
        mixture = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        road = torch.tensor(road, dtype=torch.float32)
        wind = torch.tensor(wind, dtype=torch.float32)
        powertrain = torch.tensor(powertrain, dtype=torch.float32)
        
        sources = torch.stack([road, wind, powertrain], dim=0)
        means = torch.tensor([road_mean, wind_mean, powertrain_mean], dtype=torch.float32)
        stds = torch.tensor([road_std, wind_std, powertrain_std], dtype=torch.float32)
        
        return mixture, sources, means, stds

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)  # Added layer
        self.bottleneck = self.conv_block(1024, 2048)  # Increased depth
        self.dec5 = self.conv_block(2048, 1024)  # Added layer
        self.dec4 = self.conv_block(1024, 512)
        self.dec3 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))
        enc5 = self.enc5(F.max_pool2d(enc4, kernel_size=2))
        bottleneck = self.bottleneck(F.max_pool2d(enc5, kernel_size=2))
        
        dec5 = self.upconv_block(2048, 1024)(bottleneck)
        dec5 = self._pad_concat(dec5, enc5)
        dec5 = self.dec5(dec5)
        
        dec4 = self.upconv_block(1024, 512)(dec5)
        dec4 = self._pad_concat(dec4, enc4)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv_block(512, 256)(dec4)
        dec3 = self._pad_concat(dec3, enc3)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv_block(256, 128)(dec3)
        dec2 = self._pad_concat(dec2, enc2)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv_block(128, 64)(dec2)
        dec1 = self._pad_concat(dec1, enc1)
        dec1 = self.dec1(dec1)
        
        return self.out_conv(dec1)
    
    def _pad_concat(self, upsampled, bypass):
        if upsampled.size() != bypass.size():
            diffY = bypass.size()[2] - upsampled.size()[2]
            diffX = bypass.size()[3] - upsampled.size()[3]
            upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return torch.cat((upsampled, bypass), dim=1)

def weighted_mse_loss(predictions, targets, weights):
    weights = weights.view(1, 1, 1, -1)  # Ensure weights have the correct dimensions
    loss = weights * (predictions - targets) ** 2
    return loss.mean()

def train_model(model, train_loader, val_loader, test_loader, num_epochs=20, learning_rate=0.0001, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets, means, stds) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            means, stds = means.to(device), stds.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = weighted_mse_loss(outputs, targets, weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Training Loss: {epoch_loss:.4f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (inputs, targets, means, stds) in enumerate(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    means, stds = means.to(device), stds.to(device)
                    with autocast():
                        outputs = model(inputs)
                        loss = weighted_mse_loss(outputs, targets, weights)
                    val_loss += loss.item() * inputs.size(0)
                    
                    if batch_idx % 10 == 0:
                        print(f"Validation Batch {batch_idx}/{len(val_loader)}, Loss: {loss.item():.4f}")

            val_loss /= len(val_loader.dataset)
            print(f"Validation Loss: {val_loss:.4f}")

    # Test the model
    if test_loader is not None:
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets, means, stds) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                means, stds = means.to(device), stds.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = weighted_mse_loss(outputs, targets, weights)
                test_loss += loss.item() * inputs.size(0)
        print(f'Test Loss: {test_loss/len(test_loader.dataset):.4f}')
    
    torch.save(model.state_dict(), 'unet_model.pth')
    print("Model saved to 'unet_model.pth'")

if __name__ == "__main__":
    manifest_file = '/path/to/data_manifest.json'
    dataset = AudioDataset(manifest_file)
    
    use_all_data_for_training = True  # Set this to True if you want to use all data for training
    train_size = int(0.8 * len(dataset)) if not use_all_data_for_training else len(dataset)
    val_size = int(0.1 * len(dataset)) if not use_all_data_for_training else 0
    test_size = len(dataset) - train_size - val_size if not use_all_data_for_training else 0
    
    if use_all_data_for_training:
        train_dataset = dataset
        val_dataset, test_dataset = None, None
    else:
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) if test_dataset else None
    
    model = UNet(in_channels=1, out_channels=3)
    weights = torch.tensor([1.0, 1.0, 3.0]).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_model(model, train_loader, val_loader, test_loader, num_epochs=20, learning_rate=0.0001, device='cuda' if torch.cuda.is_available() else 'cpu')
