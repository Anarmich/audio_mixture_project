import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json

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
        
        mixture = torch.tensor(mixture, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        road = torch.tensor(road, dtype=torch.float32)
        wind = torch.tensor(wind, dtype=torch.float32)
        powertrain = torch.tensor(powertrain, dtype=torch.float32)
        
        sources = torch.stack([road, wind, powertrain], dim=0)
        
        return mixture, sources

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
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
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))
        dec4 = self.dec4(F.interpolate(bottleneck, scale_factor=2))
        dec3 = self.dec3(F.interpolate(dec4, scale_factor=2))
        dec2 = self.dec2(F.interpolate(dec3, scale_factor=2))
        dec1 = self.dec1(F.interpolate(dec2, scale_factor=2))
        return self.out_conv(F.interpolate(dec1, scale_factor=2))

def weighted_mse_loss(predictions, targets, weights):
    loss = weights.view(1, 1, 1, -1) * (predictions - targets) ** 2
    return loss.mean()

def train_model(model, train_loader, val_loader, test_loader, num_epochs=20, learning_rate=0.0001, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = weighted_mse_loss(outputs, targets, weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = weighted_mse_loss(outputs, targets, weights)
                val_loss += loss.item()
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
    
    # Test the model
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = weighted_mse_loss(outputs, targets, weights)
            test_loss += loss.item()
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')
    
    torch.save(model.state_dict(), 'unet_model.pth')
    print("Model saved to 'unet_model.pth'")

if __name__ == "__main__":
    manifest_file = '/path/to/data_manifest.json'
    dataset = AudioDataset(manifest_file)
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = UNet(in_channels=1, out_channels=3)
    weights = torch.tensor([1.0, 1.0, 3.0])  # Adjust weights as necessary
    
    train_model(model, train_loader, val_loader, test_loader, num_epochs=20, learning_rate=0.0001, device='cuda')
