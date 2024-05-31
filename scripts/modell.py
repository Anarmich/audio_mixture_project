import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

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
        print("Forward pass started")
        
        # Encoder
        enc1 = self.enc1(x)
        print(f"enc1 shape: {enc1.shape}")
        enc2 = self.enc2(enc1)
        print(f"enc2 shape: {enc2.shape}")
        enc3 = self.enc3(enc2)
        print(f"enc3 shape: {enc3.shape}")

        # Bottleneck
        bottleneck = self.bottleneck(enc3)
        print(f"bottleneck shape: {bottleneck.shape}")

        # Decoder
        dec3 = self.upconv3(bottleneck)
        print(f"dec3 shape after upconv: {dec3.shape}")

        # Padding to match the size
        if dec3.size() != enc3.size():
            diffY = enc3.size()[2] - dec3.size()[2]
            diffX = enc3.size()[3] - dec3.size()[3]
            dec3 = F.pad(dec3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            print(f"dec3 shape after padding: {dec3.shape}")

        dec3 = torch.cat((dec3, enc3), dim=1)
        print(f"dec3 shape after cat: {dec3.shape}")
        dec3 = self.dec3(dec3)
        print(f"dec3 shape after conv_block: {dec3.shape}")

        dec2 = self.upconv2(dec3)
        print(f"dec2 shape after upconv: {dec2.shape}")

        # Padding to match the size
        if dec2.size() != enc2.size():
            diffY = enc2.size()[2] - dec2.size()[2]
            diffX = enc2.size()[3] - dec2.size()[3]
            dec2 = F.pad(dec2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            print(f"dec2 shape after padding: {dec2.shape}")

        dec2 = torch.cat((dec2, enc2), dim=1)
        print(f"dec2 shape after cat: {dec2.shape}")
        dec2 = self.dec2(dec2)
        print(f"dec2 shape after conv_block: {dec2.shape}")

        dec1 = self.upconv1(dec2)
        print(f"dec1 shape after upconv: {dec1.shape}")

        # Padding to match the size
        if dec1.size() != enc1.size():
            diffY = enc1.size()[2] - dec1.size()[2]
            diffX = enc1.size()[3] - dec1.size()[3]
            dec1 = F.pad(dec1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            print(f"dec1 shape after padding: {dec1.shape}")

        dec1 = torch.cat((dec1, enc1), dim=1)
        print(f"dec1 shape after cat: {dec1.shape}")
        dec1 = self.dec1(dec1)
        print(f"dec1 shape after conv_block: {dec1.shape}")

        out = self.out_conv(dec1)
        print(f"output shape: {out.shape}")

        print("Forward pass finished")
        return out

def extract_spectrogram(audio_path, n_fft=2048, hop_length=512):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db, sr

def plot_spectrogram(data, title, sr=22050, hop_length=512):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(data, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

if __name__ == "__main__":
    # Example with batch size of 1
    batch_size = 1
    in_channels = 1
    out_channels = 3
    height = 1025
    width = 345

    # Instantiate the model
    model = UNet(in_channels=in_channels, out_channels=out_channels)

    # Create a batch of spectrograms
    x = torch.randn(batch_size, in_channels, height, width)  # Batch of 1 spectrogram

    # Forward pass through the model
    print("Running forward pass with input shape:", x.shape)
    output = model(x)

    # Print the output shape
    print("Output shape:", output.shape)  # Should print torch.Size([1, 3, 1025, 345])

    # Convert the input and output tensors to numpy arrays
    x_np = x[0, 0].detach().cpu().numpy()  # Convert the input tensor to numpy array
    output_np = output[0].detach().cpu().numpy()  # Convert the output tensor to numpy array

    # Plot the input spectrogram
    plot_spectrogram(x_np, "Input Spectrogram", sr=44100, hop_length=512)

    # Plot the output spectrograms
    for i in range(out_channels):
        plot_spectrogram(output_np[i], f"Output Spectrogram - Channel {i+1}", sr=44100, hop_length=512)
