import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from trainn import UNet  # Importing UNet from trainn.py

def load_model(model_path, device):
    model = UNet(in_channels=1, out_channels=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def prepare_input_spectrogram(spectrogram_path):
    spectrogram = np.load(spectrogram_path)
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return spectrogram

def predict_separation(model, spectrogram, device):
    spectrogram = spectrogram.to(device)
    with torch.no_grad():
        outputs = model(spectrogram)
    outputs = outputs.cpu().numpy()[0]
    road_output, wind_output, powertrain_output = outputs[0], outputs[1], outputs[2]
    return road_output, wind_output, powertrain_output

def save_spectrogram(spectrogram, output_path):
    np.save(output_path, spectrogram)

def plot_spectrogram(spectrogram, title):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=44100, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define paths
    model_path = 'unmix_model.pth'
    input_spectrogram_path = '/path/to/your/input_mixture_spectrogram.npy'  # Update with your path
    road_output_path = 'road_pred.npy'
    wind_output_path = 'wind_pred.npy'
    powertrain_output_path = 'powertrain_pred.npy'

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = load_model(model_path, device)

    # Prepare input spectrogram
    input_spectrogram = prepare_input_spectrogram(input_spectrogram_path)

    # Predict separation
    road_output, wind_output, powertrain_output = predict_separation(model, input_spectrogram, device)

    # Save predicted spectrograms
    save_spectrogram(road_output, road_output_path)
    save_spectrogram(wind_output, wind_output_path)
    save_spectrogram(powertrain_output, powertrain_output_path)

    # Plot predicted spectrograms
    plot_spectrogram(road_output, 'Predicted Road Spectrogram')
    plot_spectrogram(wind_output, 'Predicted Wind Spectrogram')
    plot_spectrogram(powertrain_output, 'Predicted Powertrain Spectrogram')
