import argparse
import torch
from torch.utils.data import DataLoader
from openunmix import model, transforms, utils
from CustomDataset import CustomDataset  # Assuming CustomDataset is saved in CustomDataset.py

def get_dataset(args):
    return CustomDataset(root=args.root, split='train', input_file=args.input_file, output_files=args.output_files), \
           CustomDataset(root=args.root, split='valid', input_file=args.input_file, output_files=args.output_files)

def train(args, unmix, encoder, device, train_loader, optimizer):
    unmix.train()
    pbar = tqdm(train_loader, disable=args.quiet)
    for batch in pbar:
        x, y = batch['mixture'].to(device), batch['targets'].to(device)
        optimizer.zero_grad()
        X = encoder(x)
        Y_hat = unmix(X)
        Y = encoder(y)
        loss = sum(torch.nn.functional.mse_loss(Y_hat[i], Y[i]) for i in range(len(Y)))
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

def main():
    parser = argparse.ArgumentParser(description="Train Open-Unmix Model")
    parser.add_argument('--root', type=str, required=True, help="Root directory of dataset")
    parser.add_argument('--input_file', type=str, required=True, help="Input file name")
    parser.add_argument('--output_files', type=str, required=True, help="Comma-separated list of output file names")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--quiet', action='store_true', help="Less verbose training")
    parser.add_argument('--no_cuda', action='store_true', default=False, help="Disables CUDA training")
    args = parser.parse_args()

    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    train_dataset, valid_dataset = get_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    stft, _ = transforms.make_filterbanks(n_fft=4096, n_hop=1024, sample_rate=train_dataset.sample_rate)
    encoder = torch.nn.Sequential(stft, model.ComplexNorm(mono=False)).to(device)
    unmix = model.OpenUnmix(n_fft=4096, n_hop=1024, nb_channels=2).to(device)
    optimizer = torch.optim.Adam(unmix.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train(args, unmix, encoder, device, train_loader, optimizer)
        # Optional: Add validation step and checkpoint saving

if __name__ == "__main__":
    main()
