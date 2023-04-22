import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from network import AE
from modules.dataset import CLICDataset
import sys
import argparse
import os
import numpy as np
from PIL import Image

def get_clic_dataset(data_root, patch_size):
    dataset = CLICDataset(data_root, patch_size)
    return dataset

def save_compressed_image(y, epoch, path):
    # save the compressed Image
    y_np = (y.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    print(f"Compressed Image shape: {y_np.shape}")
    y_out = Image.fromarray(y_np)
    y_out.save(f"{path}/{epoch}.png")

def train(args):
    batch_idx = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AE(args.lmbda, args.num_filters).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    train_dataset = get_clic_dataset(args.train_root, args.patch_size)
    validation_dataset = get_clic_dataset(args.val_root, args.patch_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True)

    sample_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for index in sample_indices:
        sample_image = validation_dataset[index]
        os.makedirs(f"{args.test_path}/internal_{index}", exist_ok=True)
        sample_image_out = Image.fromarray((sample_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        sample_image_out.save(f"{args.test_path}/internal_{index}/sample_image.png")

    writer = SummaryWriter(log_dir=args.train_path)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch_idx, x  in enumerate(train_loader):
            x = x.to(device)
            metrics = model.train_step(x, optimizer)
            train_loss += metrics['loss']
            if (batch_idx + 1) % args.steps_per_epoch == 0:
                break

        train_loss /= (batch_idx + 1)

        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch_idx, x in enumerate(validation_loader):
                x = x.to(device)
                metrics = model.test_step(x)
                validation_loss += metrics['loss']
        validation_loss /= (batch_idx + 1)


        # Save multiple compressed images for each epoch
        with torch.no_grad():
            for index in sample_indices:
                sample_image = validation_dataset[index].unsqueeze(0).to(device)
                compressed = model.encoder(sample_image)
                sample_reconstructed = model.decoder(compressed)
                os.makedirs(f"{args.test_path}/internal_{index}", exist_ok=True)
                save_compressed_image(sample_reconstructed.cpu(), epoch, f"{args.test_path}/internal_{index}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', validation_loss, epoch)

        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.6f}, Validation Loss: {validation_loss:.6f}")

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), f"{args.model_path}_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), args.model_path)
    writer.close()


def parse_args(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyper-parameters
    parser.add_argument("--lmbda", type=float, default=0.01, help="Lambda for rate-distortion tradeoff.")
    parser.add_argument("--num_filters", type=int, default=128, help="Number of filters per layer.")
    parser.add_argument("--train_path", default="/tmp/train_bls2017", help="Path where to log training metrics for TensorBoard.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation.")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of image patches for training and validation.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs for training.")
    parser.add_argument("--steps_per_epoch", type=int, default=1000, help="Perform validation and produce logs after this many batches.")
    parser.add_argument("--model_path", default="./models/", help="Path where to save/load the trained model.")
    parser.add_argument("--train_root", default="./data/train", help="Path to the training data.")
    parser.add_argument("--val_root", default="./data/valid", help="Path to the validation data.")
    parser.add_argument("--test_path", default="./test/", help="Path to the validation data.")
    parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving model checkpoints.")

    args = parser.parse_args(argv[1:])
    return args

if __name__ == "__main__":
    args = parse_args(sys.argv)
    print(args)
    train(args)
