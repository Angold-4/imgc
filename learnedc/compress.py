from network import VAE
import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

# The VAE definition and training function should be imported from another module
# Example:
# from vae_training import VAE, train

def compress_image(input_path, output_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    img = Image.open(input_path)
    img_tensor = torch.Tensor(transform(img)).unsqueeze(0).to(device)
    img_compressed, _, _ = model(img_tensor)
    save_image(img_compressed, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compress a 32x32 image")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("output", help="Path to the output image")
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 64
    model = VAE(latent_dim).to(device)
    # Load the trained model weights
    model.load_state_dict(torch.load("models/vae_weights_19.pth", map_location=device))

    compress_image(args.input, args.output, model, device)
