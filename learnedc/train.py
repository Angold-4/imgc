from network import VAE
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def loss_function(x, x_recon, mean, logvar):
    recon_loss = nn.MSELoss()(x_recon, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_divergence


def train(model, dataloader, optimizer, device):
    model.train()
    for _, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        x_recon, mean, logvar = model(data)
        loss = loss_function(data, x_recon, mean, logvar)
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    # Model, optimizer, and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 64
    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, dataloader, optimizer, device)
        pth = f"vae_weights_{epoch}.pth"
        torch.save(model.state_dict(), pth)
