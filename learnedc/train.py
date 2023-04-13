from network import VAE
import torch
import os
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Load the VAE model
latent_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim).to(device)


# Define the loss function
def loss_function(recon_x, x, mu, logvar, mu_hyper, logvar_hyper, beta=1.0):
    distortion = F.mse_loss(recon_x, x, reduction='sum')
    KLD_main = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD_hyper = -0.5 * torch.sum(1 + logvar_hyper - mu_hyper.pow(2) - logvar_hyper.exp())
    rate = KLD_main + KLD_hyper
    return distortion + beta * rate

# Set up the optimizer

# Define the data loading
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = STL10(root='./data', split='train', download=True, transform=transform)
testset = STL10(root='./data', split='test', download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=24, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=24, shuffle=False, num_workers=4)

optimizer = optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-5)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

if __name__ == '__main__':
    if not os.path.exists("model"):
        os.makedirs("model")
    if not os.path.exists("vis"):
        os.makedirs("vis")

    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(trainloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu_main, logvar_main, mu_hyper, logvar_hyper = vae(data)
            loss = loss_function(recon_batch, data, mu_main, logvar_main, mu_hyper, logvar_hyper)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(trainloader.dataset):.4f}') # type: ignore

        # Evaluation loop
        vae.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(testloader):
                data = data.to(device)
                recon_batch, mu_main, logvar_main, mu_hyper, logvar_hyper = vae(data)
                test_loss += loss_function(recon_batch, data, mu_main, logvar_main, mu_hyper, logvar_hyper).item()

        test_loss /= len(testloader.dataset) # type: ignore
        print(f'====> Test set loss: {test_loss:.4f}')
        scheduler.step(test_loss)
        torch.save(vae.state_dict(), f"model/vae_weights_{epoch}.pth")

        # Visualization
        with torch.no_grad():
            # Get a test image from the testloader
            test_images, _ = next(iter(testloader))
            test_image = test_images[0].unsqueeze(0).to(device)

            # Encode, sample and decode the test image
            mu_main, logvar_main = vae.encoder(test_image)
            z_main = vae.reparameterize(mu_main, logvar_main)
            mu_hyper, logvar_hyper = vae.hyperprior_encoder(z_main)
            z_hyper = vae.reparameterize(mu_hyper, logvar_hyper)
            z_hyper_decoded = vae.hyperprior_decoder(z_hyper)
            reconstructed_image = vae.decoder(torch.cat((z_main, z_hyper_decoded), dim=1))

            # Visualize original image, latent representation, and reconstructed image
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # Original image
            axes[0].imshow(test_image.cpu().squeeze().permute(1, 2, 0))
            axes[0].set_title('Original Image')

            # Latent representation
            latent_representation = z_hyper.cpu().squeeze().numpy()
            num_rows = int(np.ceil(np.sqrt(latent_representation.shape[0])))
            num_cols = int(np.floor(latent_representation.shape[0] / num_rows))
            axes[1].imshow(latent_representation.reshape((num_rows, num_cols)), cmap='gray')
            axes[1].set_title('Latent Representation')

            # Reconstructed image
            axes[2].imshow(reconstructed_image.cpu().squeeze().permute(1, 2, 0))
            axes[2].set_title('Reconstructed Image')

            # Save the visualization
            plt.savefig(f"vis/visualization_{epoch}.png")
            plt.close(fig)
