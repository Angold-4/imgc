import torch 
import torch.nn as nn
import torch.nn.functional as F

"""
class HyperpriorEncoder(nn.Module):
    def __init__(self, latent_dim, hyperprior_latent_dim):
        super(HyperpriorEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, hyperprior_latent_dim * 2)
        )

    def forward(self, z):
        out = self.layers(z)
        mu, logvar = out[:, :hyperprior_latent_dim], out[:, hyperprior_latent_dim:]
        return mu, logvar

class HyperpriorDecoder(nn.Module):
    def __init__(self, latent_dim, hyperprior_latent_dim):
        super(HyperpriorDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(hyperprior_latent_dim, latent_dim // 4),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim)
        )

    def forward(self, w):
        z = self.layers(w)
        return z
"""

class GDN(nn.Module):
    def __init__(self, channels, beta_min=1e-6):
        super(GDN, self).__init__()
        self.channels = channels
        self.beta_min = beta_min

        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.epsilon = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        return x * torch.rsqrt(self.epsilon + F.conv2d(x**2, self.beta**2) + self.alpha**2)

class IGDN(nn.Module):
    def __init__(self, channels, beta_min=1e-6):
        super(IGDN, self).__init__()
        self.channels = channels
        self.beta_min = beta_min

        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.epsilon = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        return x * torch.sqrt(self.epsilon + F.conv2d(x**2, self.beta**2) + self.alpha**2)


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            GDN(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            GDN(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            GDN(256),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            GDN(512),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            GDN(1024),
        )
        self.fc_mu = nn.Linear(1024 * 3 * 3, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 3 * 3, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 1024 * 3 * 3)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            IGDN(512),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            IGDN(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            IGDN(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            IGDN(64),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1024, 3, 3)
        x = self.conv_layers(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
