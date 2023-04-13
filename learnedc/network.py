import torch 
import torch.nn as nn
import torch.nn.functional as F

class GDN(nn.Module):
    def __init__(self, channels, beta_min=1e-6):
        super(GDN, self).__init__()
        self.channels = channels
        self.beta_min = beta_min

        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.epsilon = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        x = x * torch.rsqrt(self.epsilon + F.conv2d(x**2, self.beta**2) + self.alpha**2)
        return x

class IGDN(nn.Module):
    def __init__(self, channels, beta_min=1e-6):
        super(IGDN, self).__init__()
        self.channels = channels
        self.beta_min = beta_min

        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.epsilon = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        x = x * torch.sqrt(self.epsilon + F.conv2d(x**2, self.beta**2) + self.alpha**2)
        return x

class HyperpriorEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(HyperpriorEncoder, self).__init__()
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, z):
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)
        return mu, logvar


class HyperpriorDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(HyperpriorDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, latent_dim)

    def forward(self, z):
        return self.fc(z)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512 * 6 * 6, latent_dim)
        self.fc_logvar = nn.Linear(512 * 6 * 6, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(2 * latent_dim, 512 * 6 * 6)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 6, 6)
        x = self.conv_layers(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.hyperprior_encoder = HyperpriorEncoder(latent_dim)
        self.hyperprior_decoder = HyperpriorDecoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_main, logvar_main = self.encoder(x)
        z_main = self.reparameterize(mu_main, logvar_main)
        mu_hyper, logvar_hyper = self.hyperprior_encoder(z_main)
        z_hyper = self.reparameterize(mu_hyper, logvar_hyper)
        z_hyper_decoded = self.hyperprior_decoder(z_hyper)
        x_recon = self.decoder(torch.cat((z_main, z_hyper_decoded), dim=1))
        return x_recon, mu_main, logvar_main, mu_hyper, logvar_hyper
