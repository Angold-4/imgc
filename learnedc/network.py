import torch 
import torch.nn as nn
import torch.nn.functional as F

class GDN(nn.Module):
    def __init__(self, channels, beta_min=1e-6, gamma_init=0.1, reparam_offset=2):
        super(GDN, self).__init__()
        self.channels = channels
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** 0.5
        self.gamma_bound = self.reparam_offset

        self.build()

    def build(self):
        self.gamma = nn.Parameter(torch.Tensor(self.channels).fill_(self.gamma_init))
        self.beta = nn.Parameter(torch.Tensor(self.channels).fill_(self.beta_min))

    def forward(self, x):
        beta = torch.abs(self.beta) + self.beta_bound
        gamma = torch.abs(self.gamma) + self.gamma_bound
        norm = torch.sqrt(torch.sum((x ** 2), dim=1, keepdim=True) / self.channels + self.pedestal)
        norm = norm * (beta[None, :, None, None])
        return x / (norm ** gamma[None, :, None, None])

class IGDN(GDN):
    def forward(self, x):
        beta = torch.abs(self.beta) + self.beta_bound
        gamma = torch.abs(self.gamma) + self.gamma_bound
        return x * (x.abs() ** (gamma[None, :, None, None] - 1) + beta[None, :, None, None] ** 2) ** 0.5

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
            GDN(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            GDN(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            GDN(256),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
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
            IGDN(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            IGDN(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            IGDN(64),
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
