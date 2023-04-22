import torch
import torch.optim as optim
import torch.quantization
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from modules.GDN import GDN
from torch_optimizer import Lamb
from torch.utils.data import DataLoader
from torchvision import transforms
from modules.normalize import Normalize, Denormalize

class Encoder(nn.Sequential):
    """The analysis transform."""
    def __init__(self, num_filters):
        super().__init__(
            nn.Conv2d(3, num_filters, kernel_size=9, stride=4, padding=4, bias=True),
            GDN(num_filters),
            nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, bias=True),
            GDN(num_filters),
            nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, bias=False)
        )

class Decoder(nn.Sequential):
    """The synthesis transform."""
    def __init__(self, num_filters):
        super().__init__(
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            GDN(num_filters, inverse=True),
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            GDN(num_filters, inverse=True),
            nn.ConvTranspose2d(num_filters, 3, kernel_size=9, stride=4, padding=4, output_padding=3, bias=True),
        )
    
class AE(nn.Module):
    def __init__(self, lmbda, num_filters):
        super(AE, self).__init__()
        self.lmbda = lmbda
        self.encoder = Encoder(num_filters)
        self.decoder = Decoder(num_filters)

    def forward(self, x):
        y = self.encoder(x)
        x_hat = self.decoder(y)
        return y, x_hat

    def loss_function(self, x, x_hat):
        mse = nn.MSELoss()(x, x_hat)
        loss = self.lmbda * mse
        return loss, mse

    def train_step(self, x, optimizer):
        optimizer.zero_grad()
        y, x_hat = self.forward(x)
        loss, mse = self.loss_function(x, x_hat)
        loss.backward()
        optimizer.step()
        return {'loss': loss.item(), 'mse': mse.item()}

    def test_step(self, x):
        with torch.no_grad():
            y, x_hat = self.forward(x)
            loss, mse = self.loss_function(x, x_hat)
        return {'loss': loss.item(), 'mse': mse.item()}

    def compress(self, x):
        y = self.encoder(x)
        x_shape = torch.tensor(x.shape[1:])
        y_shape = torch.tensor(y.shape[1:])
        return y, x_shape, y_shape

    def decompress(self, y, x_shape, y_shape):
        x_hat = self.decoder(y)
        x_hat = x_hat[0, :x_shape[0], :x_shape[1], :].round()
        return torch.clamp(x_hat, 0, 255).byte()

"""
if __name__ == "__main__":
    # Initialize the autoencoder with the desired number of filters
    num_filters = 64
    autoencoder = SimpleAutoencoder(num_filters)

    # Generate a random input tensor with the shape (batch_size, channels, height, width)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 256, 256)

    # Pass the input tensor through the autoencoder
    output_tensor = autoencoder(input_tensor)

    # Print the input and output tensors for visual inspection
    print("Input tensor:")
    print(input_tensor)

    print("\nOutput tensor:")
    print(output_tensor)
"""
