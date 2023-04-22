import torch
import torch.nn as nn
import torch.nn.functional as F

class GDN(nn.Module):
    def __init__(self, channels, inverse=False, rectify=True, alpha_parameter=1, beta_parameter=None, gamma_parameter=None, epsilon_parameter=1e-6):
        super(GDN, self).__init__()
        self.channels = channels
        self.inverse = inverse
        self.rectify = rectify
        self.alpha_parameter = alpha_parameter
        self.beta_parameter = beta_parameter
        self.gamma_parameter = gamma_parameter
        self.epsilon_parameter = epsilon_parameter

        if self.beta_parameter is None:
            self.beta = nn.Parameter(torch.ones(self.channels))
        if self.gamma_parameter is None:
            eye = torch.eye(self.channels)
            self.gamma = nn.Parameter(eye.mul(0.1))

    def forward(self, x):
        if self.rectify:
            x = F.relu(x)

        if self.alpha_parameter == 1 and self.rectify:
            norm_pool = x
        elif self.alpha_parameter == 1:
            norm_pool = torch.abs(x)
        elif self.alpha_parameter == 2:
            norm_pool = torch.square(x)
        else:
            norm_pool = torch.pow(x, self.alpha_parameter)

        norm_pool = F.conv2d(norm_pool, self.gamma.unsqueeze(-1).unsqueeze(-1), padding=0) + self.beta.view(1, -1, 1, 1)

        if self.epsilon_parameter == 1:
            pass
        elif self.epsilon_parameter == 0.5:
            norm_pool = torch.sqrt(norm_pool)
        else:
            norm_pool = torch.pow(norm_pool, self.epsilon_parameter)

        if self.inverse:
            return x * norm_pool
        else:
            return x / norm_pool
