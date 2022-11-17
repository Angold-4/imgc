import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class ConvRNNCellBase(nn.Module):
    def __repr__(self):
        s = ('{name}({input_channels}, {hidden_channels}, kernel_size={kernel_size}'', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        s += ', hidden_kernel_size={hidden_kernel_size}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


"""
ConvLSTM
A Convolutional Long short-term Memory RNN
The layers and cell gate will be preserved across learning
"""
class ConvLSTMCell(ConvRNNCellBase):
    def __init__(self,
                # the reason that this model support variable rate
                input_channels,  # number of channels in the input image (rgb, can be any size)
                hidden_channels, # number of channels produced by the convolution
                kernel_size=3,   # 3x3 size of the convolution kernel
                stride=1,
                padding=0,
                dilation=1,
                hidden_kernel_size=1,
                bias=True):

        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.hidden_kernel_size = _pair(hidden_kernel_size)

        hidden_padding = _pair(hidden_kernel_size // 2)

        gate_channels = 4 * self.hidden_channels # four groups

        self.conv_ih = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=gate_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias)

        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=gate_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            dilation=1,
            bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_ih.reset_parameters()
        self.conv_hh.reset_parameters()

    def forward(self, input, hidden): # on rnn, hidden stands for the last hidden
        hx, cx = hidden # previous output and cell state

        gates = self.conv_ih(input) + self.conv_hh(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1) # each convolution group
        # chunk -> list of Tensors

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate) # cell gate means the values in ingate that need to be preserved
        outgate = F.sigmoid(outgate)

        # forgetgate * previous cell state + invalue * decide which value to be inserted
        cy = (forgetgate * cx) + (ingate * cellgate) # cell state

        # final output is performing a filtering on the outvalues
        hy = outgate * F.tanh(cy) # output

        return hy, cy
