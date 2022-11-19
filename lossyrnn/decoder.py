import os
import argparse

from torchvision.utils import save_image

import numpy as np

import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, help='path to model')
parser.add_argument('--input', required=True, type=str, help='input codes')
parser.add_argument('--output', default='.', type=str, help='output folder')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--iterations', type=int, default=16, help='unroll iterations')
args = parser.parse_args()

content = np.load(args.input)
codes = np.unpackbits(content['codes'])
codes = np.reshape(codes, content['shape']).astype(np.float32) * 2 - 1

codes = torch.from_numpy(codes)
iters, batch_size, channels, height, width = codes.size()
height = height * 16
width = width * 16

codes = Variable(codes, volatile=True)

import network

decoder = network.DecoderCell()
decoder.eval()

decoder.load_state_dict(torch.load(args.model))

with torch.no_grad():
    # hidden cell state, each image is differnet, but with the same start value
    decoder_h_1 = (torch.zeros(batch_size, 512, height // 16, width // 16),
                  torch.zeros(batch_size, 512, height // 16, width // 16))
    decoder_h_2 = (torch.zeros(batch_size, 512, height // 8, width // 8),
                  torch.zeros(batch_size, 512, height // 8, width // 8))
    decoder_h_3 = (torch.zeros(batch_size, 256, height // 4, width // 4),
                  torch.zeros(batch_size, 256, height // 4, width // 4))
    decoder_h_4 = (torch.zeros(batch_size, 128, height // 2, width // 2),
                  torch.zeros(batch_size, 128, height // 2, width // 2))

if args.cuda and torch.cuda.is_available() == True:
    decoder = decoder.cuda()
    codes = codes.cuda()
    decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
    decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
    decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
    decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

image = torch.zeros(1, 3, height, width) + 0.5
for iters in range(min(args.iterations, codes.size(0))):
    output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
        codes[iters], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
    image = image + output.data.cpu()
    save_image(image, os.path.join(args.output, '{:02d}.png').format(iters))
