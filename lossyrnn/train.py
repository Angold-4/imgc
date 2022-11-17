import time
import os
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as data

from torch.autograd import Variable
from torchvision import transforms, datasets

""" Parse the arguments """

parser = argparse.ArgumentParser()
parser.add_argument( '--batch-size', '-N', type=int, default=32, help='batch size')
parser.add_argument( '--train', '-f', required=True, type=str, help='folder of training images')
parser.add_argument('--max-epochs', '-e', type=int, default=200, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument('--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
args = parser.parse_args()

if torch.cuda.is_available() != args.cuda:
    print("It seems that your device do not support cuda, if you want to use CPU to training, please remove '--cuda'")
    exit()

""" Load 32x32 patches from images to training """

train_transform = transforms.Compose([
    transforms.RandomCrop((32, 32)), # random crop 32x32 piece from that photo
    transforms.ToTensor(),
])

train_set = datasets.ImageFolder(root=args.train, transform=train_transform)

train_loader = data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

print('total images: {}; total batches: {}'.format(len(train_set), len(train_loader)))


""" Load networks on CPU / GPU """

import network

encoder = network.EncoderCell()
binarizer = network.Binarizer()
decoder = network.DecoderCell()
if args.cuda:
    encoder = encoder.cuda()
    binarizer = binarizer.cuda()
    decoder = decoder.cuda()

# Optimizer
solver = optim.Adam([
    {'params': encoder.parameters()}, 
    {'params': binarizer.parameters()},
    {'params': decoder.parameters()},], lr=args.lr)


def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'

    encoder.load_state_dict(
        torch.load('checkpoint/encoder_{}_{:08d}.pth'.format(s, epoch)))
    binarizer.load_state_dict(
        torch.load('checkpoint/binarizer_{}_{:08d}.pth'.format(s, epoch)))
    decoder.load_state_dict(
        torch.load('checkpoint/decoder_{}_{:08d}.pth'.format(s, epoch)))


def save(index, epoch=True):
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    if epoch:
        s = 'epoch'
    else:
        s = 'iter'

    torch.save(encoder.state_dict(), 'checkpoint/encoder_{}_{:08d}.pth'.format(s, index))

    torch.save(binarizer.state_dict(), 'checkpoint/binarizer_{}_{:08d}.pth'.format(s, index))

    torch.save(decoder.state_dict(), 'checkpoint/decoder_{}_{:08d}.pth'.format(s, index))


# resume()

# Decays the learning rate of each parameter group by gamma 
# once the number of epoch reaches one of the milestones.
scheduler = LS.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100], gamma=0.5)
