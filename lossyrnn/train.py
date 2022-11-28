import time
import os
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as data

from torchvision.utils import save_image

from torchvision import transforms

import dataset

def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'

    encoder.load_state_dict(
        torch.load('checkpoint{}encoder_{}_{:08d}.pth'.format(os.sep, s, epoch)))
    binarizer.load_state_dict(
        torch.load('checkpoint{}binarizer_{}_{:08d}.pth'.format(os.sep, s, epoch)))
    decoder.load_state_dict(
        torch.load('checkpoint{}decoder_{}_{:08d}.pth'.format(os.sep, s, epoch)))


def save(index, epoch=True):
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    if epoch:
        s = 'epoch'
    else:
        s = 'iter'

    torch.save(encoder.state_dict(), 'checkpoint{}encoder_{}_{:08d}.pth'.format(os.sep, s, index))

    torch.save(binarizer.state_dict(), 'checkpoint{}binarizer_{}_{:08d}.pth'.format(os.sep, s, index))

    torch.save(decoder.state_dict(), 'checkpoint{}decoder_{}_{:08d}.pth'.format(os.sep, s, index))

def batchsave(original, learned, batchnum, epoch):
    # save 1 image from the verification patch (0->156)
    if batchnum > 156:
        return

    if not os.path.exists('cptrainimg'):
        os.mkdir('cptrainimg')

    if not os.path.exists('cptrainimg{}batch_{}'.format(os.sep, batchnum)):
        os.mkdir('cptrainimg{}batch_{}'.format(os.sep, batchnum))

    batchimg = torch.randn((3, 32, 32)) # define as global variable
    for ib in range(learned[0].size(0)): # image in baches
        # layer -> a display layer: orginalx1, learnedx16, outputx1
        layer = original[ib]
        output = torch.zeros((3, 32, 32))
        if torch.cuda.is_available():
            output = output.cuda()
        for i in range(0, 16):
            xl = learned[i][ib]
            layer = torch.cat((layer, xl), 2)
            output += xl

        layer = torch.cat((layer, output), 2)
        if ib == 0: # init
            batchimg = layer
        else:
            batchimg = torch.cat((batchimg, layer), 1)

    save_image(batchimg, 'cptrainimg{}batch_{}{}epoch_{}.jpg'.format(os.sep, batch, os.sep, epoch))


def saveimg(data, learned, output, epoch, batch):
    if not os.path.exists('cptrainimg'):
        os.mkdir('cptrainimg')

    if not os.path.exists('cptrainimg{}batch_{}'.format(batch)):
        os.mkdir('batch_{}'.format(batch))

    xdata = data[0]
    for i in range(1, 32):
        xdata = torch.cat((xdata, data[i]), 1)
    save_image(xdata, 'cptrainimg/data_{}_{}.png'.format(epoch, batch))

    xdata = learned[0]
    for i in range(1, 32):
        xdata = torch.cat((xdata, learned[i]), 1)
    save_image(xdata, 'cptrainimg/learned_{}_{}_{}.png'.format(epoch, batch))

    xdatar = output[0][0]
    xdatag = output[0][1]
    xdatab = output[0][2]
    xdatal = torch.cat((xdatar, xdatag, xdatab), 1)
    xdata = xdatal
    for i in range(1, 32):
        xdatar = output[i][0]
        xdatag = output[i][1]
        xdatab = output[i][2]
        xdatal = torch.cat((xdatar, xdatag, xdatab), 1)
        xdata = torch.cat((xdata, xdatal), 0)

    save_image(xdata, 'cptrainimg/batch_{}/residuals_{}.png'.format(batch, epoch), normalize=True)


if __name__ == "__main__":

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

    if torch.cuda.is_available() == False and torch.cuda.is_available() != args.cuda:
        print("It seems that your device do not support cuda, if you want to use CPU to training, please remove '--cuda'")
        exit()

    """ Load 32x32 patches from images to training """

    """ Training consists of two parts -> 157 Verification patch + many training example """

    train_transform = transforms.Compose([
        transforms.RandomCrop((32, 32)), # random crop 32x32 piece from that photo
        transforms.ToTensor(),
    ])

    train_set = dataset.ImageFolder(root=args.train, transform=train_transform)
    """ Since we do not want to mix up the validate && training dataset, set suffle = False """
    train_loader = data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

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


# Decays the learning rate of each parameter group by gamma 
# once the number of epoch reaches one of the milestones.
# performs backward on this final model
    scheduler = LS.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100], gamma=0.5)

    last_epoch = 0
    if args.checkpoint:
        resume(args.checkpoint)
        last_epoch = args.checkpoint
        scheduler.last_epoch = last_epoch - 1

    for epoch in range(last_epoch + 1, args.max_epochs + 1):
        for batch, data in enumerate(train_loader):
            batch_t0 = time.time()

            if args.cuda:
                ## init lstm state -> The hidden layer (cell state)
                encoder_h_1 = (torch.zeros(data.size(0), 256, 8, 8).cuda(),
                               torch.zeros(data.size(0), 256, 8, 8).cuda())
                encoder_h_2 = (torch.zeros(data.size(0), 512, 4, 4).cuda(),
                               torch.zeros(data.size(0), 512, 4, 4).cuda())
                encoder_h_3 = (torch.zeros(data.size(0), 512, 2, 2).cuda(),
                               torch.zeros(data.size(0), 512, 2, 2).cuda())

                decoder_h_1 = (torch.zeros(data.size(0), 512, 2, 2).cuda(),
                               torch.zeros(data.size(0), 512, 2, 2).cuda())
                decoder_h_2 = (torch.zeros(data.size(0), 512, 4, 4).cuda(),
                               torch.zeros(data.size(0), 512, 4, 4).cuda())
                decoder_h_3 = (torch.zeros(data.size(0), 256, 8, 8).cuda(),
                               torch.zeros(data.size(0), 256, 8, 8).cuda())
                decoder_h_4 = (torch.zeros(data.size(0), 128, 16, 16).cuda(),
                               torch.zeros(data.size(0), 128, 16, 16).cuda())
                patches = data.cuda()

            else:
                encoder_h_1 = (torch.zeros(data.size(0), 256, 8, 8),
                               torch.zeros(data.size(0), 256, 8, 8))
                encoder_h_2 = (torch.zeros(data.size(0), 512, 4, 4),
                               torch.zeros(data.size(0), 512, 4, 4))
                encoder_h_3 = (torch.zeros(data.size(0), 512, 2, 2),
                               torch.zeros(data.size(0), 512, 2, 2))

                decoder_h_1 = (torch.zeros(data.size(0), 512, 2, 2),
                               torch.zeros(data.size(0), 512, 2, 2))
                decoder_h_2 = (torch.zeros(data.size(0), 512, 4, 4),
                               torch.zeros(data.size(0), 512, 4, 4))
                decoder_h_3 = (torch.zeros(data.size(0), 256, 8, 8),
                               torch.zeros(data.size(0), 256, 8, 8))
                decoder_h_4 = (torch.zeros(data.size(0), 128, 16, 16),
                               torch.zeros(data.size(0), 128, 16, 16))
                patches = data

            solver.zero_grad()

            losses = []
            learned = [] # the img residuals that feed into next iteration

            # res = patches - 0.5 # why do we need that?

            res = patches


            bp_t0 = time.time()

            output = torch.Tensor((3, 3, 32, 32))

            """ Perform many iterations in the same data """
            for i in range(args.iterations):
                # Auto-Gain Addictive Reconstruction:
                # we call that encoder, means call the forward function (encoder_h_x stands for lstm cell state in our model)
                # for every data, it is different, init 0 info, for the same data, it is accumulated
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(res, encoder_h_1, encoder_h_2, encoder_h_3)
                codes = binarizer(encoded)
                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                        codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

                res = res - output # result: result plus residuals -> final reconstructed image 
                # the similarity between original and reconstructed (loss function)

                learned.append(output) # learned result, feed to the next iteration
                losses.append(res.abs().mean())

            batchsave(original=patches, learned=learned, batchnum=batch, epoch=epoch)

            bp_t1 = time.time()
            # Sum of the residuals
            loss = sum(losses) / args.iterations # sum them together
            loss.backward() # Tensor.backward(), calculate gradiance

            solver.step()   # solver update the model

            batch_t1 = time.time()

            print('[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.
                format(epoch, batch + 1, len(train_loader), loss.item(), bp_t1 - bp_t0, batch_t1 - batch_t0)) 

            print(('{:.4f} ' * args.iterations + '\n').format(* [l.data.item() for l in losses]))

            index = (epoch - 1) * len(train_loader) + batch

            ## save checkpoint and display the output image every 500 training steps
            if index % 500 == 0:
                save(0, False)

        scheduler.step()
        save(epoch)
