import os
import random
import helpers
import torch.backends.cudnn as cudnn
import torch

def command_line_options():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='openset_mnist_kmnist_emnist')
    parser.add_argument('--dataroot', required=False, default='/tmp', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
    parser.add_argument('--niter', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--l1', type=float, default=100, help='weight for reconstruction loss')
    parser.add_argument('--l2', type=float, default=1, help='weight for fake loss of G')
    parser.add_argument('--l_d', type=float, default=50, help='weight for fake loss of D')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes to generate fake images')
    parser.add_argument("--approach", "-a", choices=['SoftMax', 'BG', 'entropic', 'objectosphere'], default='SoftMax')
    parser.add_argument("--arch", default='LeNet_plus_plus', choices=['Resnet18', 'LeNet_plus_plus'])
    parser.add_argument("--gen_arch", default='small', choices=['small', 'large'])
    parser.add_argument('--filter_threshold', "-ft", dest="ft", default=0.8, type=float)
    parser.add_argument('--gpu', default=2, type=int)
    parser.add_argument("--fake_method", "-fm",  default='Cross_gan', choices=['SoftMax', 'BG', 'Cross_KU', 'Cross_noise', 'Cross_adv', 'Cross_gan'])
    parser.add_argument("--filter", default='False', choices=['True', 'False'])
    parser.add_argument("--remove_unknown", default='True', choices=['True', 'False'])
    parser.add_argument("--wgan", default='True', choices=['True', 'False'])
    # input info
    return parser.parse_args()

def check(opt):

    folders = [opt.outf, './model', './figures']
    try:
        for folder in folders:
            os.makedirs(folder)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)

    helpers.seed_everything(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
        raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)
