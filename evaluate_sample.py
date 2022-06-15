
"""evaluation on trained Combined-GAN and open-set model"""

from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import combinedGAN
import DataManager
import check
import architectures
import random
import visualization

# set parameters
opt = check.command_line_options()
#opt.manualSeed = random.randint(0, 1000)
#opt.dataset = 'mnist'
opt.batchSize = 128
check.check(opt)
print(opt)

device = torch.device(f"cuda:{opt.gpu}" if opt.cuda else "cpu")

# testing dataset
if opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, train=False, download=True,
                         transform=transforms.ToTensor())
    opt.nz = 256
    net = architectures.LeNet_plus_plus(use_BG=opt.arch == "BG")
    net.load_state_dict(torch.load(f"model/trained_model/LeNet_plus_plus_Cross_gan.pth", map_location=torch.device('cpu')))
    opt.gen_arch = 'small'
    combined_gan = combinedGAN.combinedGAN(opt)
    combined_gan.netEnc.load_state_dict(torch.load(f"model/trained_model/Enc_mnist.pth", map_location=torch.device('cpu')))
    combined_gan.netG.load_state_dict(torch.load(f"model/trained_model/G_mnist.pth", map_location=torch.device('cpu')))

elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot,
                           train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                           ]))
    opt.nz = 512
    net = architectures.resnet18(pretrained=False, use_BG=opt.arch == "BG")
    net.load_state_dict(torch.load(f"model/trained_model/Resnet18_Cross_gan.pth", map_location=torch.device('cpu')))
    opt.gen_arch = 'large'
    combined_gan = combinedGAN.combinedGAN(opt)
    combined_gan.netEnc.load_state_dict(
        torch.load(f"model/trained_model/Enc_cifar10.pth", map_location=torch.device('cpu')))
    combined_gan.netG.load_state_dict(
        torch.load(f"model/trained_model/G_cifar10.pth", map_location=torch.device('cpu')))

net.to(device)
combined_gan.netEnc.to(device)
combined_gan.netG.to(device)
assert dataset

n_classes = int(opt.n_classes)
n_samples = opt.batchSize // n_classes

# set balanced batch sampler to load dataset from different classes
balanced_batch_sampler_val = DataManager.BalancedBatchSampler(dataset, n_classes, n_samples)
val_dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=balanced_batch_sampler_val,num_workers=int(opt.workers))

#get samples
data = iter(val_dataloader).next()

def split(data):
    """
    Split data into two subsets that have different class labels
    """
    labels = torch.unique(data[1]).tolist()
    label_to_indices = {label: torch.where(data[1] == label) for label in labels}
    Label, data_real_img = [], []
    for label_ in labels:
        label_indice = label_to_indices[label_]
        label_indice = label_indice[0]
        data_split = torch.index_select(data[0], 0, label_indice)
        label_split = torch.index_select(data[1], 0, label_indice)
        real_img = data_split
        data_real_img.append(real_img)  # stack image tensors
        Label.append(label_split)
    data1 = [data_real_img[0], Label[0]]
    data2 = [data_real_img[1], Label[1]]

    return data1, data2

data1, data2 = split(data)

# show generated images
combined_gan.save_img("eval", data1, data2, rec=True)

x_fake = combined_gan.gen(data1, data2)
y_fake = torch.ones(x_fake.size(0), dtype=torch.long, device=device) * -1
data_union = torch.concat([data[0], x_fake.detach()], 0)
label_union = torch.concat([data[1], y_fake], 0)

gt = label_union.tolist()
logs, feat = net(data_union)

# get deep features
ylist = label_union.to("cpu").detach().tolist()
y_s = list(set(ylist))
feature = {label: [] for label in range(10)}
feature[-1] = []
for i in range(len(label_union)):
    feature[ylist[i]].append(feat.to("cpu").detach().tolist()[i])  # feature wrt classes
features = feat.to("cpu").detach().tolist()

neg_feature = feature[-1]
neg_features = np.array(neg_feature)
pos_features = np.array(features[:len(features)-len(neg_feature)])
pos_gt = np.array(gt[:len(pos_features)])

# 2D visualization of deep features
filename = f"./output/sample" + '_{}.{}'
visualization.plotter_2D_sample(pos_features, pos_gt, neg_features=neg_features, file_name=str(filename), CIFAR10=opt.dataset == 'cifar10')



