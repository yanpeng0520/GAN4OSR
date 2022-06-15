"""
class Encoder,Generator, Discriminator are components of a small version of generative model for image input of 28*28
class Encoder_Larger, Generator_Larger, Discriminator_Larger are components of a large version of generative model for image input of 32*32
resnet18 function is used as 2D adaptive ResNet-18, named ResNet-18++
"""

import torch.nn as nn
import torchvision.models as models
import types
import torch

class Encoder(nn.Module):
    def __init__(self, nz):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            # input is 1 x 28 x 28
            nn.Conv2d(1, 32, 6, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 32 x 12 x 12
            nn.Conv2d(32, 128, 6, 2, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 4 x 4
            nn.Conv2d(128, nz, 4, 2, 0, bias=False),
            nn.BatchNorm2d(nz),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. 512 x 1 x 1
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Generator(nn.Module):
    def __init__(self, n_classes, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz * n_classes, 128, 4, 2, 0, bias=False),
            nn.BatchNorm2d(128), # number of feature
            nn.ReLU(True),
            # state size. 128 x 4 x 4
            nn.ConvTranspose2d(128, 32, 6, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size. 32 x 12 x 12
            nn.ConvTranspose2d(32, 1, 6, 2, 0, bias=False),
            nn.Tanh()
            # state size. 1 x 28 x 28
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, nz):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 1 x 28 x 28
            nn.Conv2d(1, 32, 6, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 32 x 12 x 12
            nn.Conv2d(32, 128, 6, 2, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, nz, 4, 2, 0, bias=False),
            nn.BatchNorm2d(nz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nz, 10, 1, 1, 0, bias=False),
        )

    def forward(self, input):
        output = [self.main(input).squeeze(), self.main[: -1](input).squeeze()]
        return output

class Encoder_Larger(nn.Module):
    def __init__(self, nz):
        super(Encoder_Larger, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 4 x 4
            nn.Conv2d(256, nz, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nz),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. nz x 1 x 1
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Generator_Larger(nn.Module):
    def __init__(self, n_classes, nz):
        super(Generator_Larger, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz * n_classes, 256, 4, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 8 x 8
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 16 x 16
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator_Larger(nn.Module):
    def __init__(self, nz):
        super(Discriminator_Larger, self).__init__()
        self.main = nn.Sequential(
            # input is 3 x 32 x 32
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 4 x 4
            nn.Conv2d(256, nz, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nz),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. nz x 1 x 1
            nn.Conv2d(nz, 10, 1, 1, 0, bias=False),
            # state size. nz x 1 x 1
        )

    def forward(self, input):
        output = [self.main(input).squeeze(), self.main[: -1](input).squeeze()]
        return output

# 2D Adaptation of LeNet-18
def custom_forward_impl(self, x) :

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    return self.fc(x), self.fc[0](x)

def custom_forward(self, x):
        return self._forward_impl(x)

def resnet18(pretrained=True, use_BG=False):
    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    if use_BG:
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 2), nn.Linear(2, 11, bias=False))
    else:
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 2), nn.Linear(2, 10, bias=False))
    model._forward_impl = types.MethodType(custom_forward_impl, model)
    model.forward = types.MethodType(custom_forward, model)
    return model

class LeNet_plus_plus(nn.Module):
    def __init__(self,  use_BG=False, num_classes=10):
        super(LeNet_plus_plus, self).__init__()

        # first convolution block
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=2)
        self.conv1_2 = nn.Conv2d(in_channels=self.conv1_1.out_channels, out_channels=32, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.batch_norm1 = nn.BatchNorm2d(self.conv1_2.out_channels, track_running_stats=False)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # second convolution block
        self.conv2_1 = nn.Conv2d(in_channels=self.conv1_2.out_channels, out_channels=64, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.conv2_2 = nn.Conv2d(in_channels=self.conv2_1.out_channels, out_channels=64, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.batch_norm2 = nn.BatchNorm2d(self.conv2_2.out_channels, track_running_stats=False)

        # third convolution block
        self.conv3_1 = nn.Conv2d(in_channels=self.conv2_2.out_channels, out_channels=128, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.conv3_2 = nn.Conv2d(in_channels=self.conv3_1.out_channels, out_channels=128, kernel_size=(5, 5), stride=1,
                                 padding=2)
        self.batch_norm3 = nn.BatchNorm2d(self.conv3_2.out_channels, track_running_stats=False)

        # fully-connected layers
        self.fc1 = nn.Linear(in_features=self.conv3_2.out_channels * 3 * 3,
                             out_features=2, bias=True)
        if use_BG:
            self.fc2 = nn.Linear(in_features=2, out_features=num_classes + 1, bias=False)
        else:
            self.fc2 = nn.Linear(in_features=2, out_features=num_classes, bias=False)
        #self.fc2 = nn.Linear(in_features=2, out_features=10, bias=True)
        # activation function
        self.prelu_act = nn.PReLU()

    def forward(self, x, features=False):
        # compute first convolution block output
        x = self.prelu_act(self.pool(self.batch_norm1(self.conv1_2(self.conv1_1(x)))))
        # compute second convolution block output
        x = self.prelu_act(self.pool(self.batch_norm2(self.conv2_2(self.conv2_1(x)))))
        # compute third convolution block output
        x = self.prelu_act(self.pool(self.batch_norm3(self.conv3_2(self.conv3_1(x)))))
        # turn into 1D representation (1D per batch element)
        x = x.view(-1, self.conv3_2.out_channels * 3 * 3)
        # first fully-connected layer to compute 2D feature space
        z = self.fc1(x)
        # second fully-connected layer to compute the logits
        y = self.fc2(z)
        if features:
            # return both the logits and the deep features
            return y
        else:
            return y, z

