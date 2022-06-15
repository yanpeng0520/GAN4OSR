
"""CombinedGAN: encoder, generator, and discriminator."""

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
import torchvision.utils as vutils
import architectures
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt

class combinedGAN():
    def __init__(self, opt):
        self.batchSize = opt.batchSize
        self.device = torch.device(f"cuda:{opt.gpu}" if opt.cuda else "cpu")
        self.l_d = opt.l_d
        self.l1 = opt.l1
        self.l2 = opt.l2
        self.n_classes = opt.n_classes
        self.gen_arch = opt.gen_arch
        self.outf = opt.outf
        self.nz = opt.nz
        self.wgan = opt.wgan

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            # for every Linear layer
            if classname.find('Linear') != -1:
                y = m.in_fertures
                # m.weight.data shoud be taken from a normal distribution
                m.weight.data.normal_(0.0, 1 / np.sqrt(y))
                # m.bias.data should be 0
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                torch.nn.init.normal_(m.weight, 1.0, 0.02)
                torch.nn.init.zeros_(m.bias)

        # initialize network
        if self.gen_arch =="small":
            self.netEnc = architectures.Encoder(self.nz)
        else:
            self.netEnc = architectures.Encoder_Larger(self.nz)
        self.netEnc = self.netEnc.to(self.device)
        self.netEnc.apply(weights_init)

        if self.gen_arch == "small":
            self.netG = architectures.Generator(self.n_classes, self.nz)
        else:
            self.netG = architectures.Generator_Larger(self.n_classes, self.nz)
        self.netG = self.netG.to(self.device)
        self.netG.apply(weights_init)

        if self.gen_arch == "small":
            self.netD = architectures.Discriminator(self.nz)
        else:
            self.netD = architectures.Discriminator_Larger(self.nz)
        self.netD = self.netD.to(self.device)
        self.netD.apply(weights_init)

        # if torch.cuda.device_count() > 1:
        #     self.netEnc = nn.DataParallel(self.netEnc)
        #     self.netG = nn.DataParallel(self.netG)
        #     self.netD = nn.DataParallel(self.netD)

        # setup optimizer
        self.optimizerEnc = optim.Adam(self.netEnc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # custom activation function
    def custom_activation(self, output):
        logexpsum = torch.sum(torch.exp(output), dim=1)
        result = logexpsum / (logexpsum + 1.0)
        return result

    def trainD(self, data, data_pair):
        """
        ErrD_real: cross entropy loss for original images
        ErrD_rec : cross entropy loss for reconstructed images
        errD_adv_real : adversarial loss for real images
        errD_adv_fake: adversarial loss for generated images
        errD_fake: cross entropy loss for the generated images, labeled as 1/n_classes wrt original classes
        """

        self.netD.zero_grad()

        data_real_img, Label, outputD_real, latent_space, features = self.split(data, data_pair)
        ErrD_real, ErrD_rec, f_real = [], [], []
        errD_adv_real, errD_adv_fake = [], []

        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.to(self.device)
                inter = a + alpha * (b - a)
                return inter

            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp

        for j in range(2):
            errD_real = nn.CrossEntropyLoss()(outputD_real[j], Label[j])
            ErrD_real.append(errD_real)

            # cross entropy loss for reconstructed images
            latent_space_rec = self.netEnc(data_real_img[j])
            rec_img = self.netG(latent_space_rec.repeat(1, self.n_classes, 1, 1))
            rec_output = self.netD(rec_img.detach())[0]
            errD_rec = nn.CrossEntropyLoss()(rec_output, Label[j])
            ErrD_rec.append(errD_rec)

            D_real = self.custom_activation(outputD_real[j])
            D_fake = self.custom_activation(rec_output)

            if self.wgan:
                wd = D_real.mean() - D_fake.mean()
                d_loss = -wd
                df_gp = gradient_penalty(self.netD, data_real_img[j], rec_img)
                (d_loss+df_gp).backward(retain_graph=True)

            else:
                # maximize log(D(x)) + log(1 - D(G(z))), label reconstrcuted images as fake(0), original images as real(1)
                # adversarial training with real, label as 1
                label = torch.full((data[0].size(dim=0),), 1, dtype=torch.float, device=self.device)
                nn.BCELoss()(D_real, label).backward(retain_graph=True)
                errD_adv_real.append(nn.BCELoss()(D_real, label))

                # adversarial training with reconstructed images, label as 0
                label.fill_(0)
                nn.BCELoss()(D_fake, label).backward(retain_graph=True)
                errD_adv_fake.append(nn.BCELoss()(D_fake, label))

        x_fake = self.netG(torch.cat(latent_space, 1))

        # cross entropy loss for the generated images, labeled as 1/n_classes wrt original classes
        Label = F.one_hot(torch.cat(Label, 0), num_classes=10).to(self.device)
        Label_fake = torch.zeros((data[1].size(0), 10), device=self.device)
        for i in range(data[0].size(dim=0)):
            label = torch.add(Label[i], Label[i+data[0].size(dim=0)])
            Label_fake[i] = label.float() / self.n_classes
        output_fake, feature_fake = self.netD(x_fake.detach())
        errD_fake = nn.CrossEntropyLoss()(output_fake, Label_fake)
        errD_fake = errD_fake.mean()

        errD = torch.mean(torch.stack(ErrD_real)) + torch.mean(torch.stack(ErrD_rec)) + \
               self.l_d * errD_fake # l_d should be larger
        errD.backward()
        self.optimizerD.step()
        return ErrD_real, errD_fake

    def trainG(self, data, data_pair):

        """
        errG_fake: cross entropy loss for the generated images, labeled as 1/n_classes wrt original classes
        errG_adv: adversarial loss for generated images
        errCon: construction loss between genreated iamges and original images
        """
        self.netEnc.zero_grad()
        self.netG.zero_grad()

        data_real_img, Label, _, latent_space, _ = self.split(data, data_pair)

        fake = self.netG(torch.cat(latent_space, 1))
        outputG = self.netD(fake)[0]
        Label = F.one_hot(torch.cat(Label, 0), num_classes=10).to(self.device)
        Label_fake = torch.zeros((data[1].size(0), 10), device=self.device)
        for i in range(data[0].size(dim=0)):
            label = torch.add(Label[i], Label[i+data[0].size(dim=0)])
            Label_fake[i] = label.float() / self.n_classes # fake label is the mean of label

        errG_fake = nn.CrossEntropyLoss()(outputG, Label_fake)

        recon_imgs, errG_adv = [], []
        errCons = 0
        for i in range(self.n_classes):
            recon_img = self.netG(torch.cat([latent_space[i]] * self.n_classes, 1))
            recon_imgs.append(recon_img)
            errCons += F.l1_loss(recon_img, data_real_img[i])
            output = self.netD(recon_img)[0]  # The detach() method constructs a new view on a tensor which is declared not to need gradients
            D_fake = self.custom_activation(output)

            if self.wgan:
                g_loss = -D_fake.mean()
                g_loss.backward(retain_graph=True) # update
            else:
                # maximize log(D(G(z)))
                label = torch.full((data[0].size(dim=0),), 1, dtype=torch.float, device=self.device)
                nn.BCELoss()(D_fake, label).backward(retain_graph=True)
                nn.BCELoss()(D_fake, label)
                errG_adv.append(nn.BCELoss()(D_fake, label))

        errG = self.l1 * errCons + self.l2 * errG_fake
        errG.backward() # second update
        self.optimizerG.step()
        self.optimizerEnc.step()
        return errG_fake, errCons

    def split(self, data, data_pair):
        data_real_img = [data[0], data_pair[0]]
        Label = [data[1], data_pair[1]]
        outputD_real = [self.netD(data[0])[0], self.netD(data_pair[0])[0]]
        features = [self.netD(data[0])[1], self.netD(data_pair[0])[1]]
        latent_space = [self.netEnc(data[0]), self.netEnc(data_pair[0])]
        return data_real_img, Label, outputD_real, latent_space, features

    def train(self):
        self.netEnc.train()
        self.netG.train()
        self.netD.train()

    def eval(self):
        self.netEnc.eval()
        self.netG.eval()
        self.netD.eval()

    def gen(self, data, data_pair):
        data_real_img, Label, outputD_real, latent_space, features = self.split(data, data_pair)
        return self.netG(torch.cat(latent_space, 1))

    def save_img(self, s, data, data_pair, epoch=0, size=16, fake_adjust=None, rec=True):
        """Save images from data pair: originial images, fake images, reconstructed images
        Parameters:
        size: image tensor size
        fake_adjust: fake images after adjust
        rec: Set this to True when the reconstructed images are needed to display
        """
        rec_fake = []
        data_real_img, Label, outputD_real, latent_space, features = self.split(data, data_pair)
        for j in range(self.n_classes):
            latent_space_rec = self.netEnc(data_real_img[j])  # check the images of one class
            rec_fake.append(self.netG(latent_space_rec.repeat(1, self.n_classes, 1, 1)).detach())

        fake = self.netG(torch.cat(latent_space, 1)).to(self.device)

        if fake_adjust == None:
            if rec != True:
                assembled_image = torch.cat(
                    [data[0][:size].to(self.device), data_pair[0][:size], fake[:size]], 0)
            else:
                assembled_image = torch.cat([data[0][:size].to(self.device), data_pair[0][:size], fake[:size], rec_fake[0][:size],
                 rec_fake[1][:size]], 0)
        else:
            assembled_image = torch.cat([data[0][:size].to(self.device), data_pair[0][:size], fake[:size], fake_adjust[:size], torch.cat(rec_fake, 0)[:size]], 0)

        # image tensor layout: [reconstructed image 1, original image 1, generated image, original image 2, reconstructed image 2]
        tensor_row = size // 8
        assenbled_image_row = assembled_image.shape[0] // 8
        assembled_image_new = torch.zeros_like(assembled_image)
        for i in range(tensor_row):
            tensor_i = i * (assenbled_image_row // tensor_row)
            if rec != True:
                assembled_image_new[(tensor_i) * 8: (tensor_i) * 8 + 8] = assembled_image[8 * i: 8 * i + 8]
                assembled_image_new[(tensor_i + 1) * 8: (tensor_i + 1) * 8 + 8] = assembled_image[8 * (i + tensor_row * 2): 8 * ( i + tensor_row * 2) + 8]
                assembled_image_new[(tensor_i + 2) * 8: (tensor_i + 2) * 8 + 8] = assembled_image[8 * (i + tensor_row): 8 * (i + tensor_row) + 8]
            else:
                assembled_image_new[(tensor_i) * 8: (tensor_i) * 8 + 8] = assembled_image[8 * (i + tensor_row * 3): 8 * (i + tensor_row * 3) + 8]
                assembled_image_new[(tensor_i + 1) * 8: (tensor_i + 1) * 8 + 8] = assembled_image[8 * i: 8 * i + 8]
                assembled_image_new[(tensor_i + 2) * 8: (tensor_i + 2) * 8 + 8] = assembled_image[8 * (i + tensor_row * 2): 8 * (i + tensor_row * 2) + 8]
                assembled_image_new[(tensor_i + 3) * 8: (tensor_i + 3) * 8 + 8] = assembled_image[8 * (i + tensor_row): 8 * (i + tensor_row) + 8]
                assembled_image_new[(tensor_i + 4) * 8: (tensor_i + 4) * 8 + 8] = assembled_image[8 * (i + tensor_row * 4): 8 * (i + tensor_row * 4) + 8]

        def imshow(inp):
            """Imshow for Tensor."""
            inp = inp.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imsave('%s/assembled_%s_samples_iter_%03d.png' % (self.outf, s, epoch), inp)
        if data[0].shape[1] != 1:
            assembled_image_new = torchvision.utils.make_grid(assembled_image_new.detach().cpu())
            imshow(assembled_image_new)
        else:
            vutils.save_image(assembled_image_new.detach(),
                          '%s/assembled_%s_samples_iter_%03d.png' % (self.outf, s, epoch))
