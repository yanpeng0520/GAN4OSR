from __future__ import print_function
import random
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from datetime import datetime

import vast
import helpers
import combinedGAN
import architectures
import DataManager
import attack
from loss import entropic_openset_loss
import check
import preprocess
import eval_metrics

# input arguments
opt = check.command_line_options()
print(opt)
print(opt.fake_method)
check.check(opt)

device = torch.device(f"cuda:{opt.gpu}" if opt.cuda else "cpu")
save_C_dir = f"./model/{opt.arch}_{opt.fake_method}.pth"
save_enc_dir = f"./model/Enc.pth"
save_G_dir = f"./model/G.pth"

if opt.fake_method in ["SoftMax", "BG"]:
    loss_func = nn.CrossEntropyLoss(reduction='mean')
else:
    loss_func = entropic_openset_loss(device)

# obtain dataset
data_manager = DataManager.Dataset(opt)
train_data, val_data, test_data = data_manager.select_data(opt.fake_method)

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opt.batchSize,
    shuffle=True,
    pin_memory=True)

val_dataloader = torch.utils.data.DataLoader(
    val_data,
    batch_size=opt.batchSize,
    shuffle=True,
    pin_memory=True)

# choose model depends on dataset
if opt.dataset in ['cifar10', 'openset_cifar10_cifar100', 'openset_svhn_cifar100']:
    model = architectures.resnet18(pretrained=True, use_BG=opt.fake_method == "BG")
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    opt.arch = 'Resnet18'
    opt.gen_arch = 'large'
else:
    model = architectures.LeNet_plus_plus(use_BG=opt.fake_method == "BG").to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    opt.arch = 'LeNet_plus_plus'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

combined_gan = combinedGAN.combinedGAN(opt)
print("Trainable params of Encoder", count_parameters(combined_gan.netEnc))
print("Trainable params of G", count_parameters(combined_gan.netG))
print("Trainable params of D", count_parameters(combined_gan.netD))

torch.autograd.set_detect_anomaly(True)

loss_D_real_history, loss_D_fake_history = [], []
loss_G_history, loss_cons_history = [], []
accuracy_real_history, accuracy_fake_history, loss_val_history = [], [], []

progressbar = helpers.Progressbar()
fake_method = opt.fake_method
filter = opt.filter

# training process
start_time= datetime.now()
for epoch in range(opt.niter):
    loss_history = []
    train_loss = torch.zeros(2, dtype=float)
    train_accuracy = torch.zeros(2, dtype=int)
    train_magnitude = torch.zeros(2, dtype=float)
    train_confidence = torch.zeros(2, dtype=float)
    fake_num = 0
    filtered_num = 0
    model.train()

    for i, data in progressbar(enumerate(train_dataloader, 0)): # index from 0
        optimizer.zero_grad()
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)

        # remove unknown
        if fake_method not in ['Cross_KU', ]:
            data[0], data[1] = preprocess.remove_unknown(data[0], data[1])

        # forward
        logit, feat = model(data[0])
        loss = loss_func(logit, data[1])

        # backpropagation
        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()

        feat = feat.detach()
        X, y = data[0], data[1]

        # only generate from known
        if opt.remove_unknown == 'True':
           X, y = preprocess.remove_unknown(X, y) #if opt.remove_not_mnist == 'True' else X, y

        logit, feat = model(data[0])

        if filter == "True":
            X, y, y_old = preprocess.filter_correct(X, y, logit)
            X, y, y_old = preprocess.filter_threshold(X, y, logit)  # filter samples with p over 0.9

        fake_adjust = None
        if len(X) > 1 and fake_method not in ['SoftMax', 'Cross_KU', 'BG']:

            data_filtered = [X, y] # filtered data
            filtered_num += len(X)
            if i == len(train_dataloader) - 1:
                print(filtered_num)

                # random fake image
            if fake_method == 'Cross_noise':
                 x_fake = (torch.rand(data[0].shape))
                 x_fake = x_fake.to(device)
            elif fake_method == 'Cross_noise_img':
               x_fake = attack.Attack().random_perturbation(X, device)
            elif fake_method == 'Cross_adv': #fgsm attack
               x_fake = attack.Attack().fgsm_attack(X, y, model, loss_func, epsilon=0.1)

            #generating samples using GAN
            elif fake_method == 'Cross_gan':
                # data pair for original data ( different class labels)
                data_pair = torch.zeros(X.shape, device=device)
                label_pair = torch.zeros(y.shape, dtype=y.dtype, device=device)
                labels = y.tolist()
                for k in range(X.size(dim=0)):
                    index_include = [index for index in range(len(labels)) if labels[index] == labels[k]]
                    index_exclude = [index for index in range(len(labels)) if index not in index_include]
                    if index_exclude != []:
                       index = random.choice(index_exclude)
                       data_pair[k] = X[index]
                       label_pair[k] = y[index]

                data_pair = [data_pair, label_pair]

                # Update G network
                errG_fake, errCons = combined_gan.trainG(data_filtered, data_pair)  #train G
                loss_G_history.append(errG_fake.item())
                loss_cons_history.append(errCons.item())

                if i % 100 == 0:
                    combined_gan.save_img("training", data_filtered, data_pair, epoch)

                # Update D network
                data_rep = data_filtered # remove later
                errD_real, errD_fake = combined_gan.trainD(data_rep, data_pair)

                x_fake = combined_gan.gen(data_filtered, data_pair)

            y_fake = torch.ones(x_fake.size(0), dtype=torch.long, device=device) * -1
            data_union = torch.concat([data[0], x_fake.detach()], 0)
            label_union = torch.concat([data[1], y_fake], 0)

            pred = model(x_fake)
            loss = loss_func(pred, y_fake)
            loss.mean().backward()
            optimizer.step()

        else:
            data_union = data[0]
            label_union = data[1]

        # training loss
        logit, features = model(data_union)
        loss_OSR = loss_func(logit, label_union)
        train_loss += torch.tensor((loss_OSR, 1))

        # compute accuracy and confidence
        train_accuracy += vast.losses.accuracy(logit, label_union)
        train_confidence += vast.losses.confidence(logit, label_union)

        # compute magnitude of deep feature
        if opt.fake_method not in ("SoftMax", "BG"):
            train_magnitude += vast.losses.sphere(features, label_union)

    # validation process
    with torch.no_grad():
        model.eval()
        val_accuracy = torch.zeros(2, dtype=int)
        accuracy_fake = torch.zeros(2, dtype=int)
        val_magnitude = torch.zeros(2, dtype=float)
        val_confidence = torch.zeros(2, dtype=float)
        roc_y = torch.tensor([], dtype=torch.long).to(device)
        roc_pred = torch.tensor([], dtype=torch.long).to(device)
        val_loss = 0

        for X, y in val_dataloader:
            X = X.to(device)
            y = y.to(device)
            if fake_method in ['SoftMax', ]:
                X, y = preprocess.remove_unknown(X, y)
            outputs, features = model(X)
            loss = loss_func(outputs, y)
            val_loss += loss.item()
            val_accuracy += vast.losses.accuracy(outputs, y)
            val_confidence += vast.losses.confidence(outputs, y)

            if opt.fake_method not in ("SoftMax", "BG"):
                val_magnitude += vast.losses.sphere(features, y)

            # for roc
            roc_y = torch.cat((roc_y, y.detach()))
            roc_pred = torch.cat((roc_pred, torch.nn.functional.softmax(outputs, dim=1).detach()))
        n_batches = len(val_dataloader)
        val_loss /= n_batches

        if fake_method != 'SoftMax':
           auc_score = eval_metrics.roc_auc(roc_pred.to("cpu").detach(), roc_y.to("cpu").detach())
           ccr, fprt, auoc = eval_metrics.auoc(roc_pred.to("cpu").detach().numpy(), roc_y.to("cpu").detach().numpy(), BG=fake_method == 'BG')
        else:
            auc_score = 0; auoc = 0

    # save network based on confidence metric of validation set
    save_status = "NO"
    if epoch <= 1:
       prev_save_score = None
    save_score = auoc if fake_method != 'SoftMax' else val_confidence[0]
    if prev_save_score is None or (save_score > prev_save_score):
        prev_save_score = save_score
        save_status = "YES"
        torch.save(model.state_dict(), save_C_dir)
        torch.save(combined_gan.netEnc.state_dict(), save_enc_dir)
        torch.save(combined_gan.netG.state_dict(), save_G_dir)

    # print some statistics
    print(f"[{epoch}/{opt.niter}]"
          f"train loss {float(train_loss[0]) / float(train_loss[1]):.2f} "
          f"total accuracy {float(train_accuracy[0]) / float(train_accuracy[1]):.2f} "
          f"confidence {train_confidence[0] / train_confidence[1]:.2f} "
          f"magnitude {train_magnitude[0] / train_magnitude[1] if train_magnitude[1] else -1:.2f} -- "
          f"|| "
          f"val loss {float(val_loss):.2f}"
          f"accuracy {float(val_accuracy[0]) / float(val_accuracy[1]):.2f} "
          f"confidence {val_confidence[0] / val_confidence[1]:.2f} "
          f"auc_score {auc_score:.2f} "
          f"auoc_score {auoc:.2f} "
          f"magnitude {val_magnitude[0] / val_magnitude[1] if val_magnitude[1] else -1:.2f} -- "
          f"Saving Model {save_status}")

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))