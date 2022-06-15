
import vast
import torch
import numpy
import pandas as pd
import json
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from vast.tools import viz
import numpy as np
import DataManager
import visualization
import architectures
import eval_metrics

class Evaluate():
    def __init__(self, opt, labels):
        self.device = torch.device(f"cuda:{opt.gpu}" if opt.cuda else "cpu")
        self.data_test = opt.dataset
        self.data_manager = DataManager.Dataset(opt)
        self.labels = labels
        self.outf = opt.outf
        # networks
        self.networks = {
            which: self.load_network(which) for which in list(self.labels.keys())
        }
    def load_network(self, which):
        if opt.arch == 'LeNet_plus_plus':
            net = architectures.LeNet_plus_plus(use_BG=which == "BG")
        elif opt.arch == 'Resnet18':
             net = architectures.resnet18(pretrained=False, use_BG=which == "BG")

        net.load_state_dict(torch.load(f"./model/{opt.arch}_{which}.pth"))
        net.to(self.device)
        return net

    def extract(self, dataset, which, net):
        """
        This function extracts class labels, logits, deep features, and evaluation results
        :param dataset: which dataset
        :param which: which approach
        :param net: which open-set network
        """
        gt, logits = [], []
        if which == 'BG':
            feature = {label: [] for label in range(11)}
        else:
            feature = {label: [] for label in range(10)}
        feature[-1] = []

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)
        features =[]

        val_accuracy = torch.zeros(2, dtype=int)
        val_confidence = torch.zeros(2, dtype=float)
        roc_y = torch.tensor([], dtype=torch.long).to(self.device)
        roc_pred = torch.tensor([], dtype=torch.long).to(self.device)

        net.eval()
        with torch.no_grad():
            for (x, y) in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                gt.extend(y.tolist())
                logs, feat = net(x)
                logits.extend(logs.tolist())

                # create feature list for each class
                ylist = y.to("cpu").detach().tolist()
                for i in range(len(y)):
                    feature[ylist[i]].append(feat.to("cpu").detach().tolist()[i]) #feature wrt classes
                features += feat.to("cpu").detach().tolist()  #all features

                val_accuracy += vast.losses.accuracy(logs, y)
                val_confidence += vast.losses.confidence(logs, y)

                # compute roc and auoc
                roc_y = torch.cat((roc_y, y.detach()))
                roc_pred = torch.cat((roc_pred, torch.nn.functional.softmax(logs, dim=1).detach()))
            auc = eval_metrics.roc_auc(roc_pred.to("cpu").detach(), roc_y.to("cpu").detach())
            ccr, fprt, auoc = eval_metrics.auoc(roc_pred.to("cpu").detach().numpy(), roc_y.to("cpu").detach().numpy(),
                                           BG=which == 'BG')
        Val_accuracy = float(val_accuracy[0]) / float(val_accuracy[1])
        Val_confidence = (val_confidence[0] / val_confidence[1]).item()
        print(
        f"accuracy {Val_accuracy:.4f} "
        f"confidence {Val_confidence:.4f} "
        f"auc {auc:.4f}"
        f"auoc {auoc:.4f}"
        )

        metrics = [round(Val_accuracy, 4), round(Val_confidence, 4), round(auc, 4), round(auoc, 4)]
        return numpy.array(gt), numpy.array(logits), feature, features, metrics  # orginal label and logit

    def evaluate(self, test_data_list, ku):
        """
        evaluate on different testing dataset
        :param test_data_list: testing dataset list
        :param ku: BG dataset
        """

        results = {}
        for test_data_name in test_data_list:
            metrics = []
            for which, net in self.networks.items():
                print("Evaluating", which)
                train_data, val_data, test_data = self.data_manager.select_data(which, test_data_name)
                print("Evaluating", test_data_name)
                output = self.extract(test_data, which, net)
                metrics.append(output[-1])
                self.plot(output, test_data_name, which) # plot 2D visualization and histogram
                if which == "SoftMax":
                    self.plot(output, test_data_name, which)
                    self.plot(output, test_data_name, "Original", neg_features_indicator=False) # plot only positive samples

                # extract ccr and fpr
                test_gt, test_predicted = output[:2]
                test_predicted = torch.nn.functional.softmax(torch.tensor(test_predicted), dim=1).detach().numpy()
                results[which] = eval_metrics.auoc(test_predicted, test_gt, BG=which=="BG")[:-1]

            # dump results
            with open(f"./figures/oscr_{test_data_name}.txt", 'w') as f:
                f.write(json.dumps(results))

            # load results and display barchart
            visualization.bar_oscr(f"./figures/oscr_{test_data_name}.txt", ku, test_data_name)
            pdf = PdfPages(f"./figures/{self.outf}_{str(test_data_name)}_OSCR.pdf")
            try:
                # plot with unknown unknowns
                pyplot.figure()
                for which, res in results.items():
                  pyplot.semilogx(res[1], res[0], label=self.labels[which])
                pyplot.legend()
                pyplot.xlabel("False Positive Rate", fontsize=12)
                pyplot.ylabel("Correct Classification Rate", fontsize=12)
                pyplot.tight_layout()
                pdf.savefig(bbox_inches='tight', pad_inches=0)
            finally:
                pdf.close()

            dict = {key : metrics[i] for i, key in enumerate(self.labels.keys())}
            data = pd.DataFrame.from_dict(dict)
            data.set_axis(['Accuracy', 'Confidence', 'AUC', 'AUOC'])
            data.to_csv(f"./figures/{self.outf}_{str(test_data_name)}_results.csv")
            visualization.bar_metric(dict, ku, test_data_name)

    def plot(self, ot, data_item, which, neg_features_indicator=True):
        """
        plot 2D visualization and histogram
        :param ot: output of deep feature extraction
        :param data_item: testing dataset
        :param which: which approach
        :param neg_features_indicator: Set True if we need negetive features
        """
        test_gt, test_predicted, feature, features,_ = ot

        neg_feature = feature[10] if which == 'BG' else feature[-1]
        pos_features = np.array(features[:len(features)-len(neg_feature)])

        pos_gt = test_gt[:len(pos_features)]
        neg_features = np.array(neg_feature)

        viz.plot_histogram(pos_features, neg_features, file_name="figures/{}_"+which+'_'+self.outf+'_'+data_item+".pdf")

        filename = f"./figures/{self.outf}_{which}_"+data_item + '_{}.{}'
        visualization.plotter_2D(pos_features,
                                 pos_gt,
                                 neg_features=neg_features if neg_features_indicator==True else None,
                                 file_name=str(filename),
                                 final=False,
                                 )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='openset_mnist_emnist_kmnist')
    parser.add_argument('--dataroot', required=False, default='/tmp', help='path to dataset')
    parser.add_argument('--gpu', default=3, type=int)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument("--arch", default='LeNet_plus_plus', choices=['Resnet18', 'LeNet_plus_plus'])
    parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
    opt = parser.parse_args()

    ku = 'EMNIST'
    labels = {
        "SoftMax": "SM",
        "BG": f"SM + BG({ku})",
        "Cross_KU": f"EOS + BG({ku})",
        "Cross_noise": "EOS + Noise",
        "Cross_adv": "EOS + FGSM",
        "Cross_gan": "EOS + Combined-GAN"
    }

    test_data_list = ['openset_mnist_emnist', 'openset_mnist_emnist_kmnist', 'openset_mnist_emnist_fashion']
    Evaluate(opt, labels).evaluate(test_data_list, ku)


