
""" For 2D visualization and plotting bar chart"""

import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json

def bar_oscr(dir, ku, dataset):
    """
    This function is used to create bar chart
    :param dir: location for saved data and evaluation results
    :param ku: name of known unknown class
    :param dataset: testing dataset
    """

    with open(dir, 'r') as f:
        oscr_data = json.loads(f.read())

    # get ccr regarding to fpr
    if (dataset == 'openset_cifar10_cifar100') | (dataset == 'openset_svhn_cifar100'):
        thres = [10 ** -3, 10 ** -2, 10 ** -1, 1]
    else:
        thres = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1]

    ccr_t = {}
    for key, value in oscr_data.items():
        ccr = []
        for t in thres:
            fpr = min(value[1], key=lambda x: abs(x - t))
            index = value[1].index(fpr)
            ccr.append(value[0][index])
        ccr_t[key] = ccr

    # set width of bar
    barWidth = 0.1

    # Set position of bar on X axis
    if (dataset == 'openset_cifar10_cifar100') | (dataset == 'openset_svhn_cifar100'):
        br1 = np.arange(4)
    else:
        br1 = np.arange(5)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]

    pdf = PdfPages(f"./figures/{dataset}_oscr.pdf")
    plt.figure()
    br = [br1, br2, br3, br4, br5, br6]
    label = ['SM', f'SM + BG({ku})', f'EOS + BG({ku})',
             'EOS + Noise', 'EOS + FGSM',
             'EOS + Combined-GAN']
    for i, key in enumerate(ccr_t.keys()):
        plt.bar(br[i], ccr_t[key], width=barWidth,
                edgecolor='grey', label=label[i])

    # Adding Xticks
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('Correct Classification Rate', fontsize=12)

    if ((dataset == 'openset_cifar10_cifar100') | (dataset == 'openset_svhn_cifar100')):
        plt.xticks([r + barWidth for r in range(4)], ['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1'])
    else:
        plt.xticks([r + barWidth for r in range(5)], ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1'])

    plt.ylim((0, 1.1))
    plt.legend(loc='upper center', ncol=len(label) // 3, bbox_to_anchor=(0.5, 1.2),
               fancybox=True, shadow=True)
    # plt.show()
    plt.savefig(f"./figures/{dataset}_oscr.pdf", format="pdf", bbox_inches="tight")

def bar_metric(dict, ku, dataset):
    """
    This function is used to create bar chart
    :param dict: dictionary of evaluation results
    :param ku: name of known unknown class
    :param dataset: testing dataset
    """

    # set width of bar
    barWidth = 0.1
    # Set position of bar on X axis
    br1 = np.arange(3)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]

    plt.figure()
    br = [br1, br2, br3, br4, br5, br6]
    label = ['SM', f'SM + BG({ku})', f'EOS + BG({ku})',
             'EOS + Noise', 'EOS + FGSM',
             'EOS + Combined-GAN']
    for i, key in enumerate(dict.keys()):
        plt.bar(br[i], dict[key][1:], width=barWidth, edgecolor='grey', label=label[i])

    # Adding Xticks
    plt.xlabel('Evaluation Metrics', fontsize=12)
    plt.ylabel('Evaluation Values', fontsize=12)
    plt.xticks([r + barWidth for r in range(3)], ['Confidence', 'AUC', 'AUOC'])
    plt.ylim((0, 1.1))
    # plt.legend()
    plt.legend(loc='upper center', ncol=len(label) // 3, bbox_to_anchor=(0.5, 1.2),
               fancybox=True, shadow=True)
    # plt.show()
    plt.savefig(f"./figures/{dataset}_metric.pdf", format="pdf", bbox_inches="tight")
    # plt.legend()

"""
The following 2D visualization implementation partly taken from https://github.com/Vastlab/vast
"""

colors_global = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [67, 99, 216],
        [245, 130, 49],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [188, 246, 12],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [154, 99, 36],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 216, 177],
        [0, 0, 117],
        [128, 128, 128],
        [255, 255, 255],
        [0, 0, 0],
    ]
).astype(np.float)
colors_global = colors_global / 255.0


def plotter_2D_sample(
    pos_features,
    labels,
    neg_features=None,
    file_name="foo.pdf",
    CIFAR10=False
):

    random_index = random.sample(range(colors_global.shape[0]), 2)
    label = np.array(list(set(labels)))
    if CIFAR10 == True:
        fig_label = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        fig_label = label

    cdict = {label[0]: colors_global[random_index[0], :].tolist(), label[1]: colors_global[random_index[1], :].tolist()}
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(2):
        ix = np.where(labels == label[i])
        label_ = fig_label[label[i]] if CIFAR10 == True else fig_label[i]
        ax.scatter(
            pos_features[ix, 0],
            pos_features[ix, 1],
            c=cdict[label[i]],
            label=label_,
            s=20,
            marker="o",
        )

    if neg_features is not None:
        ax.scatter(
            neg_features[:, 0],
            neg_features[:, 1],
            c="k",
            s=20,
            marker="x",
            label=f"{fig_label[label[0]]}+{fig_label[label[1]]}" if CIFAR10 == True else f"{fig_label[0]}+{fig_label[1]}"
        )
    ax.legend(loc='upper right')
    # ax.set_xlim([-10, 10])
    # ax.set_ylim([-10, 10])
    ax.axis("equal")
    fig.savefig(file_name.format('2D_plot', 'png'), bbox_inches='tight')

def plotter_2D(
    pos_features,
    labels,
    neg_features=None,
    file_name="foo.pdf",
    final=False,
):
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = colors_global
    if neg_features is not None:
        # Remove black color from knowns
        colors = colors_global[:-1, :]

    colors_with_repetition = colors.tolist()
    for i in range(int(len(set(labels.tolist())) / colors.shape[0])):
        colors_with_repetition.extend(colors.tolist())
    colors_with_repetition.extend(
        colors.tolist()[: int(colors.shape[0] % len(set(labels.tolist())))]
    )
    colors_with_repetition = np.array(colors_with_repetition)

    labels_to_int = np.zeros(labels.shape[0])
    for i, l in enumerate(set(labels.tolist())):
        labels_to_int[labels == l] = i

    ax.scatter(
        pos_features[:, 0],
        pos_features[:, 1],
        c=colors_with_repetition[labels_to_int.astype(np.int)],
        edgecolors="none",
        s=1,
    )
    if neg_features is not None:
        ax.scatter(
            neg_features[:, 0],
            neg_features[:, 1],
            c="k",
            edgecolors="none",
            s=5,
            marker="*",
        )
    if final:
        fig.gca().spines["right"].set_position("zero")
        fig.gca().spines["bottom"].set_position("zero")
        fig.gca().spines["left"].set_visible(False)
        fig.gca().spines["top"].set_visible(False)
        ax.tick_params(
            axis="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        ax.axis("equal")
    fig.savefig(file_name.format('2D_plot', 'png'), bbox_inches='tight')

