
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import vast
from helpers import HiddenPrints
import random

class Dataset():
    def __init__(self, opt):
        """
        This class assemble different benchmark datasets to do Open-Set Recognition
        parameter opt: here we want to get opt.dataroot to save data and opt.dataset (dataset name). For example, if opt.dataset = "openset_mnist_emnist_kmnist", MNIST is used as KK, EMNIST is used as KU, and KMNIST is used as UU
        """
        self.dataroot = opt.dataroot
        self.dataset = opt.dataset

    def select_data(self, fake_method):
        """
        :param fake_method: method for OSR
        """

        # MNIST as KK
        if self.dataset == "openset_mnist_emnist":
            return self.openset_mnist_emnist(BG=fake_method == "BG")
        elif self.dataset == "openset_mnist_emnist_kmnist":
            return self.openset_mnist_emnist_kmnist(BG=fake_method == "BG")
        elif self.dataset == "openset_mnist_kmnist_emnist":
            return self.openset_mnist_kmnist_emnist(BG=fake_method == "BG")
        elif self.dataset == "openset_mnist_emnist_fashion":
            return self.openset_mnist_emnist_fashion(BG=fake_method == "BG")

       # KMNIST as KK, mnist as KU
        elif self.dataset == "openset_kmnist_mnist_emnist":
            return self.openset_kmnist_mnist_emnist(BG=fake_method == "BG")
        elif self.dataset == "openset_kmnist_mnist_fashion":
            return self.openset_kmnist_mnist_fashion(BG=fake_method == "BG")
        elif self.dataset == "openset_kmnist_mnist":
            return self.openset_kmnist_mnist_fashion(BG=fake_method == "BG")

       # fashion as KK, mnist as KU
        elif self.dataset == "openset_fashion_mnist_emnist":
            return self.openset_fashion_mnist_emnist(BG=fake_method == "BG")
        elif self.dataset == "openset_fashion_mnist_kmnist":
            return self.openset_fashion_mnist_kmnist(BG=fake_method == "BG")
        elif self.dataset == "openset_fashion_mnist":
            return self.openset_fashion_mnist(BG=fake_method == "BG")

        # cifar-10 as KK
        elif self.dataset == "cifar10":
            return datasets.CIFAR10
        elif self.dataset == "openset_cifar10_cifar100":
            return self.openset_cifar10_cifar100(BG=fake_method == "BG")
        elif self.dataset == "openset_cifar10_svhn":
            return self.openset_cifar10_svhn(BG=fake_method == "BG")

        # cifar-10 as KK
        elif self.dataset == "openset_svhn_cifar100":
            return self.openset_svhn_cifar100(BG=fake_method == "BG")
        elif self.dataset == "openset_svhn_cifar10":
            return self.openset_svhn_cifar10(BG=fake_method == "BG")

    def split_data(self, train_data, test_data, val_set_size = 20000):
        """
        further split training data into training and validation dataset
        :param val_set_size: validation size
        :return: training data, validation data, testing data
        """

        #val_set_size = 20000
        train_set_size = len(train_data) - val_set_size
        train_data, val_data = torch.utils.data.random_split(train_data, [train_set_size, val_set_size])

        return train_data, val_data, test_data

    def openset_mnist_emnist_simple(self):
        # training set
        digits_train = datasets.MNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )

        # testing set
        digits_test = datasets.MNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
        )
        letters_test = datasets.EMNIST(
            root=self.dataroot,
            split="letters",
            train=False,
            download=True,
            transform=ToTensor()
        )
        # no printing
        with HiddenPrints():
            train_data = vast.tools.ConcatDataset([digits_train])
            test_data = vast.tools.ConcatDataset([digits_test, letters_test])

        return self.split_data(train_data, test_data)

    def openset_mnist_emnist(self, BG=False):
        # training set
        digits_train = datasets.MNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )

        # testing set
        digits_test = datasets.MNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
        )

        # we use 1-13 letters as Known Unknown classes, and 14-26 letters as Unknown Unknown classes
        train_list = list(range(1,14))
        test_list = list(range(14,27))
        letters_train = SubClass_emnist(root=self.dataroot, split="letters", train=True, include_list=train_list, download=True, transform=ToTensor())
        letters_test = SubClass_emnist(root=self.dataroot, split="letters", train=False, include_list=test_list, download=True, transform=ToTensor())

        # no printing
        with HiddenPrints():
            train_data = vast.tools.ConcatDataset([digits_train, letters_train], BG)
            test_data = vast.tools.ConcatDataset([digits_test, letters_test], BG)

        return self.split_data(train_data, test_data, val_set_size=20400)

    def openset_mnist_emnist_kmnist(self, BG=False):
        # train set
        digits_train = datasets.MNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )
        letters_train = datasets.EMNIST(
            root=self.dataroot,
            split="letters",
            train=True,
            download=True,
            transform=ToTensor()
        )

        # testing set
        digits_test = datasets.MNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
        )

        characters_test = datasets.KMNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
        )

        # no printing
        with HiddenPrints():
            test_data = vast.tools.ConcatDataset([digits_test, characters_test], BG)
            train_val_data = vast.tools.ConcatDataset([digits_train, letters_train], BG)

        return self.split_data(train_val_data, test_data)

    def openset_mnist_kmnist_emnist(self, BG=False):
        # training set
        digits_train = datasets.MNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )
        characters_train = datasets.KMNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )

        # testing set
        digits_test = datasets.MNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
        )
        letters_test = datasets.EMNIST(
            root=self.dataroot,
            split="letters",
            train=False,
            download=True,
            transform=ToTensor()
        )

        # no printing
        with HiddenPrints():
            train_val_data = vast.tools.ConcatDataset([digits_train, characters_train], BG)
            test_data = vast.tools.ConcatDataset([digits_test, letters_test], BG)

        return self.split_data(train_val_data, test_data)

    def openset_kmnist_mnist_emnist(self, BG=False):
        # training set
        characters_train = datasets.KMNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )
        digits_train = datasets.MNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )

        # testing set
        characters_test = datasets.KMNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
        )

        letters_test = datasets.EMNIST(
            root=self.dataroot,
            split="letters",
            train=False,
            download=True,
            transform=ToTensor()
        )

        # no printing
        with HiddenPrints():
            train_val_data = vast.tools.ConcatDataset([characters_train, digits_train], BG)
            test_data = vast.tools.ConcatDataset([characters_test, letters_test], BG)

        return self.split_data(train_val_data, test_data)

    def openset_kmnist_mnist_fashion(self, BG=False):
        # training set
        characters_train = datasets.KMNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )
        digits_train = datasets.MNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )

        # testing set
        characters_test = datasets.KMNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
        )

        fashion_test = datasets.FashionMNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
        )

        # no printing
        with HiddenPrints():
            train_val_data = vast.tools.ConcatDataset([characters_train, digits_train], BG)
            test_data = vast.tools.ConcatDataset([characters_test, fashion_test], BG)

        return self.split_data(train_val_data, test_data)

    def openset_kmnist_mnist(self, BG=False):
        # training set
        characters_train = datasets.KMNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )
        digits_train = datasets.MNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )

        # testing set
        characters_test = datasets.KMNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
        )

        digits_test = datasets.MNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
        )

        # no printing
        with HiddenPrints():
            train_val_data = vast.tools.ConcatDataset([characters_train, digits_train], BG)
            test_data = vast.tools.ConcatDataset([characters_test, digits_test], BG)

        return self.split_data(train_val_data, test_data)

    def openset_mnist_emnist_fashion(self, BG=False):
        # train set
        digits_train = datasets.MNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )
        letters_train = datasets.EMNIST(
            root=self.dataroot,
            split="letters",
            train=True,
            download=True,
            transform=ToTensor()
        )

        # testing set
        digits_test = datasets.MNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
        )
        fashion_test = datasets.FashionMNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
            )

        # no printing
        with HiddenPrints():
            train_val_data = vast.tools.ConcatDataset([digits_train, letters_train], BG)
            test_data = vast.tools.ConcatDataset([digits_test, fashion_test], BG)

        return self.split_data(train_val_data, test_data)

    def openset_fashion_mnist_emnist(self, BG=False):
        # train set
        fashion_train = datasets.FashionMNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
            )
        digits_train = datasets.MNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )

        # testing set
        fashion_test = datasets.FashionMNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
            )
        letters_test = datasets.EMNIST(
            root=self.dataroot,
            split="letters",
            train=False,
            download=True,
            transform=ToTensor()
        )

        # no printing
        with HiddenPrints():
            train_val_data = vast.tools.ConcatDataset([fashion_train, digits_train], BG)
            test_data = vast.tools.ConcatDataset([fashion_test, letters_test], BG)

        return self.split_data(train_val_data, test_data)

    def openset_fashion_mnist_kmnist(self, BG=False):
        # train set
        fashion_train = datasets.FashionMNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
            )
        digits_train = datasets.MNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )

        # testing set
        fashion_test = datasets.FashionMNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
            )
        characters_test = datasets.KMNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
        )

        # no printing
        with HiddenPrints():
            train_val_data = vast.tools.ConcatDataset([fashion_train, digits_train], BG)
            test_data = vast.tools.ConcatDataset([fashion_test, characters_test], BG)

        return self.split_data(train_val_data, test_data)

    def openset_fashion_mnist(self, BG=False):
        # train set
        fashion_train = datasets.FashionMNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
            )
        digits_train = datasets.MNIST(
            root=self.dataroot,
            train=True,
            download=True,
            transform=ToTensor()
        )

        # testing set
        fashion_test = datasets.FashionMNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
            )
        digits_test = datasets.MNIST(
            root=self.dataroot,
            train=False,
            download=True,
            transform=ToTensor()
        )

        # no printing
        with HiddenPrints():
            train_val_data = vast.tools.ConcatDataset([fashion_train, digits_train], BG)
            test_data = vast.tools.ConcatDataset([fashion_test, digits_test], BG)

        return self.split_data(train_val_data, test_data)

    def openset_cifar10_cifar100(self, BG=False):
        # train set
        cifar10_train = datasets.CIFAR10(
            root=self.dataroot,
            train=True,
            download=True,
        transform = transforms.Compose([
            transforms.Resize(32),  # change to the default size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]))

        # randomly select 45 subclasses for training and validation, and select another 45 subclasses for testing
        size = 45
        train_list = random.sample(range(0, 100), size)
        exclude_list = list(set(set([*range(0, 100)]) - set(train_list)))
        test_list = random.sample(exclude_list, size)
        cifar100_train = SubClass_cifar10(root=self.dataroot, train=True, include_list=train_list, download=True, transform = transforms.Compose([
            transforms.Resize(32),  # change to the default size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]))

        cifar100_test = SubClass_cifar10(root=self.dataroot, train=False, include_list=test_list, download=True,  transform = transforms.Compose([
            transforms.Resize(32),  # change to the default size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]))

        # testing set
        cifar10_test = datasets.CIFAR10(
            root=self.dataroot,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),  # change to the default size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))

        # no printing
        with HiddenPrints():
            train_val_data = vast.tools.ConcatDataset([cifar10_train, cifar100_train], BG)
            test_data = vast.tools.ConcatDataset([cifar10_test, cifar100_test], BG)

        return self.split_data(train_val_data, test_data, val_set_size=13750)

    def openset_cifar10_svhn(self, BG=False):
        # train set
        svhn_train = datasets.SVHN(
            root=self.dataroot,
            split='train',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),  # change to the default size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))

        cifar10_train = datasets.CIFAR10(
            root=self.dataroot,
            train=True,
            download=True,
        transform = transforms.Compose([
            transforms.Resize(32),  # change to the default size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]))

        # testing set
        svhn_test = datasets.SVHN(
            root=self.dataroot,
            split='test',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),  # change to the default size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))

        cifar10_test = datasets.CIFAR10(
            root=self.dataroot,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),  # change to the default size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))

        # no printing
        with HiddenPrints():
            train_data = vast.tools.ConcatDataset([cifar10_train, svhn_train], BG)
            test_data = vast.tools.ConcatDataset([cifar10_test, svhn_test], BG)

        return self.split_data(train_data, test_data, val_set_size=12600)

    def openset_svhn_cifar100(self, BG=False):
        # train set
        svhn_train = datasets.SVHN(
            root=self.dataroot,
            split='train',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),  # change to the default size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))

        # randomly select 45 subclasses for training and validation, and select another 45 subclasses for testing
        size = 45
        train_list = random.sample(range(0, 100), size)
        exclude_list = list(set(set([*range(0, 100)]) - set(train_list)))
        test_list = random.sample(exclude_list, size)
        cifar100_train = SubClass_cifar10(root=self.dataroot, train=True, include_list=train_list, download=True, transform = transforms.Compose([
            transforms.Resize(32),  # change to the default size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]))

        cifar100_test = SubClass_cifar10(root=self.dataroot, train=False, include_list=test_list, download=True,  transform = transforms.Compose([
            transforms.Resize(32),  # change to the default size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]))

        # testing set
        svhn_test = datasets.SVHN(
            root=self.dataroot,
            split='test',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),  # change to the default size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))

        with HiddenPrints():
            train_val_data = vast.tools.ConcatDataset([svhn_train, cifar100_train], BG)
            test_data = vast.tools.ConcatDataset([svhn_test, cifar100_test], BG)

        return self.split_data(train_val_data, test_data, val_set_size=15950)

    def openset_svhn_cifar10(self, BG=False):
        # train set
        svhn_train = datasets.SVHN(
            root=self.dataroot,
            split='train',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),  # change to the default size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))

        cifar10_train = datasets.CIFAR10(
            root=self.dataroot,
            train=True,
            download=True,
        transform = transforms.Compose([
            transforms.Resize(32),  # change to the default size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]))

        # testing set
        svhn_test = datasets.SVHN(
            root=self.dataroot,
            split='test',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),  # change to the default size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))

        cifar10_test = datasets.CIFAR10(
            root=self.dataroot,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),  # change to the default size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))

        with HiddenPrints():
            train_data = vast.tools.ConcatDataset([svhn_train, cifar10_train])
            test_data = vast.tools.ConcatDataset([svhn_test, cifar10_test])

        return self.split_data(train_data, test_data)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches size: n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)  # all labels
        self.labels = torch.LongTensor(self.labels_list) # to tensor
        self.labels_set = list(set(self.labels.numpy())) # classes
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set} # label: index
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l]) # shuffle data of all labels
        self.used_label_indices_count = {label: 0 for label in self.labels_set} # label: 0
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False) # select n_classes from set without repetition
            indices = []
            for class_ in classes:  # select n_classes from classes
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples]) # indice[used indices : unused indices]
                self.used_label_indices_count[class_] += self.n_samples  # how many labels used for each class
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]): # some classes are used all, then shuffle the indices, and set used indice to 0
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size

class SubClass_emnist(datasets.EMNIST):
    def __init__(self, *args, include_list=[],  **kwargs):
        """
        This class gets a subclass of EMNIST.
        :param include_list: a list of classes provided
        """
        super(SubClass_emnist, self).__init__(*args, **kwargs)

        if include_list == []:
            return
        labels = np.array(self.targets)
        include = np.array(include_list).reshape(1, -1)
        mask = (labels.reshape(-1, 1) == include).any(axis=1)
        self.data = self.data[mask]
        self.targets = labels[mask].tolist()

class SubClass_cifar10(datasets.CIFAR10):
    def __init__(self, *args, include_list=[],  **kwargs):
        """
        This class gets a subclass of CIFAR-10.
        :param include_list: a list of classes provided
        """
        super(SubClass_cifar10, self).__init__(*args, **kwargs)

        if include_list == []:
            return
        labels = np.array(self.targets)
        include = np.array(include_list).reshape(1, -1)
        mask = (labels.reshape(-1, 1) == include).any(axis=1)
        self.data = self.data[mask]
        self.targets = labels[mask].tolist()
