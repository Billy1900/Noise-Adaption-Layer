import torch
import numpy as np
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from data.data_utils import dataset_split
from parse_config import args


class mnist_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.5, split_per=0.9, random_seed=1,
                 num_class=10):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        original_images = np.load('data/mnist/train_images.npy')
        original_labels = np.load('data/mnist/train_labels.npy')
        self.train_data, self.val_data, self.train_labels, self.val_labels = dataset_split(original_images,
                                                                                           original_labels,
                                                                                           noise_rate, split_per,
                                                                                           random_seed, num_class)
        # print(self.val_labels)
        pass

    def __getitem__(self, index):

        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class mnist_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.test_data = np.load('data/mnist/test_images.npy')
        self.test_labels = np.load('data/mnist/test_labels.npy') - 1  # 0-9

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)


class cifar10_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.0, split_per=0.9, random_seed=1,
                 num_class=10):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        original_images = np.load('data/cifar10/train_images.npy')
        original_labels = np.load('data/cifar10/train_labels.npy')
        self.train_data, self.val_data, self.train_labels, self.val_labels = dataset_split(original_images,
                                                                                           original_labels,
                                                                                           noise_rate, split_per,
                                                                                           random_seed, num_class)
        if self.train:
            self.train_data = self.train_data.reshape((45000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

        else:
            self.val_data = self.val_data.reshape((5000, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        if self.train:
            img, label = self.train_data[index], self.train_labels[index]

        else:
            img, label = self.val_data[index], self.val_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class cifar10_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.test_data = np.load('data/cifar10/test_images.npy')
        self.test_labels = np.load('data/cifar10/test_labels.npy')
        self.test_data = self.test_data.reshape((10000, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)


class cifar100_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.0, split_per=0.9, random_seed=1,
                 num_class=100):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        original_images = np.load('data/cifar100/train_images.npy')
        original_labels = np.load('data/cifar100/train_labels.npy')
        self.train_data, self.val_data, self.train_labels, self.val_labels = dataset_split(original_images,
                                                                                           original_labels,
                                                                                           noise_rate, split_per,
                                                                                           random_seed, num_class)
        if self.train:
            self.train_data = self.train_data.reshape((45000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

        else:
            self.val_data = self.val_data.reshape((5000, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        if self.train:
            img, label = self.train_data[index], self.train_labels[index]

        else:
            img, label = self.val_data[index], self.val_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class cifar100_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.test_data = np.load('data/cifar100/test_images.npy')
        self.test_labels = np.load('data/cifar100/test_labels.npy')
        self.test_data = self.test_data.reshape((10000, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)


def transform_train(dataset_name):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transform


def transform_test(dataset_name):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transform


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


def DataLoad_MNIST_CIFAR(DatasetName, noise_rate=0.0):
    # dataset
    global train_data, test_data
    if DatasetName == 'mnist':
        train_data = mnist_dataset(True, transform=transform_train(DatasetName),
                                   target_transform=transform_target,
                                   noise_rate=noise_rate, random_seed=args.seed)
        test_data = mnist_test_dataset(transform=transform_test(DatasetName),
                                       target_transform=transform_target)
        args.n_epoch = 30
        args.num_classes = 10
    elif DatasetName == 'cifar10':
        train_data = cifar10_dataset(True, transform=transform_train(DatasetName),
                                     target_transform=transform_target,
                                     noise_rate=noise_rate, random_seed=args.seed)
        test_data = cifar10_test_dataset(transform=transform_test(DatasetName),
                                         target_transform=transform_target)
        args.n_epoch = 30
        args.num_classes = 10
    elif DatasetName == 'cifar100':
        train_data = cifar100_dataset(True, transform=transform_train(DatasetName),
                                      target_transform=transform_target,
                                      noise_rate=noise_rate, random_seed=args.seed)
        test_data = cifar100_test_dataset(transform=transform_test(DatasetName),
                                          target_transform=transform_target)
        args.n_epoch = 30
        args.num_classes = 100
    # data_loader
    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8)

    test_loader = DataLoader(dataset=test_data,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=8)
    return train_loader, test_loader
