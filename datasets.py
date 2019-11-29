"""
Pytorch datasets for MNIST and CIFAR10, with splitting over several "workers".
User interface is provided with get_data_loaders_per_machine function.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10


def get_data_loaders_per_machine(dataset_name, mode, n_nodes=1, batch_size=100):
    if mode == "train":
        data = get_dataset(dataset_name, train=True)
        loaders = []
        data_per_node = len(data) // n_nodes
        indices = list(np.arange(len(data)))
        np.random.shuffle(indices)
        for node_id, start in zip(range(n_nodes), range(0, len(indices), data_per_node)):
            node_data_indices = indices[start:start+data_per_node]
            loader = DataLoader(Subset(data, node_data_indices), batch_size=batch_size, shuffle=True)
            loaders.append(loader)
        return loaders
    elif mode in ("val", "test"):
        # n_nodes is not used in this case
        data = get_dataset(dataset_name, train=False)
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        return loader
    else:
        raise ValueError("Invalid arguments")


def get_dataset(dataset_name, train):
    if dataset_name == "mnist":
        transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
        ])
        data = MNIST(root='./data', train=train, download=True, transform=transform)
        return data

    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        data = CIFAR10(root='./data', train=train, download=True, transform=transform)
        return data

    else:
        raise ValueError("Unknown dataset {}".format(dataset_name))


def test_data_utils():
    loaders = get_data_loaders_per_machine("mnist", "train", 3)
    for loader in loaders:
        print(len(loader))
        for (x,y) in loader:
            print(x.shape, y.shape)
            break

    loader = get_data_loaders_per_machine("cifar10", "val")
    print(len(loader))


if __name__ == "__main__":
    test_data_utils()


