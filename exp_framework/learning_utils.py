from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import random


def create_mnist_loaders(digit_groups, batch_size=128, train=True):
    """ """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    if train:
        ds = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        ds = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    indices = []
    for digit_group in digit_groups:
        ti = [i for i, label in enumerate(ds.targets) if label in digit_group]
        random.shuffle(ti)  # shuffles in place
        indices.append(ti)

    subsets = [Subset(ds, ti) for ti in indices]
    loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets
    ]

    data = ConcatDataset(subsets)
    whole_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    test_loader = []
    split_indices = []
    for loader in loaders:
        test_loader += list(loader)
        split_indices.append(len(test_loader))
        # if len(split_indices) == 0:
        #     split_indices.append(len(test_loader))
        # else:
        #     split_indices.append(split_indices[-1] + len(test_loader))

    return whole_loader, split_indices
    # return loaders, split_indices
