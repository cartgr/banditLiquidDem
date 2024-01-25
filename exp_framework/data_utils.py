from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import random


class Data:
    def __init__(
        self,
        data_set_name: str,
        train_digit_groups: list,
        test_digit_groups: list,
        batch_size: int,
    ):
        self.data_set_name = data_set_name
        self.train_digit_groups = train_digit_groups
        self.test_digit_groups = test_digit_groups
        self.batch_size = batch_size

        if data_set_name not in ["mnist", "rotated_mnist"]:
            raise ValueError(
                "data_set_name must be one of ['mnist', 'rotated_mnist'], not "
                + data_set_name
            )

        # if data_set_name == "mnist":
        (
            self.train_data_loader,
            self.train_splits,
            self.train_digit_group_loaders,
        ) = create_mnist_loaders(
            digit_groups=train_digit_groups,
            batch_size=batch_size,
            train=True,
            return_digit_group_loaders=True,
        )
        (
            self.test_data_loader,
            self.test_splits,
            self.test_digit_group_loaders,
        ) = create_mnist_loaders(
            digit_groups=test_digit_groups,
            batch_size=batch_size,
            train=False,
            return_digit_group_loaders=True,
        )
        # elif data_set_name == "rotated_mnist":


def create_mnist_loaders(
    digit_groups,
    batch_size=128,
    train=True,
    return_digit_group_loaders=False,
):
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

    if return_digit_group_loaders:
        return whole_loader, split_indices, loaders

    return whole_loader, split_indices
    # return loaders, split_indices
