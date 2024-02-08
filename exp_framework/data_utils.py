from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import random
from avalanche.benchmarks.classic import RotatedMNIST
import torch


class Data:
    def __init__(
        self,
        data_set_name: str,
        train_digit_groups: list = None,
        test_digit_groups: list = None,
        batch_size: int = 128,
        seed=0,
    ):
        self.data_set_name = data_set_name
        self.train_digit_groups = train_digit_groups
        self.test_digit_groups = test_digit_groups
        self.batch_size = batch_size

        if data_set_name == "mnist":
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
        elif data_set_name == "rotated_mnist":
            # rotated_mnsit = RotatedMNIST(n_experiences=5, rotations_list=[-180, -90, 0, 90, 180])
            rotated_mnsit = RotatedMNIST(n_experiences=5, seed=seed)
            train_stream = rotated_mnsit.train_stream
            test_stream = rotated_mnsit.test_stream
            (
                self.train_data_loader,
                self.train_splits,
                self.train_digit_group_loaders,
            ) = create_rotated_mnist_loaders(
                train_stream=train_stream,
                test_stream=test_stream,
                train=True,
                batch_size=batch_size,
                return_digit_group_loaders=True,
            )
            (
                self.test_data_loader,
                self.test_splits,
                self.test_digit_group_loaders,
            ) = create_rotated_mnist_loaders(
                train_stream=train_stream,
                test_stream=test_stream,
                train=False,
                batch_size=batch_size,
                return_digit_group_loaders=True,
            )
        else:
            raise ValueError(
                "data_set_name must be one of ['mnist', 'rotated_mnist'], not "
                + data_set_name
            )


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

def shuffle_dataset(dataset):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return Subset(dataset, indices)


def create_rotated_mnist_loaders(
    train_stream,
    test_stream,
    train,
    batch_size=128,
    return_digit_group_loaders=False,
):
    train_datasets = []
    test_datasets = []
    train_splits = []
    test_splits = []

    # Process the training stream
    for experience in train_stream:
        current_training_set = (
            shuffle_dataset(experience.dataset) if train else experience.dataset
        )
        train_datasets.append(current_training_set)
        train_splits.append(
            len(train_datasets) * (len(current_training_set) // batch_size)
        )

    # Process the testing stream
    for experience in test_stream:
        current_test_set = (
            shuffle_dataset(experience.dataset) if train else experience.dataset
        )
        test_datasets.append(current_test_set)
        test_splits.append(len(test_datasets) * (len(current_test_set) // batch_size))

    # Create ConcatDataset from datasets
    data = ConcatDataset(train_datasets if train else test_datasets)
    whole_loader = DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    if return_digit_group_loaders:
        return whole_loader, train_splits if train else test_splits, None

    return whole_loader, train_splits if train else test_splits


def custom_collate_fn(batch):
    # This function will process each batch to exclude the task_id
    batch_images, batch_labels = zip(
        *[(data, target) for (data, target, task_id) in batch]
    )
    return torch.stack(batch_images), torch.tensor(batch_labels)
