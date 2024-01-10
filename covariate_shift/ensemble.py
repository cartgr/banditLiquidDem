import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from scipy import stats
import random


class Ensemble:


    def __init__(self, models_per_train_digit_group, training_epochs, batch_size, window_size, train_digit_groups=[[0,1], [2,3], [4,5], [6,7], [8,9]], test_digit_groups=[[0,1,2,3,4], [5,6,7,8,9]]):
        
        # parameters used during training
        self.models_per_train_digit_group = models_per_train_digit_group
        self.training_epochs = training_epochs
        self.train_digit_groups = train_digit_groups
        self.test_digit_groups = test_digit_groups
        self.batch_size = batch_size

        # ML stuff created during training
        self.train_loaders = []
        self.test_loaders = []
        self.test_split_indices = []
        self.models = []

        self.delegation_mechanism = DelegationMechanism(batch_size=batch_size, window_size=window_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the transformation and load the MNIST dataset
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        # Load MNIST dataset
        self.train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        self.test = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        # Define the loss function
        self.criterion = nn.CrossEntropyLoss()


    def prepare_loaders(self):
        self.prepare_train_loaders()
        self.prepare_test_loaders()

    def prepare_train_loaders(self, digit_groups=None):
        """
        Prepare data loaders.
        :param digit_groups: list of collections where each collection (list, set, etc) is the set of digits desired in the corresponding loader
        """
        if digit_groups is None:
            digit_groups = self.train_digit_groups

        train_indices = []

        for digit_group in digit_groups:
            ti = [i for i, label in enumerate(self.train.targets) if label in digit_group]
            train_indices.append(ti)

        # Create Subsets
        subsets = [Subset(self.train, ti) for ti in train_indices]

        # Create DataLoader for all subsets
        loaders = [DataLoader(subset, batch_size=self.batch_size, shuffle=True) for subset in subsets]
        self.train_loaders = loaders

        return loaders


    def prepare_test_loaders(self, digit_groups=None):
        """
        """
        if digit_groups is None:
            digit_groups = self.test_digit_groups

        test_indices = []

        for digit_group in digit_groups:
            ti = [i for i, label in enumerate(self.test.targets) if label in digit_group]
            test_indices.append(ti)

        # Create Subsets
        subsets = [Subset(self.test, ti) for ti in test_indices]

        # Create DataLoader for all subsets
        loaders = [DataLoader(subset, batch_size=self.batch_size, shuffle=True) for subset in subsets]

        test_loader = []
        split_indices = []
        for loader in loaders:
            test_loader += list(loader)
            if len(split_indices) == 0:
                split_indices.append(len(test_loader))
            else:
                split_indices.append(split_indices[-1]+len(test_loader))
        self.test_loaders = loaders
        self.test_split_indices = split_indices

        return test_loader, split_indices


    def train_models(self):
        """
        Create and train the specified number of models on each given data loader.
        """

        models = []

        for loader in self.train_loaders:
            loader_models = [Net().to(self.device) for i in range(self.models_per_train_digit_group)]
            optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in loader_models]

            for _ in tqdm(range(self.training_epochs)):
                for idx, model in enumerate(loader_models):
                    for images, labels in loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        optimizers[idx].zero_grad()
                        logits = loader_models[idx](images)
                        loss = self.criterion(logits, labels)
                        loss.backward()
                        optimizers[idx].step()
                
            models += loader_models
        self.models = models

        return models
    


class Voter:
    def __init__(self, model, id):
        self.model = model
        self.id = id
        self.accuracy = []  # one value per sample that this voter has predicted upon
        self.batch_accuracies = []
        self.batch_accuracies_dict = dict()
        self.CI = (0, 0)
        self.ucb_score = 0

    def partial_fit(self, X, y):
        self.model.partial_fit(X, y)

    def predict(self, X):
        predictions = self.model.predict(X)

        return predictions

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __str__(self):
        return "Voter " + str(self.id)

    def __repr__(self):
        return "Voter " + str(self.id)


class DelegationMechanism:
    def __init__(self, batch_size, window_size=None):
        self.delegations = {}  # key: delegate_from (id), value: delegate_to (id)
        self.t = 0
        self.window_size = window_size
        self.batch_size = batch_size

    def delegate(self, from_id, to_id):
        # cycles are impossible with this mechanism, so we don't need to check for them
        self.delegations[from_id] = to_id

    def wilson_score_interval(self, point_wise_accuracies, confidence=0.99999):
        ups = sum(point_wise_accuracies)
        downs = len(point_wise_accuracies) - ups
        n = len(point_wise_accuracies)

        # use the specified confidence value to calculate the z-score
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = ups / n

        left = p + 1 / (2 * n) * z * z
        right = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        under = 1 + 1 / n * z * z

        return ((left - right) / under, (left + right) / under)

    def ucb(self, voter, t, c=3.0):
        """
        Calculate upper confidence bound of the bandit arm corresponding to voting directly. Loosely speaking, if this
        is high enough the voter will vote directly.
        point_wise_accuracies is the number of samples this voter has taken, i.e. n_i in UCB terms
        t is the total number of samples taken by any agent, i.e. N in UCB terms

        :param t: number of time steps passed
        :param c: exploration term; higher means more exploration/higher chance of voting directly (?)
        """
        if self.window_size is None:
            point_wise_accuracies = voter.accuracy  # one value per sample that this voter has predicted upon
            t_window = t # total number of possible data points within the window
            mean = np.mean(point_wise_accuracies)   # mean accuracy/reward of arm pulls
        else:
            # get accuracies from the most recent batches, if within the window
            sorted(voter.batch_accuracies_dict, reverse=True)
            batch_number = t//self.batch_size
            point_wise_accuracies = []

            for batch in range(batch_number-self.window_size, batch_number+1):
                if batch in voter.batch_accuracies_dict:
                    point_wise_accuracies.append(voter.batch_accuracies_dict[batch])

            # TODO: Unclear what to do in this case when the voter has not voted recently. Maybe go even higher?
            if len(point_wise_accuracies) == 0:
                # point_wise_accuracies = [0]
                mean = 0
            else:
                mean = np.mean(point_wise_accuracies)   # mean accuracy/reward of arm pulls
            
            t_window = self.window_size * self.batch_size # total number of possible data points within the window

        
        n_t = len(point_wise_accuracies)        # number of arm pulls the voter has taken

        fudge_factor = 1e-8

        # ucb = mean + np.sqrt(c * np.log(t) / (n_t + fudge_factor))
        ucb = mean + np.sqrt(c * np.log(t_window) / (n_t + fudge_factor))

        return ucb

    def calculate_CI(self, voter):
        point_wise_accuracies = voter.accuracy

        # assume the point wise accuracies are a list of bernoulli random variables
        # approximate using the Wilson score interval
        return self.wilson_score_interval(point_wise_accuracies)

    def update_delegations(self, voters):
        # first, we need to recalculate the CI for each voter
        for voter in voters:
            voter.ucb_score = self.ucb(voter, self.t)

        # now we need to do two things:
        # 1. ensure all current delegations are still valid. If not, remove them
        # 2. go through the full delegation process
        delegators_to_pop = []
        for (
            delegator,
            delegee,
        ) in self.delegations.items():  # check delegations and break invalid ones
            if delegator.ucb_score > delegee.ucb_score:
                delegators_to_pop.append(delegator)
        for delegator in delegators_to_pop:
            self.delegations.pop(delegator)

        for voter in voters:  # go through the full delegation process
            possible_delegees = []
            gaps = []
            for other_voter in voters:
                # find all other voters who have a higher ucb score
                if other_voter.id != voter.id and (
                    other_voter.ucb_score > voter.ucb_score
                ):
                    possible_delegees.append(other_voter)
                    gaps.append(other_voter.ucb_score - voter.ucb_score)
            if len(possible_delegees) > 0:
                # probabilistically delegate based on the gaps
                # larger gaps are more likely to be chosen
                sum_gaps = sum(gaps)
                probabilities = [gap / sum_gaps for gap in gaps]
                delegee = np.random.choice(possible_delegees, p=probabilities)
                self.delegate(voter, delegee)

    def get_gurus(self, voters):
        # find all voters who have not delegated to anyone
        gurus = []
        for voter in voters:
            if voter not in self.delegations.keys():
                gurus.append(voter)
        return gurus
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 output classes for MNIST digits

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
