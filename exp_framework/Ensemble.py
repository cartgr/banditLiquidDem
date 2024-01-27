import torch
from tqdm import tqdm
from .learning import Net
from .Voter import Voter
from .data_utils import Data
import random
from torch.utils.data import DataLoader, Subset, ConcatDataset


class Ensemble:
    def __init__(
        self,
        training_epochs,
        n_voters,
        delegation_mechanism,
        name,
        input_dim=28 * 28,  # for mnist
        output_dim=10,  # for mnist
    ):
        self.training_epochs = training_epochs
        self.delegation_mechanism = delegation_mechanism
        self.n_voters = n_voters

        self.name = name
        if self.name is None:
            self.name = f"Ensemble{random.randint(0, 1000)}"

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.voters = []

    def __str__(self) -> str:
        return self.name

    def initialize_voters(self):
        """
        Create a voter for each model. Use to reset voters for speedier resetting between trials.
        """
        voters = []
        # voter_id = 0
        for id in range(self.n_voters):
            model = Net(input_dim=self.input_dim, output_dim=self.output_dim).to(
                self.device
            )
            voters.append(
                Voter(
                    model,
                    # train_loader,
                    self.training_epochs,
                    id,
                )
            )
        self.voters = voters

        self.delegation_mechanism.t = 0

    def train_models(self):
        """
        Create and train the specified number of models on each given data loader.
        # TODO: Probably nice to return (or record elsewhere) training accuracy?
        """

        # models = []

        # for images, labels in self.train_loaders:
        #     self.learn_batch(images, labels)
        for loader in self.train_loader:
            for images, labels in loader:
                self.learn_batch(images, labels)

        # for voter in self.voters:

        #     for _ in range(voter.training_epochs):
        #         for loader in self.train_loaders:
        #             for images, labels in loader:
        #                 images, labels = images.to(self.device), labels.to(self.device)
        #                 voter.optimizer.zero_grad()
        #                 logits = voter.model(images)
        #                 loss = voter.criterion(logits, labels)
        #                 loss.backward()
        #                 voter.optimizer.step()

        return self.voters

    def predict(self, X, y, record_pointwise_accs=False):
        """
        Make predictions on the given examples.

        y is only used if record_pointwise_accs is True, in which case the accuracy of each guru on each example is recorded.

        """
        # gurus = self.get_gurus()
        gurus_and_weights = self.delegation_mechanism.get_gurus_with_weights(
            self.voters
        )
        # print("ensemble predict method is not incorporating weight")
        all_preds = []
        for guru, weight in gurus_and_weights.items():
            predictions = guru.predict(X)
            # if a guru has weight 2, append their predictions twice, etc.
            for i in range(weight):
                all_preds.append(predictions)

            # append len(X) 1s to guru.binary_active because all gurus are active on all examples
            guru.binary_active.extend([1] * len(X))

            if record_pointwise_accs:
                point_wise_accuracies = (predictions == y).float().tolist()
                guru.accuracy.extend(point_wise_accuracies)

        # for each voter that is not a guru, append len(X) 0s to their binary_active list
        # they are not active on any of the examples
        # TODO: Verify that "if voter not in gurus" will actually work?
        for voter in self.voters:
            if voter not in gurus_and_weights.keys():
                voter.binary_active.extend([0] * len(X))

        all_preds = torch.stack(all_preds).transpose(0, 1)
        all_preds = torch.mode(all_preds, dim=1)[0]

        return all_preds

    def predict_proba(self, X, y, record_pointwise_accs=False):
        """
        Make predictions on the given examples.
        First, get the predicted probas from each guru, then sum them up and take the argmax.

        y is only used if record_pointwise_accs is True, in which case the accuracy of each guru on each example is recorded.

        """
        gurus_and_weights = self.delegation_mechanism.get_gurus_with_weights(
            self.voters
        )

        # get the predicted probas from each guru
        all_probas = []
        for guru, weight in gurus_and_weights.items():
            probas = guru.predict_proba(X)
            # if a guru has weight 2, append their predictions twice, etc.
            for i in range(weight):
                all_probas.append(probas)

            # append len(X) 1s to guru.binary_active because all gurus are active on all examples
            guru.binary_active.extend([1] * len(X))

            if record_pointwise_accs:
                # argmax of probas is the prediction
                predictions = torch.argmax(probas, dim=1)
                point_wise_accuracies = (predictions == y).float().tolist()
                guru.accuracy.extend(point_wise_accuracies)

        # for each voter that is not a guru, append len(X) 0s to their binary_active list
        # they are not active on any of the examples

        for voter in self.voters:
            if voter not in gurus_and_weights.keys():
                voter.binary_active.extend([0] * len(X))

        all_probas = torch.stack(all_probas).transpose(0, 1)
        # sum up the probas from each guru
        all_probas = torch.sum(all_probas, dim=1)
        # take the argmax of the summed probas
        all_preds = torch.argmax(all_probas, dim=1)

        return all_preds

    def score(self, X, y, record_pointwise_accs=False):
        """
        If record_pointwise_accs is True, then record the accuracy of each guru on each example.
        Otherwise only batch accuracies are recorded for each voter.
        """
        # Track accuracy within each voter (even delegating ones)
        # TODO: Super inefficient, making each voter predict twice on the same data :/
        for voter in self.voters:
            voter.score(X, y)

        # Compute whole ensemble accuracy
        # predictions = self.predict(X, y, record_pointwise_accs=record_pointwise_accs)
        predictions = self.predict_proba(
            X, y, record_pointwise_accs=record_pointwise_accs
        )
        acc = (predictions == y).float().mean().item()

        return acc

    def get_gurus(self):
        """ """
        return self.delegation_mechanism.get_gurus(self.voters)

    def learn_batch(self, X, y):
        """
        Have each voter learn a single batch of data. Should be able to be used during training or testing?
        """
        for voter in self.voters:
            if not self.delegation_mechanism.voter_is_active(voter):
                # Do not train voters while they're delegating
                continue

            for _ in range(voter.training_epochs):
                images, labels = X.to(self.device), y.to(self.device)
                voter.optimizer.zero_grad()
                logits = voter.model(images)
                loss = voter.criterion(logits, labels)
                loss.backward()
                voter.optimizer.step()

            # print(f"Voter {voter.id} is learning batch {self.delegation_mechanism.t}")
            # print(f"using delegation mechanism {self.name}")

    def update_delegations(self, accs, train, t_increment=None):
        """
        Allow each voter to update their delegations, as applicable.

        Args:
            accs (list): full history of ensemble batch accuracies
            train (bool): True iff in training phase, False iff in testing phase. Hopefully those are the only possibilities...
        """
        self.delegation_mechanism.update_delegations(
            accs, self.voters, train, t_increment
        )

    def calculate_test_accuracy(self):
        """
        Calculate test accuracy for <some amount of data>.
        TODO: Is it best to do this for a single batch at a time? Or for the whole test set?
        """
        UCBs_over_time = {i: [] for i in range(len(self.voters))}
        liquid_dem_proba_accs = []
        liquid_dem_vote_accs = []
        liquid_dem_weighted_vote_accs = []
        full_ensemble_accs = []

        # TODO: Why only test_loaders[0]? It should be all, right?
        for data, target in tqdm(self.test_loader[0]):
            data, target = data.to(self.device), target.to(self.device)

            gurus = self.get_gurus()

            # Get guru accuracies (at pointwise and batch levels)
            for guru in gurus:
                predictions = guru.predict(data)

                # Is this a binary list with an entry corresponding to each specific test example?
                point_wise_accuracies = (predictions == target).float().tolist()
                guru.accuracy.extend(point_wise_accuracies)

                # Is this a list containing an accuracy rate for each batch of data?
                guru.batch_accuracies.append(
                    sum(point_wise_accuracies) / len(point_wise_accuracies)
                )

            # get all of the gurus to predict then take the majority vote
            liquid_dem_preds = []
            for guru in gurus:
                guru_pred = guru.predict(data)

                # find number of delegations (TODO: not quite correct, needs to allow for transitivity)
                guru_weight = 0
                for delegator, delegee in self.delegation_mechanism.delegations.items():
                    if delegee == guru.id:
                        guru_weight += 1
                if guru_weight == 0:
                    guru_weight = 1
                for i in range(
                    guru_weight
                ):  # append one "vote" per weight of each guru
                    liquid_dem_preds.append(guru_pred)

            liquid_dem_preds = torch.stack(liquid_dem_preds).transpose(0, 1)
            # take the majority vote - WEIGHTED VERSION
            liquid_dem_preds = torch.mode(liquid_dem_preds, dim=1)[0]
            liquid_dem_weighted_vote_accs.append(
                (liquid_dem_preds == target).float().mean().item()
            )

            # get all of the gurus to predict then take the majority vote -- UNWEIGHTED VERSION
            liquid_dem_preds = []
            for guru in gurus:
                liquid_dem_preds.append(guru.predict(data))
            liquid_dem_preds = torch.stack(liquid_dem_preds).transpose(0, 1)
            # take the majority vote
            liquid_dem_preds = torch.mode(liquid_dem_preds, dim=1)[0]
            liquid_dem_vote_accs.append(
                (liquid_dem_preds == target).float().mean().item()
            )

            probas = []
            for guru in gurus:
                probas.append(guru.predict_proba(data))
            probas = torch.stack(probas).transpose(0, 1)
            # take the average of class probabilities - CURRENTLY UNWEIGHTED
            probas = torch.mean(probas, dim=1)
            # take the highest probability
            liquid_dem_preds = torch.argmax(probas, dim=1)
            liquid_dem_proba_accs.append(
                (liquid_dem_preds == target).float().mean().item()
            )

            # get all of the voters to predict then take the majority vote
            full_ensemble_preds = []
            for voter in self.voters:
                full_ensemble_preds.append(voter.predict(data))
            full_ensemble_preds = torch.stack(full_ensemble_preds).transpose(0, 1)

            # take the majority vote
            full_ensemble_preds = torch.mode(full_ensemble_preds, dim=1)[0]
            full_ensemble_accs.append(
                (full_ensemble_preds == target).float().mean().item()
            )

            # print(delegation_mechanism.delegations)

            # At the end of the loop, update UCBs and delegations
            for voter in self.voters:
                UCBs_over_time[voter.id].append(voter.ucb_score)

            self.delegation_mechanism.t += len(data)
            self.delegation_mechanism.update_delegations(self.voters)

        return (
            UCBs_over_time,
            liquid_dem_proba_accs,
            liquid_dem_vote_accs,
            liquid_dem_weighted_vote_accs,
            full_ensemble_accs,
        )


class PretrainedEnsemble(Ensemble):
    def __init__(
        self,
        # pretrained_voters,  # List of pretrained Voter instances
        n_voters,
        delegation_mechanism,
        name=None,
        input_dim=28 * 28,  # for mnist
        output_dim=10,  # for mnist
    ):
        super().__init__(
            training_epochs=None,  # Not needed for pretrained models
            n_voters=n_voters,
            delegation_mechanism=delegation_mechanism,
            name=name,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        # self.voters = pretrained_voters

    def do_pretaining(self, data: Data):
        """
        Train each voter on an approprate subset of the training data.

        The Data object has the train_loader and the train splits which we will use to determine which data to train each clf on.
        We will initialize the voters in this method as well.

        Args:
            data: Data instance
            assignments: Dict mapping each voter to a list of digit groups to train on. If None, then split the data evenly.
        """
        # initialize n voters
        for id in range(self.n_voters):
            model = Net(input_dim=self.input_dim, output_dim=self.output_dim).to(
                self.device
            )
            self.voters.append(
                Voter(
                    model,
                    # train_loader,
                    self.training_epochs,
                    id,
                )
            )

        # split voters evenly across digit groups for now
        # TODO: Allow for more flexible assignments
        assignments = dict()
        for voter in self.voters:
            assignments[voter.id] = data.train_digit_group_loaders[
                voter.id % len(data.train_digit_group_loaders)
            ]

        # train each voter on their assigned loaders
        for voter in self.voters:
            for images, labels in assignments[voter.id]:
                images, labels = images.to(self.device), labels.to(self.device)
                voter.optimizer.zero_grad()
                logits = voter.model(images)
                loss = voter.criterion(logits, labels)
                loss.backward()
                voter.optimizer.step()

    def initialize_voters(
        self,
    ):
        return

    def train_models(self):
        return

    def predict(self, X, y, record_pointwise_accs=False):
        """
        Make predictions on the given examples.

        y is only used if record_pointwise_accs is True, in which case the accuracy of each guru on each example is recorded.

        """
        # gurus = self.get_gurus()
        gurus_and_weights = self.delegation_mechanism.get_gurus_with_weights(
            self.voters
        )
        # print("ensemble predict method is not incorporating weight")
        all_preds = []
        for guru, weight in gurus_and_weights.items():
            predictions = guru.predict(X)
            # if a guru has weight 2, append their predictions twice, etc.
            for i in range(weight):
                all_preds.append(predictions)

            # append len(X) 1s to guru.binary_active because all gurus are active on all examples
            guru.binary_active.extend([1] * len(X))

            if record_pointwise_accs:
                point_wise_accuracies = (predictions == y).float().tolist()
                guru.accuracy.extend(point_wise_accuracies)

        # for each voter that is not a guru, append len(X) 0s to their binary_active list
        # they are not active on any of the examples
        # TODO: Verify that "if voter not in gurus" will actually work?
        for voter in self.voters:
            if voter not in gurus_and_weights.keys():
                voter.binary_active.extend([0] * len(X))

        all_preds = torch.stack(all_preds).transpose(0, 1)
        all_preds = torch.mode(all_preds, dim=1)[0]

        return all_preds

    def score(self, X, y, record_pointwise_accs=False):
        """
        If record_pointwise_accs is True, then record the accuracy of each guru on each example.
        Otherwise only batch accuracies are recorded for each voter.
        """
        # Track accuracy within each voter (even delegating ones)
        # TODO: Super inefficient, making each voter predict twice on the same data :/
        for voter in self.voters:
            voter.score(X, y)

        # Compute whole ensemble accuracy
        predictions = self.predict(X, y, record_pointwise_accs=record_pointwise_accs)
        acc = (predictions == y).float().mean().item()

        return acc

    def get_gurus(self):
        """ """
        return self.delegation_mechanism.get_gurus(self.voters)

    def learn_batch(self, X, y):
        return

    def update_delegations(self, accs, train, t_increment=None):
        """
        Allow each voter to update their delegations, as applicable.

        Args:
            accs (list): full history of ensemble batch accuracies
            train (bool): True iff in training phase, False iff in testing phase. Hopefully those are the only possibilities...
        """
        if train:
            return
        self.delegation_mechanism.update_delegations(
            accs, self.voters, train, t_increment
        )
