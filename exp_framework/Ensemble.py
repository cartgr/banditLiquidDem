import torch
from tqdm import tqdm
from learning import Net
from Voter import Voter

class Ensemble:
    def __init__(
        self,
        models_per_train_digit_group,
        training_epochs,
        batch_size,
        window_size,
        train_loader,
        test_loader,
        train_digit_groups,
        test_digit_groups,
        delegation_mechanism
    ):
        # parameters used during training
        self.models_per_train_digit_group = models_per_train_digit_group
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.train_digit_groups = train_digit_groups
        self.test_digit_groups = test_digit_groups
        self.delegation_mechanism = delegation_mechanism

        # ML stuff created during training
        self.train_loader = train_loader
        self.test_loader = test_loader
        # self.test_split_indices = []
        self.voters = []

        # # TODO: Should pass this in so it can be varied across ensembles being compared
        # self.delegation_mechanism = DelegationMechanism(
        #     batch_size=batch_size,
        #     window_size=window_size
        # )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the loss function
        # self.criterion = nn.CrossEntropyLoss()
        
    def initialize_voters(self):
        """
        Create a voter for each model. Use to reset voters for speedier resetting between trials.
        """
        voters = []
        voter_id = 0
        for group in self.train_digit_groups:
            for _ in range(self.models_per_train_digit_group):
                model = Net().to(self.device)
                voters.append(Voter(model,
                                    # train_loader,
                                    self.training_epochs,
                                    voter_id))
                voter_id += 1
        self.voters = voters

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
    
    def predict(self, X):
        """
        Make predictions on the given examples.
        """
        gurus = self.get_gurus()
        print("ensemble predict method not tested and not incorporating weight")
        all_preds = []
        for guru in gurus:
            predictions = guru.predict(X)
            all_preds.append(predictions)

        all_preds = torch.stack(all_preds).transpose(0, 1)
        all_preds = torch.mode(all_preds, dim=1)[0]

        return all_preds
    
    def score(self, X, y):
        """
        
        """
        print("ensemble score method is untested. Why does it use mean?")

        predictions = self.predict(X)
        acc = (predictions == y).float().mean().item()

        return acc


    def get_gurus(self):
        """
        
        """
        return self.delegation_mechanism.get_gurus(self.voters)

    def learn_batch(self, X, y):
        """
        Have each voter learn a single batch of data. Should be able to be used during training or testing?
        """
        for voter in self.voters:
            for _ in range(voter.training_epochs):
                images, labels = X.to(self.device), y.to(self.device)
                voter.optimizer.zero_grad()
                logits = voter.model(images)
                loss = voter.criterion(logits, labels)
                loss.backward()
                voter.optimizer.step()

    def update_delegations(self, train):
        """
        Allow each voter to update their delegation. Likely needs some other information passed to it as well,
        such as recent accuracy.
        """
        pass


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
                for i in range(guru_weight):  # append one "vote" per weight of each guru
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
            liquid_dem_vote_accs.append((liquid_dem_preds == target).float().mean().item())

            probas = []
            for guru in gurus:
                probas.append(guru.predict_proba(data))
            probas = torch.stack(probas).transpose(0, 1)
            # take the average of class probabilities - CURRENTLY UNWEIGHTED
            probas = torch.mean(probas, dim=1)
            # take the highest probability
            liquid_dem_preds = torch.argmax(probas, dim=1)
            liquid_dem_proba_accs.append((liquid_dem_preds == target).float().mean().item())

            # get all of the voters to predict then take the majority vote
            full_ensemble_preds = []
            for voter in self.voters:
                full_ensemble_preds.append(voter.predict(data))
            full_ensemble_preds = torch.stack(full_ensemble_preds).transpose(0, 1)
            
            # take the majority vote
            full_ensemble_preds = torch.mode(full_ensemble_preds, dim=1)[0]
            full_ensemble_accs.append((full_ensemble_preds == target).float().mean().item())

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