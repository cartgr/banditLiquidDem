import torch.optim as optim
import torch.nn as nn

class Voter:
    def __init__(self, model, training_epochs, id):
        self.model = model  # classifier upon which the voter is built
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.id = id
        # self.data_loader = loader
        self.training_epochs = training_epochs
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = []  # one value per sample that this voter has predicted on
        self.batch_accuracies = []
        self.batch_accuracies_dict = dict()
        self.CI = (0, 0)
        self.ucb_score = 0

    def score(self, X, y):
        """
        Calculate, save, and return the accuracy of this model at predicting the given batch of data.

        Args:
            X (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self.predict(X)
        accuracy = (y_pred.round() == y).float().mean()
        self.batch_accuracies.append(accuracy)
        return accuracy

    def partial_fit(self, X, y):
        self.model.partial_fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __str__(self):
        return "Voter " + str(self.id)

    def __repr__(self):
        return "Voter " + str(self.id)