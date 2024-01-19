import torch.optim as optim
import torch.nn as nn

class Voter:
    def __init__(self, model, loader, training_epochs, id):
        self.model = model  # classifier upon which the voter is built
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.id = id
        self.data_loader = loader
        self.training_epochs = training_epochs
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = []  # one value per sample that this voter has predicted on
        self.batch_accuracies = []
        self.batch_accuracies_dict = dict()
        self.CI = (0, 0)
        self.ucb_score = 0

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