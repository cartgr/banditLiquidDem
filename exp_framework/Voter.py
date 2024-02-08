import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, balanced_accuracy_score, log_loss, top_k_accuracy_score
import warnings

class Voter:
    def __init__(self, model, training_epochs, id, score_method="accuracy_score"):
        self.model = model  # classifier upon which the voter is built
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.id = id
        # self.data_loader = loader
        self.training_epochs = training_epochs
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = []  # one value per sample that this voter has predicted on
        self.batch_accuracies = []
        self.batch_accuracies_dict = dict()

        self.batch_probas = []

        self.score_method = score_method
        self.metric_scores = { # each metric measured points to a list with one value per batch
            score_method: []
            # "f1_score": [],
            # "precision_score": [],
            # "recall_score": [],
            # "roc_auc_score": [],
            # "accuracy_score": [],
            # "balanced_accuracy_score": [],
            # "log_loss_score": [],
            # "top_k_accuracy_score": [],
        }

        self.CI = (0, 0)
        self.ucb_score = 0

        self.binary_active = (
            []
        )  # list storing one if voter was active on the example, zero otherwise

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
        y_proba = self.predict_proba(X).numpy(force=True)
        accuracy = (y_pred.round() == y).float().mean()
        
        # record a bunch of other metrics for use in delegation
        if self.score_method == "accuracy_score":
            self.metric_scores["accuracy_score"].append(accuracy)

        elif self.score_method == "f1_score":
            self.metric_scores["f1_score"].append(f1_score(y, y_pred, average="weighted", zero_division=0))  # may be best average method for settings where not all classes are always present

        elif self.score_method == "precision_score":
            self.metric_scores["precision_score"].append(precision_score(y, y_pred, average="weighted", zero_division=0))

        elif self.score_method == "recall_score":
            self.metric_scores["recall_score"].append(recall_score(y, y_pred, average="weighted", zero_division=0))

        elif self.score_method == "balanced_accuracy_score":
             with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_proba = self.predict_proba(X).numpy(force=True)
                self.metric_scores["balanced_accuracy_score"].append(balanced_accuracy_score(y, y_pred))
        elif self.score_method == "log_loss_score":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # y_proba = self.predict_proba(X).numpy(force=True)
                # self.metric_scores["roc_auc_score"].append(roc_auc_score(y, y_proba, average="weighted", multi_class="ovr"))
                # TODO: This is horrible and encodes reliance on class labels being digits 0 through 9
                self.metric_scores["log_loss_score"].append(1 - log_loss(y, y_proba, labels=list(range(10))))

        elif self.score_method == "top_k_accuracy_score":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # y_proba = self.predict_proba(X).numpy(force=True)
                # self.metric_scores["roc_auc_score"].append(roc_auc_score(y, y_proba, average="weighted", multi_class="ovr"))
                # TODO: This is horrible and encodes reliance on class labels being digits 0 through 9
                self.metric_scores["top_k_accuracy_score"].append(top_k_accuracy_score(y, y_proba, k=2, labels=list(range(10))))

        self.batch_accuracies.append(accuracy)

        self.batch_probas.append(y_proba)

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

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        elif isinstance(other, int):
            return self.id == other
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self) -> int:
        return self.id