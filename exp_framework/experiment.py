from matplotlib.pyplot import tick_params
from .learning_utils import *
import numpy as np
from tqdm import tqdm
import random
import numpy as np
import torch
import os


class Experiment:
    """
    A single Experiment class creates, trains, and compares several types of ensemble over multiple trials.
    """

    def __init__(self, n_trials, ensembles, data, seed=42, verbose=False):
        self.window_size = 10
        self.batch_size = 128
        # self.train_digit_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        self.train_digit_groups = data.train_digit_groups
        # self.train_digit_groups = [
        #     [0, 1, 2, 3, 4, 5, 6, 7, 8],
        #     [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # ]
        self.test_digit_groups = data.test_digit_groups
        # self.test_digit_groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        # self.test_digit_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        # self.test_digit_groups = [[2, 3], [0, 1], [8, 9], [4, 5], [6, 7]]
        # self.test_digit_groups = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

        # self.train_data_loader, self.train_splits = learning_utils.create_mnist_loaders(
        #     digit_groups=self.train_digit_groups, batch_size=self.batch_size, train=True
        # )
        # self.test_data_loader, self.test_splits = learning_utils.create_mnist_loaders(
        #     digit_groups=self.test_digit_groups, batch_size=self.batch_size, train=False
        # )

        self.train_data_loader = data.train_data_loader
        self.train_splits = data.train_splits
        self.test_data_loader = data.test_data_loader
        self.test_splits = data.test_splits

        self.ensembles = ensembles
        self.n_trials = n_trials

        # Very clunky, update as useful
        self.metrics_to_record = ["test_accuracy"]
        self.metric_values = {m: [] for m in self.metrics_to_record}

        self.batch_metric_values = dict()
        self.experiment_metric_values = dict()

        self.seed = seed
        self.verbose = verbose

    def run(self):
        """
        Run all trials within this Experiment. During each trial: Generate new ensembles, train them, and save measurements about their
        performance.
        """
        for t in tqdm(range(self.n_trials)):
            # Set seed for reproducibility
            # seed_everything(self.seed + t)

            self.single_trial(t)

        return self.batch_metric_values

    def single_trial(self, trial_num):
        """
        Run a single trial of this Experiment. Generate relevant ensembles, train them, and save measurements about their performance.
        """
        # 1 - Get ensembles ready for a new trial
        # print("Starting trial", trial_num)
        # for ensemble in self.ensembles:
        #     print("Delegations for ensemble", ensemble.name, "before trial", trial_num)
        #     print(ensemble.delegation_mechanism.delegations)

        for ensemble in self.ensembles:
            ensemble.initialize_voters()

        """
        My current thought is to essentially train and test incrementally with opportunity for delegation in each of those.
        Then the delegation mechanism for that ensemble may or may not do any delegation during training/testing.
        And training/testing may be a bit arbitrary, other words might be more suitable.
        Does that cover our use cases?
        So we can have a delegation mechanism that delegates all the time, another that delegates only during testing, 
        another that delegates differently during training and testing, etc.
        """

        # 2 - Incrementally train each ensemble, as applicable.
        # Over each increment of data, train each ensemble on that increment.
        # The idea is that there's e.g. one ensemble delegating, one not delegating, etc.
        batch_idx = 0
        current_digit_group = 0
        for images, labels in self.train_data_loader:
            batch_idx += 1
            if batch_idx in self.train_splits:
                if self.verbose:
                    print(
                        f"Switching from digit group {self.train_digit_groups[current_digit_group]} to {self.train_digit_groups[current_digit_group+1]}"
                    )
                current_digit_group += 1
            for ensemble in self.ensembles:  # TODO: use train_models from ensemble.py?
                # Train on a batch of data
                ensemble.learn_batch(images, labels)

                # Record performance - do this before any delegation so there's at least some data
                train_acc = ensemble.score(images, labels, record_pointwise_accs=False)
                self.add_batch_metric_value(
                    ensemble, trial_num, "batch_train_acc", train_acc
                )
                self.add_batch_metric_value(
                    ensemble,
                    trial_num,
                    "active_voters-train",
                    [
                        g.id
                        for g in ensemble.delegation_mechanism.get_gurus(
                            ensemble.voters
                        )
                    ],
                )

                # Delegate
                train_acc_history = self.get_batch_metric_value(
                    ensemble=ensemble,
                    trial_num=trial_num,
                    metric_name="batch_train_acc",
                )

                # if ensemble.name == "proba_slope_delegations":
                #     print("Trial", trial_num)
                #     print(len(train_acc_history))
                #     print()

                ensemble.update_delegations(
                    accs=train_acc_history, train=True, t_increment=1
                )  # For ucb, if train is true dont do anything. We want to train all clfs

        if self.verbose:
            print("Finished training. Starting testing.")

        # 3 - Test each ensemble on each increment of data, as applicable.
        # TODO: Is it reasonable to assume the testing phase is simply scoring and (potentially) delegating?
        # The idea is that there's e.g. one ensemble delegating, one not delegating, etc.
        for images, labels in self.test_data_loader:
            for ensemble in self.ensembles:
                # # Train on a batch of data -- probably shouldn't during testing?
                # ensemble.learn_batch(images, labels)

                # # t_increment is the number of samples in the current batch
                # t_increment = len(images)

                # Record performance
                test_acc = ensemble.score(
                    images, labels, record_pointwise_accs=True
                )  # we need to record guru pointwise accs on the test set for ucb delegation
                self.add_batch_metric_value(
                    ensemble, trial_num, "batch_test_acc", test_acc
                )
                self.add_batch_metric_value(
                    ensemble,
                    trial_num,
                    "active_voters-test",
                    [
                        g.id
                        for g in ensemble.delegation_mechanism.get_gurus(
                            ensemble.voters
                        )
                    ],
                )

                # Delegate
                test_acc_history = self.get_batch_metric_value(
                    ensemble=ensemble, trial_num=trial_num, metric_name="batch_test_acc"
                )
                ensemble.update_delegations(
                    accs=test_acc_history, train=False, t_increment=1
                )

        if self.verbose:
            print(self.batch_metric_values)

        return self.batch_metric_values

    def add_batch_metric_value(self, ensemble, trial_num, metric_name, metric_value):
        """
        Record the value of some metric that has a value at each individual batch/increment of learning.
        e.g. train/test accuracy, number of gurus, guru weights, etc.

        Store metrics in, arguably, too many layers of dicts: metric_values[ensemble.name][trial_num][metric_name] = metric_value
        Args:
            ensemble (_type_): _description_
            trial_num (_type_): _description_
            metric_name (_type_): _description_
            metric_value (_type_): _description_
        """
        if ensemble.name not in self.batch_metric_values:
            self.batch_metric_values[ensemble.name] = dict()
        if trial_num not in self.batch_metric_values[ensemble.name]:
            self.batch_metric_values[ensemble.name][trial_num] = dict()
        if metric_name not in self.batch_metric_values[ensemble.name][trial_num]:
            self.batch_metric_values[ensemble.name][trial_num][metric_name] = []

        self.batch_metric_values[ensemble.name][trial_num][metric_name].append(
            metric_value
        )

    def get_batch_metric_value(self, ensemble, trial_num, metric_name):
        """
        Get all values of some metric stored by an ensemble.

        Store metrics in, arguably, too many layers of dicts: metric_values[ensemble.name][trial_num][metric_name] = metric_value
        Args:
            ensemble (_type_): _description_
            trial_num (_type_): _description_
            metric_name (_type_): _description_
        """
        value = []
        if ensemble.name not in self.batch_metric_values:
            pass
        elif trial_num not in self.batch_metric_values[ensemble.name]:
            pass
        elif metric_name not in self.batch_metric_values[ensemble.name][trial_num]:
            pass

        value = self.batch_metric_values[ensemble.name][trial_num][metric_name]
        return value

    def add_experiment_metric_value(self, metric_name, metric_value):
        """
        Record the value of some metric that gets recorded only once per experiment. I forget why I put this in but probably it'll be
        convenient for something.
        May want to store by ensemble name as well? Decide when actually using...
        Args:
            metric_name (_type_): _description_
            metric_value (_type_): _description_
        """

        if metric_name not in self.batch_metric_values:
            self.batch_metric_values[metric_name] = []
        self.batch_metric_values[metric_name].append(metric_value)


def calculate_avg_std_test_accs(exp, ensemble_name, n_trials):
    """
    Calculate average and standard deviation of test accuracies for a given number of trials.

    :param exp: The experiment object containing batch metric values.
    :param ensemble_name: The name of the ensemble to calculate metrics for.
    :param n_trials: The number of trials to include in the calculation.
    :return: A tuple of two lists - average accuracies and standard deviations.
    """
    avg_test_accs = []
    std_test_accs = []

    # Initialize a list to collect all test accuracies for each batch
    all_test_accs = [
        []
        for _ in range(len(exp.batch_metric_values[ensemble_name][0]["batch_test_acc"]))
    ]

    # Iterate over each trial and collect test accuracies
    for trial in range(n_trials):
        trial_test_accs = exp.batch_metric_values[ensemble_name][trial][
            "batch_test_acc"
        ]
        for i, acc in enumerate(trial_test_accs):
            all_test_accs[i].append(acc)

    # Calculate average and standard deviation for each batch
    for batch_accs in all_test_accs:
        avg_test_accs.append(np.mean(batch_accs))
        std_test_accs.append(np.std(batch_accs))

    return avg_test_accs, std_test_accs


def calculate_avg_std_train_accs(exp, ensemble_name, n_trials):
    """
    Calculate average and standard deviation of train accuracies for a given number of trials.

    :param exp: The experiment object containing batch metric values.
    :param ensemble_name: The name of the ensemble to calculate metrics for.
    :param n_trials: The number of trials to include in the calculation.
    :return: A tuple of two lists - average accuracies and standard deviations.
    """
    avg_train_accs = []
    std_train_accs = []

    # Initialize a list to collect all train accuracies for each batch
    all_train_accs = [
        []
        for _ in range(
            len(exp.batch_metric_values[ensemble_name][0]["batch_train_acc"])
        )
    ]

    # Iterate over each trial and collect train accuracies
    for trial in range(n_trials):
        trial_train_accs = exp.batch_metric_values[ensemble_name][trial][
            "batch_train_acc"
        ]
        for i, acc in enumerate(trial_train_accs):
            all_train_accs[i].append(acc)

    # Calculate average and standard deviation for each batch
    for batch_accs in all_train_accs:
        avg_train_accs.append(np.mean(batch_accs))
        std_train_accs.append(np.std(batch_accs))

    return avg_train_accs, std_train_accs


def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
