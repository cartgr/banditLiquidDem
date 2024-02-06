from sklearn import experimental
from exp_framework.Ensemble import Ensemble
from exp_framework.experiment import Experiment
from exp_framework.Ensemble import Ensemble, PretrainedEnsemble
from exp_framework.delegation import (
    DelegationMechanism,
    UCBDelegationMechanism,
    ProbaSlopeDelegationMechanism,
    RestrictedMaxGurusDelegationMechanism,
)
from exp_framework.experiment import (
    Experiment,
    calculate_avg_std_test_accs,
    calculate_avg_std_train_accs,
)
from matplotlib import pyplot as plt
from exp_framework.data_utils import Data
import numpy as np
import matplotlib as mpl
import seaborn as sns

print("Starting!")

batch_size = 128
window_size = 50
num_trials = 10
n_voters = 10


data = Data(
    data_set_name="mnist",
    # train_digit_groups=[range(5), range(5, 10)],
    # train_digit_groups=[[0, 1, 2], [3, 4, 5,], [6, 7, 8, 9]],
    train_digit_groups=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
    # test_digit_groups=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
    # test_digit_groups=[range(5), range(5, 10)],
    test_digit_groups=[range(10)],
    batch_size=batch_size,
)

NOOP_del_mech = DelegationMechanism(batch_size=batch_size, window_size=window_size)

# create several mechanisms with a single active voter
random_better = ProbaSlopeDelegationMechanism(
    batch_size=batch_size,
    window_size=window_size,
    max_active=1,
    probability_function="random_better"
)
probabilistic_better = ProbaSlopeDelegationMechanism(
    batch_size=batch_size,
    window_size=window_size,
    max_active=1,
    probability_function="probabilistic_better"
)
probabilistic_weighted = ProbaSlopeDelegationMechanism(
    batch_size=batch_size,
    window_size=window_size,
    max_active=1,
    probability_function="probabilistic_weighted"
)

ensembles_dict = {
    "full_ensemble":
    Ensemble(
        training_epochs=1,
        n_voters=n_voters,
        delegation_mechanism=NOOP_del_mech,
        name="full_ensemble",
        input_dim=28 * 28,
        output_dim=10,
    ),
    "random_better_delegations":
    Ensemble(
        training_epochs=1,
        n_voters=n_voters,
        delegation_mechanism=random_better,
        name="random_better_delegations",
        input_dim=28 * 28,
        output_dim=10,
    ),
    "probabilistic_better_delegations":
    Ensemble(
        training_epochs=1,
        n_voters=n_voters,
        delegation_mechanism=probabilistic_better,
        name="probabilistic_better_delegations",
        input_dim=28 * 28,
        output_dim=10,
    ),
    "probabilistic_weighted_delegations":
    Ensemble(
        training_epochs=1,
        n_voters=n_voters,
        delegation_mechanism=probabilistic_weighted,
        name="probabilistic_weighted_delegations",
        input_dim=28 * 28,
        output_dim=10,
    ),
}


print("Creating Experiment")
exp = Experiment(n_trials=num_trials, ensembles=list(ensembles_dict.values()), data=data, seed=4090)

print("Running Experiment")
_ = exp.run()

exit()

ensembles = [
    Ensemble(
        training_epochs=1,
        n_voters=n_voters,
        delegation_mechanism="del_mech",
        name="simple_delegating_ensemble",
        input_dim=28 * 28,
        output_dim=10,
    )
]

exp = Experiment(n_trials=1, ensembles=ensembles, data=data)
batch_metric_values = exp.run()

####
# Temporary plotting garbage code, may want to generalize a bit and turn into reusable code
####

print("\n\n----------------------\n\n")
print(batch_metric_values)


def plot_accuracy_and_active_voters(batch_accuracies, active_voter_tuples, title=None):
    """
    Plot the batch accuracies given, coloured so each unique set of active voters gets a unique colour.

    Args:
        batch_accuracies (_type_): _description_
        active_voter_tuples (_type_): _description_
    """

    def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [
            (h, l)
            for i, (h, l) in enumerate(zip(handles, labels))
            if l not in labels[:i]
        ]
        unique = sorted(unique, key=lambda x: x[1])
        ax.legend(*zip(*unique))

    unique_active_sets = set(tuple(av) for av in active_voter_tuples)
    voter_colours = {
        v_tuple: tuple(np.random.choice(range(256), size=3) / 255)
        for v_idx, v_tuple in enumerate(unique_active_sets)
    }

    for batch_idx in range(len(batch_accuracies)):
        acc = batch_accuracies[batch_idx]
        active_tuple = tuple(active_voter_tuples[batch_idx])
        colour = voter_colours[active_tuple]
        label = f"Voters {active_tuple}"

        plt.scatter(batch_idx, acc, color=colour, marker=".", label=label)

    legend_without_duplicate_labels(plt.gca())
    plt.ylim((0, 1))
    if title is None:
        title = "Ensemble Accuracy"
    plt.title(title)
    plt.xlabel("Batch Number")
    plt.ylabel("Accuracy")
    plt.show()
    plt.close()
    plt.clf()


for i in exp.train_splits:
    plt.axvline(x=i, color="red", linestyle="--")
np.random.seed(0)
active_voters = batch_metric_values["simple_delegating_ensemble"][0][
    "active_voters-train"
]
train_accs = batch_metric_values["simple_delegating_ensemble"][0]["batch_train_acc"]

plot_accuracy_and_active_voters(
    train_accs, active_voters, title="Ensemble Training Accuracy"
)


for i in exp.test_splits:
    plt.axvline(x=i, color="red", linestyle="--")
np.random.seed(0)
active_voters = batch_metric_values["simple_delegating_ensemble"][0][
    "active_voters-test"
]
test_accs = batch_metric_values["simple_delegating_ensemble"][0]["batch_test_acc"]
plot_accuracy_and_active_voters(
    test_accs, active_voters, title="Ensemble Testing Accuracy"
)


# # active_voters = [a[0] for a in active_voters]
# unique_active_sets = set(tuple(av) for av in active_voters)
# active_tuples = [tuple(av) for av in active_voters]
# print(f"All active voter sets are {unique_active_sets}")

# default_colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
#  '#7f7f7f', '#bcbd22', '#17becf']

# voter_colours = {
#     v_tuple: tuple(np.random.choice(range(256), size=3)/255) for v_idx, v_tuple in enumerate(unique_active_sets)
# }


# for batch_idx in range(len(train_accs)):
#     acc = train_accs[batch_idx]
#     active_tuple = tuple(active_voters[batch_idx])
#     colour = voter_colours[active_tuple]
#     label = f"Voters {active_tuple}"

#     plt.scatter(batch_idx, acc, color=colour, marker=".", label=label)

# # plt.plot(train_accs)
# plt.ylim((0, 1))
# plt.title("Training Accuracy")
# plt.xlabel("Batch Number")
# plt.ylabel("Accuracy")
# plt.show()

# print("Finished experiment!")
