from sklearn import experimental
from exp_framework.Ensemble import Ensemble, PretrainedEnsemble, StudentExpertEnsemble
from exp_framework.delegation import DelegationMechanism, RestrictedMaxGurusDelegationMechanism, ProbaSlopeDelegationMechanism, UCBDelegationMechanism, StudentExpertDelegationMechanism
from exp_framework.experiment import Experiment
import matplotlib.pyplot as plt
import numpy as np
from exp_framework.data_utils import Data

print("Starting!")

batch_size = 128
window_size = 10
n_voters = 10
num_trials = 5


# Set up data for class incremental framework

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

max_active_gurus = 1
# create several mechanisms with a single active voter
random_better = ProbaSlopeDelegationMechanism(
    batch_size=batch_size,
    window_size=window_size,
    max_active=max_active_gurus,
    probability_function="random_better"
)
probabilistic_better = ProbaSlopeDelegationMechanism(
    batch_size=batch_size,
    window_size=window_size,
    max_active=max_active_gurus,
    probability_function="probabilistic_better"
)
probabilistic_weighted = ProbaSlopeDelegationMechanism(
    batch_size=batch_size,
    window_size=window_size,
    max_active=max_active_gurus,
    probability_function="probabilistic_weighted"
)
student_expert_del_mech = StudentExpertDelegationMechanism(
    batch_size=batch_size,
    window_size=window_size
)
restricted_max_gurus_mech = RestrictedMaxGurusDelegationMechanism(
    batch_size=batch_size,
    num_voters=n_voters,
    max_active_voters=max_active_gurus,
    window_size=window_size,
    t_between_delegation=3,
)
UCB_del_mech = UCBDelegationMechanism(
    batch_size=batch_size,
    window_size=window_size,
    ucb_window_size=None
)


pretrained_ensemble = PretrainedEnsemble(
    n_voters=n_voters,
    delegation_mechanism=UCB_del_mech,
    name="UCB_delegation_ensemble"
    )
pretrained_ensemble.do_pretaining(data)

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
    # "student_expert_delegations":
    # StudentExpertEnsemble(
    #     training_epochs=1,
    #     n_voters=n_voters,
    #     delegation_mechanism=student_expert_del_mech,
    #     name="student_expert_ensemble",
    #     input_dim=28 * 28,
    #     output_dim=10,
    # ),
    "restricted_max_gurus":
    Ensemble(
        training_epochs=1,
        n_voters=n_voters,
        delegation_mechanism=restricted_max_gurus_mech,
        name="student_expert_ensemble",
        input_dim=28 * 28,
        output_dim=10,
    ),
    "UCB_delegations": pretrained_ensemble
}

one_active_exp = Experiment(n_trials=num_trials, ensembles=list(ensembles_dict.values()), data=data, seed=4090)
batch_metric_values = one_active_exp.run()
exit()

####
# Temporary plotting garbage code, may want to generalize a bit and turn into reusable code
####

print("\n\n----------------------\n\n")
print(batch_metric_values)

exp = None


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
