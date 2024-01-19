from Ensemble import Ensemble
import learning_utils
from delegation import DelegationMechanism
from experiment import Experiment

print("Starting!")

batch_size = 128
window_size = 10
# train_digit_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
train_digit_groups = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
test_digit_groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
train_data_loaders, train_splits = learning_utils.create_mnist_loaders(digit_groups=train_digit_groups,
                                                         batch_size=batch_size,
                                                         train=True
                                                         )

test_data_loaders, test_splits = learning_utils.create_mnist_loaders(digit_groups=test_digit_groups,
                                                        batch_size=batch_size,
                                                        train=False
                                                        )

del_mech = DelegationMechanism(
            batch_size=batch_size,
            window_size=window_size
        )


print("Creating Experiment")
ensembles= [
    Ensemble(
            models_per_train_digit_group=1,
            training_epochs=1,
            batch_size=batch_size,
            window_size=window_size,
            train_loader=train_data_loaders,
            test_loader=test_data_loaders,
            train_digit_groups=train_digit_groups,
            test_digit_groups=test_digit_groups,
            delegation_mechanism=del_mech
            )
]
exp = Experiment(n_trials=5, ensembles=ensembles)
exp.run()

print("Finished experiment!")

