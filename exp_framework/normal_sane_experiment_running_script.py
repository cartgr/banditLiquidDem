from Ensemble import Ensemble
from delegation import DelegationMechanism, RestrictedMaxGurusUCBDelegationMechanism
from experiment import Experiment
import matplotlib.pyplot as plt

print("Starting!")

batch_size = 128
window_size = 10
n_voters = 10

# del_mech = DelegationMechanism(batch_size=batch_size, window_size=window_size)
del_mech = RestrictedMaxGurusUCBDelegationMechanism(batch_size=batch_size,
                                                    num_voters=n_voters,
                                                    max_active_voters=1,
                                                    window_size=window_size,
                                                    t_between_delegation=3)


print("Creating Experiment")

ensembles = [
    Ensemble(
        training_epochs=1,
        n_voters=n_voters,
        delegation_mechanism=del_mech,
        name="simple_delegating_ensemble",
        input_dim=28 * 28,
        output_dim=10,
    )
]

exp = Experiment(n_trials=1, ensembles=ensembles)
batch_metric_values = exp.run()


print("\n\n----------------------\n\n")
print(batch_metric_values)

train_accs = batch_metric_values["simple_delegating_ensemble"][0]["batch_train_acc"]

plt.plot(train_accs)
plt.show()

print("Finished experiment!")
