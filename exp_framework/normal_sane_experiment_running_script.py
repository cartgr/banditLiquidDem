from Ensemble import Ensemble
from delegation import DelegationMechanism
from experiment import Experiment

print("Starting!")

batch_size = 128
window_size = 10

del_mech = DelegationMechanism(
            batch_size=batch_size,
            window_size=window_size
        )


print("Creating Experiment")
ensembles= [
    Ensemble(
            training_epochs=1,
            n_voters=10,
            delegation_mechanism=del_mech
            )
]
exp = Experiment(n_trials=5, ensembles=ensembles)
exp.run()

print("Finished experiment!")

