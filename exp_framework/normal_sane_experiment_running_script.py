import os
from exp_framework.Ensemble import Ensemble, PretrainedEnsemble, StudentExpertEnsemble
from exp_framework.delegation import (
    DelegationMechanism,
    UCBDelegationMechanism,
    ProbaSlopeDelegationMechanism,
    RestrictedMaxGurusDelegationMechanism,
    StudentExpertDelegationMechanism,
)
from exp_framework.learning import Net
from exp_framework.experiment import (
    Experiment,
    calculate_avg_std_test_accs,
    calculate_avg_std_train_accs,
)
from avalanche.training.supervised import Naive
from matplotlib import pyplot as plt
from exp_framework.data_utils import Data
from avalanche.benchmarks.classic import RotatedMNIST, SplitMNIST
import numpy as np
import matplotlib as mpl
import seaborn as sns
from itertools import product
import pandas as pd
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from avalanche.training.plugins import (
    CWRStarPlugin,
    ReplayPlugin,
    EWCPlugin,
    TrainGeneratorAfterExpPlugin,
    LwFPlugin,
    SynapticIntelligencePlugin,
)
from avalanche.training import LwF, EWC, SynapticIntelligence, Replay
from avalanche.training.plugins import EvaluationPlugin
from exp_framework.MinibatchEvalAccuracy import MinibatchEvalAccuracy
from avalanche.evaluation.metrics import accuracy_metrics

batch_size = 128
window_size = 50
num_trials = 3
n_voters = 10


data = SplitMNIST(n_experiences=5, fixed_class_order=list(range(10)))

NOOP_del_mech = DelegationMechanism(batch_size=batch_size, window_size=window_size)

# probability_functions = [
#     "random_better",
#     "probabilistic_better",
#     "probabilistic_weighted",
# ]
# score_functions = [
#     "accuracy_score",
#     "balanced_accuracy_score",
#     "f1_score",
#     "precision_score",
#     "recall_score",
#     "top_k_accuracy_score",
#     "roc_auc_score",
#     "log_loss_score",
#     "max_diversity",
# ]
probability_functions = ["max_diversity"]
score_functions = ["accuracy_score"]
max_active_gurus = 1

del_mechs = {"full-ensemble": NOOP_del_mech}
for prob_func, score_func in product(probability_functions, score_functions):
    dm = ProbaSlopeDelegationMechanism(
        batch_size=batch_size,
        window_size=window_size,
        max_active=max_active_gurus,
        probability_function=prob_func,
        score_method=score_func,
    )
    del_mechs[f"{prob_func}-{score_func}"] = dm


ensembles_dict = {
    dm_name: Ensemble(
        training_epochs=1,
        n_voters=n_voters,
        delegation_mechanism=dm,
        name=dm_name,
        input_dim=28 * 28,
        output_dim=10,
    )
    for dm_name, dm in del_mechs.items()
}


def initialize_strategies_to_evaluate():
    plugins_to_evaluate = {
        "LwF": LwFPlugin(),
        "EWC": EWCPlugin(ewc_lambda=0.001),
        "SynapticIntelligence": SynapticIntelligencePlugin(si_lambda=0.5),
        # "Replay": ReplayPlugin(mem_size=100),
    }

    strategies_to_evaluate = {}
    for name, pte in plugins_to_evaluate.items():
        
        model = Net(input_dim=28 * 28, output_dim=10)
        optimize = optim.Adam(model.parameters(), lr=0.001)
        
        mb_eval = MinibatchEvalAccuracy()
        evp = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            mb_eval
        )
        cl_strategy = Naive(
            model=model,
            optimizer=optimize,
            criterion=CrossEntropyLoss(),
            train_mb_size=batch_size,
            train_epochs=1,
            eval_mb_size=batch_size,
            # plugins=[pte, evp],
            plugins=[pte, evp, mb_eval]
        )
        strategies_to_evaluate[name] = (cl_strategy, evp)
    
    return strategies_to_evaluate




# Train ensembles - single guru

one_active_exp = Experiment(
    n_trials=num_trials,
    ensembles=list(ensembles_dict.values()),
    benchmark=data,
    strategies_to_evaluate=initialize_strategies_to_evaluate,
)
_ = one_active_exp.run()





batch_metrics = one_active_exp.get_aggregate_batch_metrics()
dfs = []
for ens, metric_dict in batch_metrics.items():
    df = pd.DataFrame.from_dict(metric_dict, orient="index")
    df["ensemble_name"] = ens
    dfs.append(df)
df = pd.concat(dfs)
col_order = [len(df.columns) - 1] + list(range(len(df.columns) - 1))
df = df[df.columns[col_order]]
file_prefix = f"class_incremental_single_guru-trials={num_trials}-batch_size={batch_size}_window_size={window_size}"
path = "results"

if not os.path.exists(path):
    os.mkdir(path)

filepath = f"{path}/{file_prefix}.csv" 
df.to_csv(filepath)







# Print results - single guru

print(f"Results for mechanisms with max_active_gurus = {max_active_gurus}:")

# Collect and print train accuracies - aggregate and by batch
train_results_dict = dict()
for ens_name, ensemble in ensembles_dict.items():
    train_acc, train_acc_std = calculate_avg_std_train_accs(
        one_active_exp, ens_name, num_trials
    )
    train_results_dict[ens_name] = (train_acc, train_acc_std)

for strat_name, (strat, eval_plugin) in initialize_strategies_to_evaluate().items():
    train_acc, train_acc_std = calculate_avg_std_train_accs(
        one_active_exp, strat_name, num_trials
    )
    train_results_dict[strat_name] = (train_acc, train_acc_std)

for ens_name, (train_acc, train_acc_std) in train_results_dict.items():
    print(
        f"Mean train acc for {ens_name}: {round(np.mean(train_acc), 3)}+-{round(np.mean(train_acc_std), 3)}"
    )
# for ens_name, (train_acc, train_acc_std) in train_results_dict.items():
#     print(f"All train accs for {ens_name}: {train_acc}")

print("--------------")

# Collect and print test accuracies
results_dict = dict()
for ens_name, ensemble in ensembles_dict.items():
    test_acc, test_acc_std = calculate_avg_std_test_accs(
        one_active_exp, ens_name, num_trials
    )
    results_dict[ens_name] = (test_acc, test_acc_std)

for strat_name, (strat, eval_plugin) in initialize_strategies_to_evaluate().items():
    test_acc, test_acc_std = calculate_avg_std_test_accs(
        one_active_exp, strat_name, num_trials
    )
    results_dict[strat_name] = (test_acc, test_acc_std)

for ens_name, (test_acc, test_acc_std) in results_dict.items():
    print(
        f"Mean test acc for {ens_name}: {round(np.mean(test_acc), 3)}+-{round(np.mean(test_acc_std), 3)}"
    )