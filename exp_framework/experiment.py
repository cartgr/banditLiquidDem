from Ensemble import Ensemble
import learning_utils
from delegation import DelegationMechanism

class Experiment():
    """
    A single Experiment class creates, trains, and compares several types of ensemble over multiple trials.
    """

    def __init__(self, n_trials, ensembles):
        
        self.window_size = 10
        self.batch_size = 128
        self.train_digit_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        self.test_digit_groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

        self.train_data_loaders, self.train_splits = learning_utils.create_mnist_loaders(digit_groups=self.train_digit_groups,
                                                                                         batch_size=self.batch_size,
                                                                                         train=True
                                                                                         )
        self.test_data_loaders, self.test_splits = learning_utils.create_mnist_loaders(digit_groups=self.test_digit_groups,
                                                                                       batch_size=self.batch_size,
                                                                                       train=False
                                                                                       )

        self.ensembles = ensembles
        self.n_trials = n_trials

        # Very clunky, update as useful
        self.metrics_to_record = [
            "test_accuracy"
        ]
        self.metric_values = {m: [] for m in self.metrics_to_record}


    def run(self):
        """
        Run all trials within this Experiment. During each trial: Generate new ensembles, train them, and save measurements about their
        performance.
        """
        for _ in range(self.n_trials):
            self.single_trial()


    def single_trial(self):
        """
        Run a single trial of this Experiment. Generate relevant ensembles, train them, and save measurements about their performance.
        """
        # 1 - Generate all the ensembles compared in this experiment.
        pass
    
        # 2 - Incrementally train each ensemble, as applicable.
        # Over each increment of data, train each ensemble on that increment.
        # The idea is that there's e.g. one ensemble delegating, one not delegating, etc.
        for ensemble in self.ensembles:
            ensemble.initialize_voters()
            print("Training a model!")
            ensemble.train_models()
    
        # 3 - Measure performance of each ensemble.
        for ensemble in self.ensembles:
            # record test accuracy and other metrics

            UCBs_over_time, liquid_dem_proba_accs, liquid_dem_vote_accs, liquid_dem_weighted_vote_accs, full_ensemble_accs = ensemble.calculate_test_accuracy()
            # test_acc = ensemble.calculate_test_accuracy()
            print(f"Ensemble had test_acc={liquid_dem_weighted_vote_accs}")
            self.metric_values["test_accuracy"].append(liquid_dem_weighted_vote_accs)
        
        return self.metric_values
