import learning_utils

class Experiment():
    """
    A single Experiment class creates, trains, and compares several types of ensemble over multiple trials.
    """

    def __init__(self, n_trials, ensembles):
        
        self.window_size = 10
        self.batch_size = 128
        # train_digit_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        self.train_digit_groups = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
        self.test_digit_groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

        self.train_data_loader, self.train_splits = learning_utils.create_mnist_loaders(digit_groups=self.train_digit_groups,
                                                                                         batch_size=self.batch_size,
                                                                                         train=True
                                                                                         )
        self.test_data_loader, self.test_splits = learning_utils.create_mnist_loaders(digit_groups=self.test_digit_groups,
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
        
        self.batch_metric_values = dict()
        self.experiment_metric_values = dict()


    def run(self):
        """
        Run all trials within this Experiment. During each trial: Generate new ensembles, train them, and save measurements about their
        performance.
        """
        for t in range(self.n_trials):
            self.single_trial(t)


    def single_trial(self, trial_num):
        """
        Run a single trial of this Experiment. Generate relevant ensembles, train them, and save measurements about their performance.
        """
        # 1 - Get ensembles ready for a new trial
        
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
        for images, labels in self.train_data_loader:
            for ensemble in self.ensembles:

                # Train on a batch of data
                ensemble.learn_batch(images, labels)

                # Delegate
                ensemble.update_delegations(train=True)

                # Record performance
                train_acc = ensemble.score(images, labels)
                self.add_batch_metric_value(trial_num, "batch_train_acc", train_acc)
        print("Finished training. Starting testing.")

    
        # 3 - Test each ensemble on each increment of data, as applicable.
        # The idea is that there's e.g. one ensemble delegating, one not delegating, etc.
        for images, labels in self.test_data_loader:
            for ensemble in self.ensembles:

                # Train on a batch of data
                ensemble.learn_batch(images, labels)

                # Delegate
                ensemble.update_delegations(train=False)

                # Record performance
                test_acc = ensemble.score(images, labels)
                self.add_batch_metric_value(trial_num, "batch_test_acc", test_acc)
            
        
        print(self.batch_metric_values)

    
        # # 3 - Measure performance of each ensemble.
        # for ensemble in self.ensembles:
        #     # record test accuracy and other metrics

        #     UCBs_over_time, liquid_dem_proba_accs, liquid_dem_vote_accs, liquid_dem_weighted_vote_accs, full_ensemble_accs = ensemble.calculate_test_accuracy()
        #     # test_acc = ensemble.calculate_test_accuracy()
        #     print(f"Ensemble had test_acc={liquid_dem_weighted_vote_accs}")
        #     self.metric_values["test_accuracy"].append(liquid_dem_weighted_vote_accs)
        
        return self.batch_metric_values


    def add_batch_metric_value(self, trial_num, metric_name, metric_value):
        """
        Record the value of some metric that has a value at each individual batch/increment of learning.
        e.g. train/test accuracy, number of gurus, guru weights, etc.

        Args:
            trial_num (_type_): _description_
            metric_name (_type_): _description_
            metric_value (_type_): _description_
        """
        if trial_num not in self.batch_metric_values:
            self.batch_metric_values[trial_num] = dict()
            if metric_name not in self.batch_metric_values[trial_num]:
                self.batch_metric_values[trial_num][metric_name] = []
        elif metric_name not in self.batch_metric_values[trial_num]:
            self.batch_metric_values[trial_num][metric_name] = []
        
        self.batch_metric_values[trial_num][metric_name].append(metric_value)


    def add_experiment_metric_value(self, metric_name, metric_value):
        """
        Record the value of some metric that gets recorded only once per experiment. I forget why I put this in but probably it'll be 
        convenient for something.

        Args:
            metric_name (_type_): _description_
            metric_value (_type_): _description_
        """

        if metric_name not in self.batch_metric_values:
            self.batch_metric_values[metric_name] = []
        self.batch_metric_values[metric_name].append(metric_value)