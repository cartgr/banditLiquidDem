import numpy as np
from scipy import stats

class DelegationMechanism:
    def __init__(self, batch_size, window_size=None):
        self.delegations = {}  # key: delegate_from (id), value: delegate_to (id)
        self.t = 0
        self.window_size = window_size
        self.batch_size = batch_size

    def delegate(self, from_id, to_id):
        # cycles are impossible with this mechanism, so we don't need to check for them
        self.delegations[from_id] = to_id

    def wilson_score_interval(self, point_wise_accuracies, confidence=0.99999):
        ups = sum(point_wise_accuracies)
        # downs = len(point_wise_accuracies) - ups
        n = len(point_wise_accuracies)

        # use the specified confidence value to calculate the z-score
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = ups / n

        left = p + 1 / (2 * n) * z * z
        right = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        under = 1 + 1 / n * z * z

        return ((left - right) / under, (left + right) / under)

    def ucb(self, voter, t, c=3.0):
        """
        Calculate upper confidence bound of the bandit arm corresponding to voting directly. Loosely speaking, if this
        is high enough the voter will vote directly.
        point_wise_accuracies is the number of samples this voter has taken, i.e. n_i in UCB terms
        t is the total number of samples taken by any agent, i.e. N in UCB terms

        :param t: number of time steps passed
        :param c: exploration term; higher means more exploration/higher chance of voting directly (?)
        """
        if self.window_size is None:
            point_wise_accuracies = (
                voter.accuracy
            )  # one value per sample that this voter has predicted upon
            # t_window = t  # total number of possible data points within the window
            mean = np.mean(point_wise_accuracies)  # mean accuracy/reward of arm pulls
        else:
            # # get accuracies from the most recent batches, if within the window
            # sorted(voter.batch_accuracies_dict, reverse=True)
            # batch_number = t // self.batch_size
            # point_wise_accuracies = []

            # for batch in range(batch_number - self.window_size, batch_number + 1):
            #     if batch in voter.batch_accuracies_dict:
            #         point_wise_accuracies.append(voter.batch_accuracies_dict[batch])

            # # TODO: Unclear what to do in this case when the voter has not voted recently. Maybe go even higher?
            # if len(point_wise_accuracies) == 0:
            #     # point_wise_accuracies = [0]
            #     mean = 0
            # else:
            #     mean = np.mean(
            #         point_wise_accuracies
            #     )  # mean accuracy/reward of arm pulls

            # t_window = (
            #     self.window_size * self.batch_size
            # )  # total number of possible data points within the window

            # get the most recent window_size predictions and take the mean
            if len(voter.accuracy) < self.window_size:
                mean = np.mean(voter.accuracy)
            else:
                mean = np.mean(voter.accuracy[-self.window_size :])

        # TODO: would be good to have a way of calculating how many of times this voter has voted in the last window_size steps
        # Then our denominator would be much smaller and would give max bonus as soon as the voter has voted in the last window_size steps
        n_t = len(voter.accuracy)  # number of arm pulls the voter has taken

        fudge_factor = 1e-8

        ucb = mean + np.sqrt(c * np.log(t) / (n_t + fudge_factor))
        # ucb = mean + np.sqrt(c * np.log(t_window) / (n_t + fudge_factor))

        return ucb

    def calculate_CI(self, voter):
        point_wise_accuracies = voter.accuracy

        # assume the point wise accuracies are a list of bernoulli random variables
        # approximate using the Wilson score interval
        return self.wilson_score_interval(point_wise_accuracies)

    def update_delegations(self, voters):
        # first, we need to recalculate the CI for each voter
        for voter in voters:
            voter.ucb_score = self.ucb(voter, self.t)

        # now we need to do two things:
        # 1. ensure all current delegations are still valid. If not, remove them
        # 2. go through the full delegation process
        delegators_to_pop = []
        for (
            delegator,
            delegee,
        ) in self.delegations.items():  # check delegations and break invalid ones
            if delegator.ucb_score > delegee.ucb_score:
                delegators_to_pop.append(delegator)
        for delegator in delegators_to_pop:
            self.delegations.pop(delegator)

        for voter in voters:  # go through the full delegation process
            possible_delegees = []
            gaps = []
            for other_voter in voters:
                # find all other voters who have a higher ucb score
                if other_voter.id != voter.id and (
                    other_voter.ucb_score > voter.ucb_score
                ):
                    possible_delegees.append(other_voter)
                    gaps.append(other_voter.ucb_score - voter.ucb_score)
            if len(possible_delegees) > 0:
                # probabilistically delegate based on the gaps
                # larger gaps are more likely to be chosen
                sum_gaps = sum(gaps)
                probabilities = [gap / sum_gaps for gap in gaps]
                delegee = np.random.choice(possible_delegees, p=probabilities)
                self.delegate(voter, delegee)

    def get_gurus(self, voters):
        # find all voters who have not delegated to anyone
        gurus = []
        for voter in voters:
            if voter not in self.delegations.keys():
                gurus.append(voter)
        return gurus