import numpy as np
from scipy import stats


class DelegationMechanism:
    def __init__(self, batch_size, window_size=None):
        self.delegations = {}  # key: delegate_from (id), value: delegate_to (id)
        self.t = 0
        self.window_size = window_size
        self.batch_size = batch_size

    def delegate(self, from_id, to_id):
        """
        Create a new delegation from one voter to another which also removes any old delegations the new delegator had.
        Currently, do not bother considering cycles or any reason why a delegation may be impossible.
        Args:
            from_id (_type_): _description_
            to_id (_type_): _description_
        """
        self.delegations[from_id] = to_id

    # def calculate_CI(self, voter):
    # Seems unused? - commented for now in case this breaks anything. Who needs Git...
    #     # assume the point wise accuracies are a list of bernoulli random variables
    #     # approximate using the Wilson score interval
    #     point_wise_accuracies = voter.accuracy

    #     return wilson_score_interval(point_wise_accuracies)

    def update_delegations(self, ensemble_accs, voters, train, t_increment=None):
        """
        Update the delegation of each voter based on its recent performance, overall ensemble performance, and whether training or not.
        The base DelegationMechanism class (currently) does not delegate at all. Subclasses should override this in order to provide
        customized delegation behaviour.
        Args:
            ensemble_accs (list): all batch accuracies of the ensemble as a whole
            voters (Voter): all Voters within the ensemble, they know their own accuracy history
            train (bool): True iff in training phase
        """
        if t_increment:
            self.t += t_increment

    def get_gurus(self, voters):
        # find all voters who have not delegated to anyone
        gurus = []
        for voter in voters:
            if voter not in self.delegations.keys():
                gurus.append(voter)
        return gurus

    def get_gurus_with_weights(self, voters):
        # find all voters who have not delegated to anyone and how much weight they have

        gurus = dict()
        for voter in voters:
            if voter not in self.delegations.keys():
                gurus[voter] = 1
        for guru in gurus:
            gurus[guru] = self.count_guru_weight(guru, self.delegations)

        return gurus

    def count_guru_weight(self, guru, delegations):
        # count the number of voters that have delegated to this guru
        weight = 1
        for voter in delegations.keys():
            if delegations[voter] == guru:
                weight += self.count_guru_weight(voter, delegations)
        return weight


class UCBDelegationMechanism(DelegationMechanism):
    def __init__(self, batch_size, window_size=None, ucb_window_size=500):
        super().__init__(batch_size, window_size)
        self.ucb_window_size = ucb_window_size  # used to calculate a ucb score that does not lose strength over time

    def update_delegations(self, ensemble_accs, voters, train, t_increment=None):
        if t_increment:
            self.t += t_increment

        # if train is true dont do anything
        if train:
            return
        else:
            # first, we need to recalculate the CI for each voter
            for voter in voters:
                voter.ucb_score = ucb(voter, self.t, self.ucb_window_size)

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


def ucb(window_size, voter, t, ucb_window_size, c=3.0):
    """
    Calculate upper confidence bound of the bandit arm corresponding to voting directly. Loosely speaking, if this
    is high enough the voter will vote directly.
    point_wise_accuracies is the number of samples this voter has taken, i.e. n_i in UCB terms
    t is the total number of samples taken by any agent, i.e. N in UCB terms

    To be clear, window_size is used for the mean calculation, and ucb_window_size is used for the ucb bonus (decoupled bonus from mean)

    :param t: number of time steps passed
    :param c: exploration term; higher means more exploration/higher chance of voting directly (?)
    """
    if window_size is None:  # Using the full point wise accuracies
        point_wise_accuracies = (
            voter.accuracy
        )  # one value per sample that this voter has predicted upon
        # t_window = t  # total number of possible data points within the window
        mean = np.mean(point_wise_accuracies)  # mean accuracy/reward of arm pulls
    else:
        # get the most recent window_size predictions and take the mean
        if len(voter.accuracy) < window_size:
            mean = np.mean(voter.accuracy)
        else:
            mean = np.mean(voter.accuracy[-window_size:])

    if t < ucb_window_size:
        total_t = t
        n_t = len(voter.accuracy)
    else:
        total_t = ucb_window_size
        n_t = sum(
            voter.binary_active[-ucb_window_size:]
        )  # number of arm pulls in the last ucb_window_size examples

    # n_t = len(voter.accuracy)  # number of arm pulls the voter has taken

    fudge_factor = 1e-8

    ucb = mean + np.sqrt(c * np.log(total_t) / (n_t + fudge_factor))
    # ucb = mean + np.sqrt(c * np.log(t_window) / (n_t + fudge_factor))

    return ucb
