import numpy as np
from scipy import stats
from sklearn.preprocessing import maxabs_scale
from .Voter import *


class DelegationMechanism:
    def __init__(self, batch_size, window_size=None, verbose=False, score_method="accuracy_score"):
        self.delegations = {}  # key: delegate_from (id), value: delegate_to (id)
        self.t = 0
        self.window_size = window_size
        self.batch_size = batch_size
        self.verbose = verbose

        # TODO: Currently this is only used by ProbaSlopeDelegationMechanism
        # I just wanted it to be accessible more broadly so Voters can see which score they should compute
        self.score_method = score_method

    def add_delegation(self, from_id, to_id):
        """
        Create a new delegation from one voter to another which also removes any old delegations the new delegator had.
        Currently, do not bother considering cycles or any reason why a delegation may be impossible.
        Args:
            from_id (_type_): _description_
            to_id (_type_): _description_
        """
        if self.verbose:
            print(f"Making delegation from {from_id} to {to_id} at t={self.t}.")
        self.delegations[from_id] = to_id

    def remove_delegation(self, v_id):
        print(f"Removing delegation {v_id} made.")
        self.delegations.pop(v_id, None)

    def voter_is_active(self, v: Voter):
        # Voter v is active if they are NOT making a delegation. That is, if they are not a key in the delegation dict.
        return v.id not in self.delegations.keys()

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
            if voter.id not in self.delegations.keys():
                gurus.append(voter)
        return gurus

    def get_gurus_with_weights(self, voters):
        # find all voters who have not delegated to anyone and how much weight they have

        gurus = self.get_gurus(voters)
        guru_weights = dict()
        for guru in gurus:
            guru_weights[guru] = self.count_guru_weight(guru)

        return guru_weights

    def count_guru_weight(self, guru):
        # count the number of voters that have delegated to this guru
        weight = 1
        for voter in self.delegations.keys():
            if self.get_guru_of_voter(voter) == guru:
                weight += 1
            # if delegations[voter] == guru:
            #     weight += self.count_guru_weight(voter, delegations)
        return weight

    def get_guru_of_voter(self, v_id):
        # Recursively trace through delegations until the guru of this voter is found.
        # A voter is a guru if they do not appear as a key in self.delegations
        if v_id not in self.delegations:
            return v_id
        else:
            delegatee = self.delegations[v_id]
            while delegatee in self.delegations:
                delegatee = self.delegations[delegatee]
            return delegatee


class UCBDelegationMechanism(DelegationMechanism):
    def __init__(self, batch_size, window_size=None, ucb_window_size=500):
        super().__init__(batch_size, window_size)
        self.ucb_window_size = ucb_window_size  # used to calculate a ucb score that does not lose strength over time

    def update_delegations(self, ensemble_accs, voters, train, t_increment=None):
        if t_increment:
            self.t += t_increment

        # if train is true, we want to be training k classifiers per digit group
        if train:
            return
            # say we have 10 voters and somehow we know there are 2 digit groups (0-4, 5-9)
            # we want to train 5 voters per digit group

        else:
            # first, we need to recalculate the CI for each voter
            for voter in voters:
                voter.ucb_score = ucb(
                    self.window_size,
                    voter,
                    self.t,
                    self.ucb_window_size,
                    self.batch_size,
                )

            # print("Delegations: ", self.delegations)

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
                    self.add_delegation(voter, delegee)


class RestrictedMaxGurusDelegationMechanism(DelegationMechanism):
    def __init__(
        self,
        batch_size,
        num_voters,
        max_active_voters=1,
        window_size=None,
        t_between_delegation=3,
    ):
        super().__init__(batch_size, window_size)

        self.max_active_voters = max_active_voters
        self.t_between_delegation = t_between_delegation  # minimum number of timesteps that should pass between any delegations
        self.most_recent_delegation_time = 0

        # Each voter should delegate so that only max_active_voters remain active
        # Assumes all voters have consecutive ids :/
        self.untrained_voters = [i for i in range(max_active_voters, num_voters)]
        for from_id in range(max_active_voters, num_voters):
            to_id = from_id % max_active_voters
            self.add_delegation(from_id=from_id, to_id=to_id)

    def update_delegations(self, ensemble_accs, voters, train, t_increment=None):
        should_delegate = []
        inactive_voters = []
        for v in voters:
            # num_active_voters = self.get_gurus()

            if self.voter_is_active(v):
                """
                If acc_v is high and staying high compared with previous time steps:
                    v stays active by doing nothing
                If acc_v is high and falling compared with previous time steps:
                    v should delegate
                        maybe select the least accurate classifier to maximize chance it is untrained?
                        or, set a flag to make v delegate and then if any other classifier is already quite high accuracy let them be active?
                If acc_v is low (shouldn't happen?):
                    v should delegate
                """
                # if voter_has_improved_linreg(t_now=self.t, t_then=self.t - self.window_size, voter_batch_accs=v.batch_accuracies):
                slope = linear_regression_slope(
                    t_now=self.t,
                    t_before=self.t - self.window_size,
                    score_values=v.batch_accuracies,
                )
                # print(f"Linear regression slope is: {slope} at time {self.t}")
                if slope > -0.1:
                    # voter should remain active if they are continuing to learn
                    pass
                else:
                    # print(
                    #     f"Linear regression slope is: {slope} at time {self.t} so {v} is delegating."
                    # )
                    should_delegate.append(v)
            else:  # v is inactive, we rely on all voters always tracking their score
                """
                if acc_v is low and not improving:
                    v should remain inactive by doing nothing
                if acc_v is high and not changing:
                    for now, do nothing.
                    later, maybe consider acc_v in comparison to group accuracy or something
                if acc_v is low and improving:
                    v should become active (this probably means the classifier is already trained?)
                """
                inactive_voters.append(v)

        # # sort inactive voters by ascending accuracy to maximize chance of getting an untrained clf
        # inactive_voters.sort(key=lambda x: x.batch_accuracies[-1])
        # sort inactive voters by descending accuracy; minimizes duplicate training in case of repeated classes?
        inactive_voters.sort(key=lambda x: x.batch_accuracies[-1], reverse=True)
        potential_guru_ids = self.untrained_voters + [ia.id for ia in inactive_voters]

        # Have each delegating voter delegate to the next least accurate delegator
        for i in range(len(should_delegate)):
            # new_guru = inactive_voters[i]
            new_guru = potential_guru_ids[i]
            new_delegator = should_delegate[i]

            self.remove_delegation(new_guru)
            self.add_delegation(new_delegator.id, new_guru)

            if new_guru in self.untrained_voters:
                self.untrained_voters.remove(new_guru)

        return super().update_delegations(ensemble_accs, voters, train, t_increment)


class ProbaSlopeDelegationMechanism(DelegationMechanism):
    """
    On each batch, we check the slope of recent voter accuracy
    Voters can only delegate to other voters who have a higher slope
    Delegation is probabilistic based on the difference in slope
    """


    def __init__(self, batch_size, window_size=None, verbose=False, max_active=1, probability_function="probabilistic_better", score_method="accuracy_score"):
        super().__init__(batch_size, window_size, verbose=verbose, score_method=score_method)
        self.max_active = max_active
        self.probability_function = probability_function

    def update_delegations(self, ensemble_accs, voters, train, t_increment=None):
        if train:
            # print(self.delegations)
            # First we will calculate all slopes
            slopes = dict()
            for voter in voters:
                # slope = linear_regression_slope(
                #     t_now=self.t,
                #     t_before=self.t - self.window_size,
                #     score_values=voter.batch_accuracies,
                # )
                slope = linear_regression_slope(
                    t_now=self.t,
                    t_before=self.t - self.window_size,
                    score_values=voter.metric_scores[self.score_method],
                )

                # print(f"Slope for voter {voter.id} at time {self.t}: ", slope)
                slopes[voter.id] = slope
                # print(self.t)
            
            # get the k highest slopes so we can make sure they are not delegating
            slopes = dict(sorted(slopes.items(), key=lambda item: item[1], reverse=True))
            best_k_ids = list(slopes.keys())[:self.max_active]
            delegators_to_pop = best_k_ids

            # now we need to do two things:
            # 1. ensure all current delegations are still valid. If not, remove them
            # 2. go through the full delegation process
            # delegators_to_pop = []
            for (
                delegator,
                delegee,
            ) in self.delegations.items():
                if slopes[delegator.id] > slopes[delegee.id]:
                    delegators_to_pop.append(delegator)
            for delegator in delegators_to_pop:
                if delegator in self.delegations:
                    self.delegations.pop(delegator)

            for voter in voters:  # go through the full delegation process
                if voter.id in best_k_ids:
                    # make sure that the best max_active voters do not delegate
                    continue
                possible_delegees = []
                gaps = []
                for other_voter in voters:
                    # find all other voters who have a higher slope
                    if other_voter.id != voter.id and (
                        slopes[other_voter.id] > slopes[voter.id]
                    ):
                        possible_delegees.append(other_voter)
                        gaps.append(slopes[other_voter.id] - slopes[voter.id])

                # delegate probabilistically based on the gaps. Larger gaps are more likely to be chosen
                # Can explore a variety of delegation functions
                if len(possible_delegees) > 0:
                    
                    if self.probability_function == "random_better":
                        # delegate to a higher-slope voter uniformly at random
                        probabilities = [1 for _ in gaps]
                    elif self.probability_function == "probabilistic_better":
                        # delegate to higher-slope voters proportionally to their difference in slope
                        sum_gaps = sum(gaps)
                        probabilities = [gap / sum_gaps for gap in gaps]
                    elif self.probability_function == "probabilistic_weighted":
                        # get weight of each voters guru, use it to scale delegation probabilities
                        weights = [self.count_guru_weight(self.get_guru_of_voter(g)) for g in possible_delegees]
                        sum_gaps = sum(gaps)
                        probabilities = [(1/weights[idx]) * (gaps[idx]/sum_gaps) for idx in range(len(gaps))]
                    elif self.probability_function == "max_diversity":
                        # Each voter should delegate to the more accurate voter who is closest to themselves
                        recent_probas = voter.batch_probas[-1]
                        dists = [np.linalg.norm(g.batch_probas[-1]-recent_probas) for g in possible_delegees]
                        closest = np.argmin(dists)
                        probabilities = [0 for _ in range(len(possible_delegees))]
                        probabilities[closest] = 1
                    # normalize probabilities before selecting delegee
                    raw_sum = sum(probabilities)
                    probabilities = [p/raw_sum for p in probabilities]

                    delegee = np.random.choice(possible_delegees, p=probabilities)
                    self.add_delegation(voter, delegee)

        else:
            # break all delegations
            self.delegations = {}

        if t_increment:
            self.t += t_increment


class StudentExpertDelegationMechanism:
    def __init__(self, batch_size, window_size=None, verbose=False):
        # self.delegations = {}  # key: delegate_from (id), value: delegate_to (id)
        self.t = 0
        self.window_size = window_size
        self.batch_size = batch_size
        self.verbose = verbose
        self.student_delegations = {}
        self.expert_delegations = {}

    def add_student_delegation(self, from_id, to_id):
        """
        Create a new delegation from one voter to another which also removes any old delegations the new delegator had.
        Currently, do not bother considering cycles or any reason why a delegation may be impossible.
        Args:
            from_id (_type_): _description_
            to_id (_type_): _description_
        """
        if self.verbose:
            print(f"Making delegation from {from_id} to {to_id} at t={self.t}.")
        self.student_delegations[from_id] = to_id

    def add_expert_delegation(self, from_id, to_id):
        """
        Create a new delegation from one voter to another which also removes any old delegations the new delegator had.
        Currently, do not bother considering cycles or any reason why a delegation may be impossible.
        Args:
            from_id (_type_): _description_
            to_id (_type_): _description_
        """
        if self.verbose:
            print(f"Making delegation from {from_id} to {to_id} at t={self.t}.")
        self.expert_delegations[from_id] = to_id

    def remove_student_delegation(self, v_id):
        print(f"Removing student delegation {v_id} made.")
        self.student_delegations.pop(v_id, None)

    def remove_expert_delegation(self, v_id):
        print(f"Removing expert delegation {v_id} made.")
        self.expert_delegations.pop(v_id, None)

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

        if train:
            # print(self.delegations)
            # First we will calculate all slopes
            slopes = dict()
            for voter in voters:
                slope = linear_regression_slope(
                    t_now=self.t,
                    t_before=self.t - self.window_size,
                    score_values=voter.batch_accuracies,
                )

                # print(f"Slope for voter {voter.id} at time {self.t}: ", slope)
                slopes[voter.id] = slope
                # print(self.t)

            # now we need to do two things:
            # 1. ensure all current delegations are still valid. If not, remove them
            # 2. go through the full delegation process
            delegators_to_pop = []
            for (
                delegator,
                delegee,
            ) in self.student_delegations.items():
                if slopes[delegator.id] > slopes[delegee.id]:
                    delegators_to_pop.append(delegator)
            for delegator in delegators_to_pop:
                self.student_delegations.pop(delegator)

            for voter in voters:  # go through the full delegation process
                possible_delegees = []
                gaps = []
                for other_voter in voters:
                    # find all other voters who have a higher slope
                    if other_voter.id != voter.id and (
                        slopes[other_voter.id] > slopes[voter.id]
                    ):
                        possible_delegees.append(other_voter)
                        gaps.append(slopes[other_voter.id] - slopes[voter.id])
                if len(possible_delegees) > 0:
                    # probabilistically delegate based on the gaps
                    # larger gaps are more likely to be chosen
                    sum_gaps = sum(gaps)
                    probabilities = [gap / sum_gaps for gap in gaps]
                    delegee = np.random.choice(possible_delegees, p=probabilities)
                    self.add_student_delegation(voter, delegee)

        else:
            # no students in test phase # NOTE: the empty dict technically means everyone is a student so this isnt quite correct
            # however it shouldnt matter since we dont use the student_delegations dict in test phase
            self.student_delegations = {}

        # We will always be updating the expert delegations
        # we will just use the previous batch accuracy to determine the probability of delegation

        previous_batch_accs = dict()
        for voter in voters:
            if len(voter.batch_accuracies) > 0:
                previous_batch_accs[voter.id] = voter.batch_accuracies[-1]
            else:
                previous_batch_accs[voter.id] = 0

        # now we need to do two things:
        # 1. ensure all current delegations are still valid. If not, remove them
        # 2. go through the full delegation process
        delegators_to_pop = []
        for (
            delegator,
            delegee,
        ) in self.expert_delegations.items():
            if previous_batch_accs[delegator.id] > previous_batch_accs[delegee.id]:
                delegators_to_pop.append(delegator)
        for delegator in delegators_to_pop:
            self.expert_delegations.pop(delegator)

        for voter in voters:  # go through the full delegation process
            possible_delegees = []
            gaps = []
            for other_voter in voters:
                # find all other voters who have a higher accuracy on the last batch
                if other_voter.id != voter.id and (
                    previous_batch_accs[other_voter.id] > previous_batch_accs[voter.id]
                ):
                    possible_delegees.append(other_voter)
                    gaps.append(
                        previous_batch_accs[other_voter.id]
                        - previous_batch_accs[voter.id]
                    )
            if len(possible_delegees) > 0:
                # convert gaps to a list of floats
                gaps = [float(gap) for gap in gaps]

                sum_gaps = sum(gaps)
                probabilities = [gap / sum_gaps for gap in gaps]

                delegee = np.random.choice(possible_delegees, p=probabilities)
                self.add_expert_delegation(voter, delegee)

        if t_increment:
            self.t += t_increment

    def count_expert_guru_weight(self, guru, delegations):
        # count the number of voters that have delegated to this guru
        weight = 1
        for voter in delegations.keys():
            if self.get_expert_guru_of_voter(voter) == guru:
                weight += 1
            # if delegations[voter] == guru:
            #     weight += self.count_guru_weight(voter, delegations)
        return weight

    def get_expert_gurus(self, voters):
        # find all voters who have not delegated to anyone
        gurus = []

        for voter in voters:
            if voter.id not in self.expert_delegations.keys():
                gurus.append(voter)
        return gurus

    def get_student_gurus(self, voters):
        # find all voters who have not delegated to anyone
        gurus = []
        for voter in voters:
            if voter.id not in self.student_delegations.keys():
                gurus.append(voter)
        return gurus

    def get_gurus(self, voters, expert=False):
        if expert:
            return self.get_expert_gurus(voters)
        else:
            return self.get_student_gurus(voters)

    def get_expert_guru_of_voter(self, v_id):
        # Recursively trace through delegations until the guru of this voter is found.
        # A voter is a guru if they do not appear as a key in self.delegations
        if v_id not in self.expert_delegations:
            return v_id
        else:
            delegatee = self.expert_delegations[v_id]
            while delegatee in self.expert_delegations:
                delegatee = self.expert_delegations[delegatee]
            return delegatee

    def get_expert_gurus_with_weights(self, voters):
        # find all voters who have not delegated to anyone and how much weight they have

        gurus = self.get_expert_gurus(voters)
        guru_weights = dict()
        for guru in gurus:
            guru_weights[guru] = self.count_expert_guru_weight(
                guru, self.expert_delegations
            )

        return guru_weights

    def voter_is_student(self, v: Voter):
        # Voter v is a student if they are NOT making a student delegation. That is, if they are not a key in the student_delegations dict.
        return v.id not in self.student_delegations.keys()
    
    def voter_is_active(self, v: Voter):
        # Voter v is active if they are NOT making a delegation. That is, if they are not a key in the delegation dict.
        return self.voter_is_student(v)


class StudentExpertDelegationMechanismExperimental(StudentExpertDelegationMechanism):
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

        if train:
            # print(self.delegations)
            # First we will calculate all slopes
            metric = dict()
            slopes = dict()
            avg_accs = dict()
            for voter in voters:
                slope = linear_regression_slope(
                    t_now=self.t,
                    t_before=self.t - self.window_size,
                    score_values=voter.batch_accuracies,
                )
                slopes[voter.id] = slope

                # if len(voter.batch_accuracies) > self.window_size:
                #     avg_acc = np.mean(voter.batch_accuracies[-self.window_size :])
                # else:
                #     avg_acc = np.mean(voter.batch_accuracies)
                avg_acc = voter.batch_accuracies[-1]

                avg_accs[voter.id] = avg_acc

                # metric[voter.id] = slope * avg_acc
                # metric[voter.id] = slope + avg_acc
                # metric[voter.id] = avg_acc
                metric[voter.id] = slope
                # metric[voter.id] = avg_acc / (slope + 1e-8)

            # now we need to do two things:
            # 1. ensure all current delegations are still valid. If not, remove them
            # 2. go through the full delegation process
            delegators_to_pop = []
            for (
                delegator,
                delegee,
            ) in self.student_delegations.items():
                if metric[delegator.id] > metric[delegee.id]:
                    delegators_to_pop.append(delegator)
            for delegator in delegators_to_pop:
                self.student_delegations.pop(delegator)

            for voter in voters:  # go through the full delegation process
                possible_delegees = []
                gaps = dict()
                for other_voter in voters:
                    # find all other voters who have a higher slope
                    if other_voter.id != voter.id and (
                        metric[other_voter.id] > metric[voter.id]
                    ):
                        possible_delegees.append(other_voter)
                        gaps[other_voter.id] = metric[other_voter.id] - metric[voter.id]
                if len(possible_delegees) > 0:
                    # turn gaps into a list of probabilities (normalize)
                    gaps = [float(gap) for gap in gaps.values()]
                    sum_gaps = sum(gaps)
                    probas = [gap / sum_gaps for gap in gaps]

                    # scale probas by constant 0 < c < 1
                    c = 0.5
                    # for k, v in probas.items():
                    #     probas[k] = c * v

                    # probabilistically delegate based on the gaps
                    # shuffle the order of the possible delegees to avoide favouring earlier voters
                    # np.random.shuffle(possible_delegees)
                    # for possible_delegee in possible_delegees:
                    #     proba = probas[possible_delegee.id]
                    #     if np.random.rand() < proba:
                    #         self.add_student_delegation(voter, possible_delegee)
                    #         break

                    # sample a random rng value
                    if np.random.rand() < c:
                        # sample a random voter
                        delegee = np.random.choice(possible_delegees, p=probas)
                        self.add_student_delegation(voter, delegee)

        else:
            # no students in test phase # NOTE: the empty dict technically means everyone is a student so this isnt quite correct
            # however it shouldnt matter since we dont use the student_delegations dict in test phase
            self.student_delegations = {}

        # We will always be updating the expert delegations
        # we will just use the previous batch accuracy to determine the probability of delegation

        previous_batch_accs = dict()
        for voter in voters:
            if len(voter.batch_accuracies) > 0:
                previous_batch_accs[voter.id] = voter.batch_accuracies[-1]
            else:
                previous_batch_accs[voter.id] = 0

        # now we need to do two things:
        # 1. ensure all current delegations are still valid. If not, remove them
        # 2. go through the full delegation process
        delegators_to_pop = []
        for (
            delegator,
            delegee,
        ) in self.expert_delegations.items():
            if previous_batch_accs[delegator.id] > previous_batch_accs[delegee.id]:
                delegators_to_pop.append(delegator)
        for delegator in delegators_to_pop:
            self.expert_delegations.pop(delegator)

        for voter in voters:  # go through the full delegation process
            possible_delegees = []
            gaps = []
            for other_voter in voters:
                # find all other voters who have a higher accuracy on the last batch
                if other_voter.id != voter.id and (
                    previous_batch_accs[other_voter.id] > previous_batch_accs[voter.id]
                ):
                    possible_delegees.append(other_voter)
                    gaps.append(
                        previous_batch_accs[other_voter.id]
                        - previous_batch_accs[voter.id]
                    )
            if len(possible_delegees) > 0:
                # convert gaps to a list of floats
                gaps = [float(gap) for gap in gaps]

                sum_gaps = sum(gaps)
                probabilities = [gap / sum_gaps for gap in gaps]

                delegee = np.random.choice(possible_delegees, p=probabilities)
                self.add_expert_delegation(voter, delegee)

        if t_increment:
            self.t += t_increment


def voter_has_improved(t_now, t_then, voter_batch_accs):
    """
    Return True iff this voter is more accurate than it was at the given time. Return False if the voter is not more accurate
    or if there is not that much history.
    # TODO: Add some other argument like accuracy_metric to allow more reasonable checks of accuracy.
    Args:
        t_now int: Probably redundant; current time step. Probably refers to the final value in voter_batch_accs
        t_steps_back int: Past time step to compare against.
        voter_batch_accs list: _description_
    """
    if t_then < 0 or t_now >= len(voter_batch_accs) or t_then >= t_now:
        return False
    return voter_batch_accs[t_now] > voter_batch_accs[t_then]


def voter_has_improved_linreg(t_now, t_then, voter_batch_accs):
    """
    Return True iff this voter's accuracy is trending upwards. Return False if the voter's accuracy is trending
    down or if there is enough that much history.
    Args:
        t_now int: Probably redundant; current time step. Probably refers to the final value in voter_batch_accs
        t_steps_back int: Past time step to compare against.
        voter_batch_accs list: _description_
    """
    if t_then < 0 or t_now >= len(voter_batch_accs) or t_then >= t_now:
        return False
    recent_accs = voter_batch_accs[t_then : t_now + 1]
    lr = stats.linregress(range(len(recent_accs)), recent_accs)
    # print(f"Lin Regression result is {lr}")
    return lr.slope >= 0


def linear_regression_slope(t_now, t_before, score_values):
    """
    Return the slope of linear regression done over the given window on the given accuracies.

    Args:
        t_now (_type_): _description_
        t_then (_type_): _description_
        voter_batch_accs (_type_): _description_
    """
    if t_before < 0 or t_now >= len(score_values) or t_before >= t_now:
        return 0
    recent_accs = score_values[t_before : t_now + 1]
    lr = stats.linregress(range(len(recent_accs)), recent_accs)
    # print(f"Lin Regression result is {lr}")
    return lr.slope


def get_batch_accuracy_slope(t_now, t_before, voter_batch_accs):
    """
    Return the slope of linear regression done over the given window on the given accuracies.

    NOTE: Same as Ben's above aside from changing the t_now >= len(voter_batch_accs) to t_now > len(voter_batch_accs)
    NOTE: Just realized it depends on when you update t by t_increment in update_delegations

    Args:
        t_now (_type_): _description_
        t_then (_type_): _description_
        voter_batch_accs (_type_): _description_
    """
    if t_before < 0 or t_now > len(voter_batch_accs) or t_before >= t_now:
        return 0
    recent_accs = voter_batch_accs[t_before : t_now + 1]
    lr = stats.linregress(range(len(recent_accs)), recent_accs)
    # print(f"Lin Regression result is {lr}")
    return lr.slope


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


def ucb(window_size, voter, t, ucb_window_size, batch_size, c=3.0):
    """
    Calculate upper confidence bound of the bandit arm corresponding to voting directly. Loosely speaking, if this
    is high enough the voter will vote directly.
    point_wise_accuracies is the number of samples this voter has taken, i.e. n_i in UCB terms
    t is the total number of samples taken by any agent, i.e. N in UCB terms

    To be clear, window_size is used for the mean calculation, and ucb_window_size is used for the ucb bonus (decoupled bonus from mean).
    So the mean could be calculated over the last 1000 samples, but the ucb bonus could be based on the number of times the voter was active in the last 500 samples.

    :param window_size: number of batches to use for calculating the mean
    :param voter: voter to calculate ucb for
    :param t: number of time steps passed
    :param ucb_window_size: number of batches to use for calculating the ucb bonus
    :param batch_size: number of samples per batch
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
        if len(voter.accuracy) < window_size * batch_size:
            mean = np.mean(voter.accuracy)
        else:
            mean = np.mean(voter.accuracy[-window_size * batch_size :])
    if ucb_window_size is None:
        total_t = t * batch_size
        n_t = len(voter.accuracy)
    elif t < ucb_window_size:
        total_t = t * batch_size
        n_t = len(voter.accuracy)
    else:
        total_t = ucb_window_size * batch_size
        n_t = sum(
            voter.binary_active[-ucb_window_size:]
        )  # number of arm pulls in the last ucb_window_size examples

    fudge_factor = 1e-8  # to avoid division by zero

    ucb = mean + np.sqrt(c * np.log(total_t) / (n_t + fudge_factor))

    return ucb
