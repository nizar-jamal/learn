import numpy as np
import csv
import os

from collections import Counter

from abc import ABC, abstractmethod
from utils.data_preprocessing import load_data, dose_class, LABEL_KEY


# Base classes
class BanditPolicy(ABC):
    @abstractmethod
    def extract_features(self, x, features):
        pass

    @abstractmethod
    def choose(self, x):
        pass

    @abstractmethod
    def update(self, x, a, r):
        pass


class StaticPolicy(BanditPolicy):
    def extract_features(self, x, features):
        pass

    def update(self, x, a, r):
        pass


class RandomPolicy(StaticPolicy):
    def __init__(self, probs=None):
        self.probs = probs if probs is not None else [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    def choose(self, x):
        return np.random.choice(range(len(self.probs)), p=self.probs)


############################################################
# Problem 1: Estimation of Warfarin Dose
############################################################

############################################################
# Problem 1a: baselines


class FixedDosePolicy(StaticPolicy):
    def choose(self, x):
        """
        Args:
                x: numpy array of features
        Returns:
                output: index of the chosen action

        TODO:
                - Please implement the fixed dose which is to assign medium dose
                  to all patients.
        """
        ### START CODE HERE ###
        return dose_class(35)   # medium dose 
        ### END CODE HERE ###


class ClinicalDosingPolicy(StaticPolicy):

    def extract_features(self, x):
        """
        Args:
                x (dict): dictionary containing the possible patient features.

        Returns:
                x (float): containing the square root of the weekly warfarin dose

        TODO:
                - Prepare the features to be used in the clinical model
                  (consult section 1f of appx.pdf for feature definitions)

        Hint:
                - Look at the utils/data_preprocessing.py script to see the key values
                  of the features you can use. The age in decades is implemented for
                  you as an example.
                - You can treat Unknown race as missing or mixed race.

        """
        weekly_dose_sqrt = None
        age_in_decades = x["Age in decades"]
        enzyme_inducer_status = x["Carbamazepine (Tegretol)"] or x["Phenytoin (Dilantin)"] or x["Rifampin or Rifampicin"]
        ### START CODE HERE ###
        weekly_dose_sqrt = 4.0376 \
                - 0.2546 * x["Age in decades"] \
                + 0.0118 * x["Height (cm)"]\
                + 0.0134 * x["Weight (kg)"] \
                - 0.6752 * x["Asian"] \
                + 0.4060 * x["Black"] \
                - 0.1213 * x["Unknown race"] \
                - 0.3497 * enzyme_inducer_status \
                + 0.2791 * x["Amiodarone (Cordarone)"]
        ### END CODE HERE ###

        return weekly_dose_sqrt

    def choose(self, x):
        """
        Args:
                x (dict): dictionary containing the possible patient features.
        Returns:
                output: index of the chosen action

        TODO:
                - Create a linear model based on the values in section 1f
                  and return its output based on the input features

        Hint:
                - Use dose_class() implemented for you.
        """

        weekly_dose_sqrt = self.extract_features(x)
        ### START CODE HERE ###
        return dose_class(weekly_dose_sqrt ** 2)
        ### END CODE HERE ###


############################################################
# Problem 1b: upper confidence bound linear bandit


class LinUCB(BanditPolicy):
    def __init__(self, num_arms, features, alpha=1.0):
        """
        See Algorithm 1 from paper:
                "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
                num_arms (int): the initial number of different arms / actions the algorithm can take
                features (list of str): contains the features to use
                alpha (float): hyperparameter for step size.

        TODO:
                - Please initialize the following internal variables for the Disjoint Linear UCB Bandit algorithm:
                        * self.features
                        * self.d
                        * self.alpha
                        * self.A
                        * self.b
                  These terms align with the paper, please refer to the paper to understand what they are.
                  Feel free to add additional internal variables if you need them, but they are not necessary.

        Hint:
                Keep track of a seperate A, b for each action (this is what the Disjoint in the algorithm name means)
        """
        ### START CODE HERE ###
        self.features = features
        self.d = len(features)
        self.alpha = alpha
        self.A = {a: np.eye(self.d) for a in range(num_arms)}
        self.b = {a: np.zeros(self.d) for a in range(num_arms)}

        ### END CODE HERE ###

    def extract_features(self, x):
        """
        Args:
                x (dict): dictionary containing the possible features.

        Returns:
                out: numpy array of features
        """

        return np.array([x[f] for f in self.features])

    def choose(self, x):
        """
        See Algorithm 1 from paper:
                "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
                x:
                 - (dict): dictionary containing the possible features.
                 or
                 - (numpy array): array of features
        Returns:
                output: index of the chosen action

        Please implement the "forward pass" for Disjoint Linear Upper Confidence Bound Bandit algorithm.
        """

        #######################################################
        xvec = x
        if type(x) is dict:
            xvec = self.extract_features(x)

        #########   ~5 lines.   #############
        ### START CODE HERE ###
        theta = {a: np.linalg.inv(self.A[a]).dot(self.b[a]) for a in range(len(self.A))}
        p = {a: theta[a].dot(xvec) + self.alpha * np.sqrt(xvec.dot(np.linalg.inv(self.A[a])).dot(xvec)) for a in range(len(self.A))}
        return max(p, key=p.get)
        ### END CODE HERE ###
        #######################################################

    def update(self, x, a, r):
        """
        Args:
                x:
                 - (dict): dictionary containing the possible features.
                 or
                 - (numpy array): array of features
                a: integer, indicating the action your algorithm chose
                r: the reward you received for that action
        Returns:
                Nothing

        Please implement the update step for Disjoint Linear Upper Confidence Bound Bandit algorithm.
        """

        xvec = x
        if type(x) is dict:
            xvec = self.extract_features(x)

        ### START CODE HERE ###
        self.A[a] += np.outer(xvec, xvec)
        self.b[a] += r * xvec
        ### END CODE HERE ###


############################################################
# Problem 1c: eGreedy linear bandit


class eGreedyLinB(LinUCB):
    def __init__(self, num_arms, features, alpha=1.0):
        super(eGreedyLinB, self).__init__(num_arms, features, alpha)
        self.time = 0

    def choose(self, x):
        """
        Args:
                x (dict): dictionary containing the possible features.
        Returns:
                output: index of the chosen action

        TODO:
                - Instead of using the Upper Confidence Bound to find which action to take,
                  compute the payoff of each action using a simple dot product between Theta & the input features.
                  Then use an epsilon greedy algorithm to choose the action.
                  Use the value of epsilon provided and np.random.uniform() in your implementation.
        """

        self.time += 1
        epsilon = float(1.0 / self.time) * self.alpha
        xvec = self.extract_features(x)

        ### START CODE HERE ###
        p = {a: self.b[a].dot(xvec) for a in range(len(self.A))}
        if np.random.uniform() > epsilon:
            return max(p, key=p.get)
        return np.random.choice(range(len(self.A)))
        ### END CODE HERE ###


############################################################
# Problem 1d: Thompson sampling


class ThomSampB(BanditPolicy):
    def __init__(self, num_arms, features, alpha=1.0):
        """
        See Algorithm 1 and section 2.2 from paper:
                "Thompson Sampling for Contextual Bandits with Linear Payoffs"

        Args:
                num_arms (int): the initial number of different arms / actions the algorithm can take
                features (list of str): contains the features to use
                alpha (float): hyperparameter for step size.

        TODO:
                - Please initialize the following internal variables for the Thompson sampling bandit algorithm:
                        * self.features
                        * self.num_arms
                        * self.d
                        * self.v2 (please set this term equal to alpha)
                        * self.B
                        * self.mu
                        * self.f
                These terms align with the paper, please refer to the paper to understand what they are.
                Please feel free to add additional internal variables if you need them, but they are not necessary.

        Hints:
                - Keep track of a separate B, mu, f for each action (this is what the Disjoint in the algorithm name means)
                - Unlike in section 2.2 in the paper where they sample a single mu_tilde, we'll sample a mu_tilde for each arm
                        based on our saved B, f, and mu values for each arm. Also, when we update, we only update the B, f, and mu
                        values for the arm that we selected
                - What the paper refers to as b in our case is the medical features vector
                - The paper uses a summation (from time =0, .., t-1) to compute the model parameters at time step (t),
                        however if you can't access prior data how might one store the result from the prior time steps.

        """
        ### START CODE HERE ###
        self.features = features
        self.num_arms = num_arms
        self.d = len(features)
        self.v2 = alpha
        self.B = {a: np.eye(self.d) for a in range(num_arms)}
        self.mu = {a: np.zeros(self.d) for a in range(num_arms)}
        self.f = {a: np.zeros(self.d) for a in range(num_arms)}
        ### END CODE HERE ###

    def extract_features(self, x):
        """
        Args:
                x (dict): dictionary containing the possible features.

        Returns:
                out: numpy array of features
        """

        return np.array([x[f] for f in self.features])

    def choose(self, x):
        """
        See Algorithm 1 and section 2.2 from paper:
                "Thompson Sampling for Contextual Bandits with Linear Payoffs"

        Args:
                x (dict): dictionary containing the possible features.
        Returns:
                output: index of the chosen action

        TODO:
                - Please implement the "forward pass" for Disjoint Thompson Sampling Bandit algorithm.
                - Please use np.random.multivariate_normal to simulate the multivariate gaussian distribution in the paper.
        """

        xvec = self.extract_features(x)

        ### START CODE HERE ###
        self.mu = {a: np.random.multivariate_normal(self.mu[a], np.linalg.inv(self.v2 * self.B[a])) for a in range(self.num_arms)}
        a = {a: self.mu[a].dot(xvec) for a in range(self.num_arms)}
        return max(a, key=a.get)
        ### END CODE HERE ###

    def update(self, x, a, r):
        """
        See Algorithm 1 and section 2.2 from paper:
                "Thompson Sampling for Contextual Bandits with Linear Payoffs"

        Args:
                x (dict): dictionary containing the possible features.
                a: integer, indicating the action your algorithm chose
                r (int): the reward you received for that action

        TODO:
                - Please implement the update step for Disjoint Thompson Sampling Bandit algorithm.

        Hint:
                Which parameters should you update?
        """

        xvec = self.extract_features(x)

        ### START CODE HERE ###
        self.B[a] += np.outer(xvec, xvec)
        self.f[a] += r * xvec
        self.mu[a] = np.linalg.inv(self.B[a]).dot(self.f[a])
        ### END CODE HERE ###


############################################################
# Problem 2: Recommender system simulator
############################################################


############################################################
# Problem 2a: LinUCB with increasing number of arms


class DynamicLinUCB(LinUCB):

    def add_arm_params(self):
        """
        Add a new A and b for the new arm we added.
        Initialize them in the same way you did in the __init__ method
        """
        #######################################################
        #########  ~2 lines.   #############
        ### START CODE HERE ###
        ### END CODE HERE ###


class Simulator:
    """
    Simulates a recommender system setup where we have say A arms corresponding to items and U users initially.
    The number of users U cannot change but the number of arms A can increase over time
    """

    def __init__(
        self,
        num_users=10,
        num_arms=5,
        num_features=10,
        update_freq=20,
        update_arms_strategy="none",
    ):
        """
        Initializes the attributes of the simulation

        Args:
            num_users: The number of users in the simulation
            num_arms: The number of arms/items in the simulation initially
            num_features: number of features for arms and users
            update_freq: number of steps after which we update the number of arms
            update_arms_strategy: strategy to update the arms. One of 'popular', 'corrective', 'counterfactual'
        """
        self.num_users = num_users
        self.num_arms = num_arms
        self.num_features = num_features
        self.update_freq = update_freq
        self.update_arms_strategy = update_arms_strategy
        self.arms = {}  ## arm_id: np.array
        self.users = {}  ## user_id: np.array
        self._init(means=np.arange(-5, 6), scale=1.0)
        self.steps = 0  ## number of steps since last arm update
        self.logs = []  ## each element is of the form [user_id, arm_id, best_arm_id]

    def _init(self, means, scale):
        """
        Initializes the arms and users from a normal distribution where
        each mean is randomly sampled from [-5,5] and variance is always 1.0
        """
        for i in range(self.num_users):
            v = []
            for _ in range(self.num_features):
                v.append(np.random.normal(loc=np.random.choice(means), scale=scale))
            self.users[i] = np.array(v).reshape(-1)
        for i in range(self.num_arms):
            v = []
            for _ in range(self.num_features):
                v.append(np.random.normal(loc=np.random.choice(means), scale=scale))
            self.arms[i] = np.array(v).reshape(-1)

    def reset(self):
        """
        Returns a random user context to begin the simulation
        """
        user_ids = list(self.users.keys())
        user = np.random.choice(user_ids)
        return user, self.users[user]

    def get_reward(self, user_id, arm_id):
        """
        Returns a reward of 0 if the arm chosen is the best arm else -1
        """
        user_context = self.users[user_id]
        best_arm_id, best_score = None, None
        for a_id, arm in self.arms.items():
            score = arm.dot(user_context)
            if not best_arm_id:
                best_arm_id = a_id
                best_score = score
                continue
            if best_score < score:
                best_arm_id = a_id
                best_score = score
        ## Update the logs
        self.logs.append([user_id, arm_id, best_arm_id])
        if arm_id == best_arm_id:
            return 0
        else:
            return -1

    def update_arms(self):
        """
        Three strategies to add a new arm. We will base all these decisions only on the past self.update_frequency (call this K) users' decisions
        1. Counterfactual: Assume there exists some arm that is better than all existing arms for the past K users.
                        We are optimizing to find this arm using SGD on this dataset of K users and their true best arms.
                        The loss is the difference in scores between our current arm and the true best arms of the K users
        2. Corrective: For all the users in the past K users where we got the arm wrong, create a new arm which is the average of their true best arms
        3. Popular: Simply create a new arm which is the mean of the two most popular arms in the last K steps
        4. None: Don't update arms

        Returns True if we added a new arm else False
        """
        if self.update_arms_strategy == "none":
            return False
        if self.update_arms_strategy == "popular":
            """
            Hints:
            We want to create a new arm with features as the mean of the two most popular arms
            Iterate through the logs and find the two most popular arms.
            Note that each element in self.logs is of the form [user_id, chosen arm_id, best arm_id]
            If there is only one arm in the logs, we don't add a new arm, simply return False
            Find the mean of the true theta of the two most popular arms and add a new arm to the Simulator with a new ID
            Note that self.arms is a dictionary of the form {arm_id: theta} where theta is np.array
            The new arm ID should be the next integer in the sequence of arm IDs
            Don't forget to update self.num_arms
            """
            #######################################################
            #########  ~8 lines.   #############
            ### START CODE HERE ###
            ### END CODE HERE ###
            #######################################################
        if self.update_arms_strategy == "corrective":
            """
            Hints:
            We want to create a new arm with features as the weighted mean of the true best arms for users with incorrect predictions
            Iterate through the logs and find the users with incorrect predictions and get their true best arms.
            Note that each element in self.logs is of the form [user_id, chosen arm_id, best arm_id]
            Find the weighted mean of these best_arms and add a new arm to the simulator
            Note that self.arms is a dictionary of the form {arm_id: theta} where theta is np.array
            The new arm ID should be the next integer in the sequence of arm IDs
            Don't forget to update self.num_arms
            """
            #######################################################
            #########  ~7 lines.   #############
            ### START CODE HERE ###
            ### END CODE HERE ###
            #######################################################
        if self.update_arms_strategy == "counterfactual":
            """
            Hints:
            We want to create a new arm that optimizes the objective given in the HW PDF
            Initialize the new theta to an array of zeros and the learning rate eta to 0.1
            Perform one update of batch gradient ascent over the logs
            Use the update equation in the HW PDF to update the new theta
            Note that each element in self.logs is of the form [user_id, chosen arm_id, best arm_id]
            Note that self.arms is a dictionary of the form {arm_id: theta} where theta is np.array
            The new ID should be the next integer in the sequence of arm IDs
            Don't forget to update self.num_arms
            """
            #######################################################
            #########  ~9 lines.   #############
            ### START CODE HERE ###
            ### END CODE HERE ###
            #######################################################

        return True

    def step(self, user_id, arm_id):
        """
        Takes a step in the simulation, calculates rewards, updates logs, increases arms, and returns the new user context
        Args:
            user_id: The id of the user for which the arm was chosen
            arm_id: The id of the arm chosen
        Returns:
            new user_id for the next step of the simulation
            the user context corresponding to this new user_id
            the reward for the current user-arm interaction (0 if the best arm was chosen, -1 otherwise)
            arm_added: boolean which is True if a new arm was added in this step and False otherwise
        """
        ## Update number of steps
        self.steps += 1
        ## Get the reward for the arm played. This also updates the logs
        reward = self.get_reward(user_id, arm_id)
        ## Update the arms (add a new arm)
        arm_added = False
        if self.steps % self.update_freq == 0:
            arm_added = self.update_arms()
            self.logs = []
            self.steps = 0
            # if arm_added:
            # 	print(f'Added a new arm via {self.update_arms_strategy} strategy! New number of arms is {self.num_arms}')
        ## Get the next user context
        user_ids = list(self.users.keys())
        user = np.random.choice(user_ids)
        return user, self.users[user], reward, arm_added
