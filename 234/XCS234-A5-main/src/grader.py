#!/usr/bin/env python3
import unittest
import random
import sys
import copy
import argparse
import inspect
import collections
import os
import pickle
import gzip
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
from itertools import product
from utils.data_preprocessing import dose_class, load_data, LABEL_KEY

# Import student submission
import submission

#########
# TESTS #
#########


class Test_1a(GradedTestCase):
    @graded(timeout=2000000, is_hidden=False)
    def test_0(self):
        """1a-0-basic: test for fixed prediction"""
        data = load_data()
        learner = submission.FixedDosePolicy()
        prediction = learner.choose(dict(data.iloc[0]))
        predictions = []
        for t in range(10):
            x = dict(data.iloc[t])
            action = learner.choose(x)
            predictions.append(action)

        self.assertEqual([prediction], np.unique(predictions))

    @graded(timeout=2000000, is_hidden=False)
    def test_1(self):
        """1a-1-basic: evaluate the performance of clincal model on a single example"""
        data = load_data()
        learner = submission.ClinicalDosingPolicy()
        x = dict(data.iloc[0])
        label = x.pop(LABEL_KEY)
        action = learner.choose(x)

        self.assertEqual(action, dose_class(label))

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_1b(GradedTestCase):
    @graded(timeout=2, is_hidden=False)
    def test_0(self):
        """1b-0-basic: basic test for correct initialization"""
        data = load_data()
        x = dict(data.iloc[0])
        learner = submission.LinUCB(3, x.keys(), alpha=1)

        self.assertEqual(learner.features, x.keys())
        self.assertEqual(learner.d, len(x.keys()))
        self.assertTrue(np.array_equal(learner.A[0], np.eye(learner.d)))
        self.assertTrue(np.array_equal(learner.b[0], np.zeros(learner.d)) or np.array_equal(learner.b[0], np.zeros((learner.d, 1))))

    @graded(timeout=2000000, is_hidden=False)
    def test_1(self):
        """1b-1-basic: evaluate the choose function of LinUCB on a single example"""
        features = [
            "Age in decades",
            "Height (cm)",
            "Weight (kg)",
            "Male",
            "Female",
            "Asian",
            "Black",
            "White",
            "Unknown race",
            "Carbamazepine (Tegretol)",
            "Phenytoin (Dilantin)",
            "Rifampin or Rifampicin",
            "Amiodarone (Cordarone)",
            "VKORC1AG",
            "VKORC1AA",
            "VKORC1UN",
            "CYP2C912",
            "CYP2C913",
            "CYP2C922",
            "CYP2C923",
            "CYP2C933",
            "CYP2C9UN",
        ]
        data = load_data()
        x = dict(data.iloc[0])
        learner = submission.LinUCB(3, features, alpha=1)
        prediction_class = 0 # "low"
        action = learner.choose(x)

        self.assertEqual(action, prediction_class)

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_1c(GradedTestCase):
    @graded(timeout=10, is_hidden=False)
    def test_0(self):
        """1c-0-basic: evaluate the choose function of egreedy disjoint linear UCB on single example"""
        features = [
            "Age in decades",
            "Height (cm)",
            "Weight (kg)",
            "Male",
            "Female",
            "Asian",
            "Black",
            "White",
            "Unknown race",
            "Carbamazepine (Tegretol)",
            "Phenytoin (Dilantin)",
            "Rifampin or Rifampicin",
            "Amiodarone (Cordarone)",
            "VKORC1AG",
            "VKORC1AA",
            "VKORC1UN",
            "CYP2C912",
            "CYP2C913",
            "CYP2C922",
            "CYP2C923",
            "CYP2C933",
            "CYP2C9UN",
        ]
        data = load_data()
        x = dict(data.iloc[0])
        learner = submission.eGreedyLinB(3, features, alpha=1)

        np.random.seed(0)
        random_predictions = []
        for _ in range(1000):
            learner.time = 0
            random_predictions.append(learner.choose(x))

        np.random.seed(0)
        egreedy_predictions = []
        for _ in range(1000):
            learner.time = 1
            egreedy_predictions.append(learner.choose(x))

        np.random.seed(0)
        greedy_predictions = []
        for _ in range(1000):
            learner.time = 1e10
            greedy_predictions.append(learner.choose(x))

        random_low = collections.Counter(random_predictions)[0] / 1000
        egreedy_low = collections.Counter(egreedy_predictions)[0] / 1000
        greedy_low = collections.Counter(greedy_predictions)[0] / 1000

        self.assertAlmostEqual(random_low, 0.333, delta=0.05)
        self.assertAlmostEqual(egreedy_low, 0.666, delta=0.05)
        self.assertAlmostEqual(greedy_low, 1.0, delta=0.0001)

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_1d(GradedTestCase):
    @graded(timeout=2, is_hidden=False)
    def test_0(self):
        """1d-0-basic: basic test for correct initialization"""
        data = load_data()
        x = dict(data.iloc[0])
        learner = submission.ThomSampB(3, x.keys(), alpha=0.001)

        self.assertEqual(learner.features, x.keys())
        self.assertTrue(np.array_equal(learner.B[0], np.eye(learner.d)))
        self.assertTrue(np.array_equal(learner.mu[0], np.zeros((learner.d))))
        self.assertTrue(np.array_equal(learner.f[0], np.zeros((learner.d))))

    @graded(timeout=5000000, is_hidden=False)
    def test_1(self):
        """1d-1-basic: basic evaluation of the choose function of Thompson Sampling on a single example"""
        features = [
            "Age in decades",
            "Height (cm)",
            "Weight (kg)",
            "Male",
            "Female",
            "Asian",
            "Black",
            "White",
            "Unknown race",
            "Carbamazepine (Tegretol)",
            "Phenytoin (Dilantin)",
            "Rifampin or Rifampicin",
            "Amiodarone (Cordarone)",
            "VKORC1AG",
            "VKORC1AA",
            "VKORC1UN",
            "CYP2C912",
            "CYP2C913",
            "CYP2C922",
            "CYP2C923",
            "CYP2C933",
            "CYP2C9UN",
        ]
        data = load_data()
        learner = submission.ThomSampB(3, features, alpha=0.001)
        prediction = "medium"
        x = dict(data.iloc[0])
        predictions = []
        np.random.seed(0)
        for _ in range(1000):
            predictions.append(learner.choose(x))

        low = collections.Counter(predictions)[0] / 1000
        self.assertAlmostEqual(low, 0.333, delta=0.05)

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_2a(GradedTestCase):
    @graded(timeout=1)
    def test_0(self):
        """2a-0-basic: basic test for DynamicLinUCB.add_arm_params"""
        initial_num_arms = 3
        num_features = 10
        learner = submission.DynamicLinUCB(initial_num_arms, ["None" for _ in range(num_features)])

        learner.add_arm_params()

        self.assertEqual(initial_num_arms + 1, len(learner.A))
        self.assertTrue(np.array_equal(learner.A[0], learner.A[-1]))
        self.assertTrue(np.array_equal(learner.b[0], learner.b[-1]))


class Test_2b(GradedTestCase):

    def setUp(self):

        self.k = 1000
        self.num_users = 25
        self.num_arms = 10
        self.num_features = 10
        self.T = 10000

    @graded(timeout=1)
    def test_0(self):
        """2b-0-basic: test whether arm number increased for Simulator.update_arms with `popular` strategy"""

        num_arms = self.num_arms

        sim = submission.Simulator(
            num_users=self.num_users,
            num_arms=self.num_arms,
            num_features=self.num_features,
            update_freq=self.k,
            update_arms_strategy="popular",
        )

        learner = submission.RandomPolicy(probs=[1/num_arms] * num_arms)

        u, x = sim.reset()
        for _ in range(self.T):
            action = learner.choose(x)
            _, _, _, arm_added = sim.step(u, action)
            if arm_added:
                num_arms += 1
                learner = submission.RandomPolicy(probs=[1 / num_arms] * num_arms)
                self.assertEqual(sim.num_arms, num_arms, "Invalid current number of arms!")

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded(timeout=1000000)
    def test_3(self):
        """2b-3-basic: test whether arm number increased for Simulator.update_arms with `corrective` strategy"""

        num_arms = self.num_arms

        sim = submission.Simulator(
            num_users=self.num_users,
            num_arms=self.num_arms,
            num_features=self.num_features,
            update_freq=self.k,
            update_arms_strategy="corrective",
        )

        learner = submission.RandomPolicy(probs=[1 / num_arms] * num_arms)

        u, x = sim.reset()
        for _ in range(self.T):
            action = learner.choose(x)
            _, _, _, arm_added = sim.step(u, action)
            if arm_added:
                num_arms += 1
                learner = submission.RandomPolicy(probs=[1 / num_arms] * num_arms)
                self.assertEqual(
                    sim.num_arms, num_arms, "Invalid current number of arms!"
                )

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded(timeout=1)
    def test_5(self):
        """2b-5-basic: test whether arm number increased for Simulator.update_arms with `counterfactual` strategy"""

        num_arms = self.num_arms

        sim = submission.Simulator(
            num_users=self.num_users,
            num_arms=self.num_arms,
            num_features=self.num_features,
            update_freq=self.k,
            update_arms_strategy="counterfactual",
        )

        learner = submission.RandomPolicy(probs=[1 / num_arms] * num_arms)

        u, x = sim.reset()
        for _ in range(self.T):
            action = learner.choose(x)
            _, _, _, arm_added = sim.step(u, action)
            if arm_added:
                num_arms += 1
                learner = submission.RandomPolicy(probs=[1 / num_arms] * num_arms)
                self.assertEqual(
                    sim.num_arms, num_arms, "Invalid current number of arms!"
                )

    ### BEGIN_HIDE ###
    ### END_HIDE ###


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)


if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)
