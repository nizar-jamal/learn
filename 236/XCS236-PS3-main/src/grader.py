#!/usr/bin/env python3
import sys
import unittest
import argparse
from graderUtil import graded, CourseTestRunner, GradedTestCase
import torch
import inspect
import numpy as np
from network import *

# import student submission
import submission

#########
# TESTS #
#########

device = torch.device("cpu")
SEED = 236

def checkClose(x, y, rtol=1e-5, atol=0, reduce_func=all):
    if isinstance(x, torch.Tensor):
        return torch.allclose(x, y, rtol, atol)
    elif isinstance(y, torch.Tensor):
        return reduce_func(torch.allclose(xi, y, rtol, atol) for xi in x)
    else:
        return reduce_func(
            torch.allclose(xi, yi, rtol, atol) for xi, yi in zip(x, y)
        )

class Test_1a(GradedTestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_size = 2
        self.hidden_size = 100
        self.n_hidden = 1
        self.atol = 0.05
        self.sol_MADE = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.flow_network.MADE)
        self.expected_x_mapping = torch.tensor([[-0.9382,  0.1923],
                                                [-0.2535,  0.9370],
                                                [ 0.3254,  0.3384],
                                                [ 0.7355, -1.1332]])

    @torch.no_grad()
    @graded()
    def test_0(self):
        """1a-0-basic: check output shapes for MADE forward"""
        torch.manual_seed(SEED)
        z = torch.randn(self.batch_size, self.input_size).to(device)
        made = submission.flow_network.MADE(self.input_size, self.hidden_size, self.n_hidden).to(device)
        x, log_det = made(z)
        self.assertTrue(log_det.shape == torch.Size([self.batch_size]), "log_det output from MADE forward incorrect shape")
        self.assertTrue(x.shape == torch.Size([self.batch_size, self.input_size]), "x output from MADE forward incorrect shape")

    @torch.no_grad()
    @graded()
    def test_1(self):
        """1a-1-basic: check x mapping for MADE forward"""
        torch.manual_seed(SEED)
        z = torch.randn(self.batch_size, self.input_size).to(device)
        made = submission.flow_network.MADE(self.input_size, self.hidden_size, self.n_hidden).to(device)
        x, _ = made(z)
        self.assertTrue(np.allclose(x,  self.expected_x_mapping, atol=self.atol), "Incorrect x mapping created from MADE forward")
    
    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_1b(GradedTestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_size = 2
        self.hidden_size = 100
        self.n_hidden = 1
        self.atol = 0.05
        self.sol_MADE = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.flow_network.MADE)
        self.expected_z_mapping = torch.tensor([[-0.8924,  0.1690],
                                                [-0.0995,  0.7830],
                                                [ 0.5708,  0.4065],
                                                [ 1.0457, -1.0252]])
    
    @torch.no_grad()
    @graded()
    def test_0(self):
        """1b-0-basic: check output shapes for MADE inverse"""
        torch.manual_seed(SEED)
        x = torch.randn(self.batch_size, self.input_size).to(device)

        made = submission.flow_network.MADE(self.input_size, self.hidden_size, self.n_hidden).to(device)
        z, log_det = made.inverse(x)
        self.assertTrue(log_det.shape == torch.Size([self.batch_size]), "log_det output from MADE inverse incorrect shape")
        self.assertTrue(z.shape == torch.Size([self.batch_size, self.input_size]), "z output from MADE inverse incorrect shape")

    @torch.no_grad()
    @graded()
    def test_1(self):
        """1b-1-basic: check z mapping for MADE inverse"""
        torch.manual_seed(SEED)
        x = torch.randn(self.batch_size, self.input_size).to(device)

        made = submission.flow_network.MADE(self.input_size, self.hidden_size, self.n_hidden).to(device)
        z, _ = made.inverse(x)
        self.assertTrue(np.allclose(z,  self.expected_z_mapping, atol=self.atol), "Incorrect z mapping created from MADE inverseX")
    
    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_1c(GradedTestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_size = 2
        self.hidden_size = 100
        self.n_hidden = 1
        self.n_flows = 5
        self.atol = 0.05
        self.sol_MAF = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.flow_network.MAF)
        self.expected_log_probs = torch.tensor(-2.2746)
    
    @torch.no_grad()
    @graded()
    def test_0(self):
        """1c-0-basic: check output shapes for log_probs in MAF"""
        torch.manual_seed(SEED)
        x = torch.randn(self.batch_size, self.input_size).to(device)
        torch.manual_seed(SEED)
        maf = submission.flow_network.MAF(self.input_size, self.hidden_size, self.n_hidden, self.n_flows).to(device)
        log_probs = maf.log_probs(x)
        self.assertTrue(log_probs.shape == torch.Size([]), "incorrect shape for log_probs in MAF")
    
    @torch.no_grad()
    @graded()
    def test_1(self):
        """1c-1-basic: check log_probs values in MAF"""
        torch.manual_seed(SEED)
        x = torch.randn(self.batch_size, self.input_size).to(device)
        torch.manual_seed(SEED)
        maf = submission.flow_network.MAF(self.input_size, self.hidden_size, self.n_hidden, self.n_flows).to(device)
        log_probs = maf.log_probs(x)
        self.assertTrue(np.allclose(log_probs,  self.expected_log_probs, atol=self.atol), "Incorrect log_probs value for MAF")

    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_2b(GradedTestCase):
    def setUp(self):
        self.batch_size = 4
        self.atol = 0.05
        self.sol_loss_nonsaturating_d = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.gan.loss_nonsaturating_d)
        self.sol_loss_nonsaturating_g = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.gan.loss_nonsaturating_g)
        self.expected_d_loss = torch.tensor(1.3850)
        self.expected_g_loss = torch.tensor(0.6956)
    
    @torch.no_grad()
    @graded()
    def test_0(self):
        """2b-0-basic: check shapes for loss_nonsaturating_d for GAN"""
        torch.manual_seed(SEED)
        x_real = torch.randn(self.batch_size, 1, 28, 28).to(device)

        g = Generator().to(device)
        d = Discriminator().to(device)

        torch.manual_seed(SEED)
        d_loss = submission.gan.loss_nonsaturating_d(g, d, x_real, device=device)
        self.assertTrue(d_loss.shape == torch.Size([]), "Incorrect shape for d_loss in loss_nonsaturating_d for GAN")

    @torch.no_grad()
    @graded()
    def test_1(self):
        """2b-1-basic: check outputs for loss_nonsaturating_d for GAN"""
        torch.manual_seed(SEED)
        x_real = torch.randn(self.batch_size, 1, 28, 28).to(device)

        g = Generator().to(device)
        d = Discriminator().to(device)

        torch.manual_seed(SEED)
        d_loss = submission.gan.loss_nonsaturating_d(g, d, x_real, device=device)
        self.assertTrue(np.allclose(d_loss, self.expected_d_loss, atol=self.atol), "Incorrect values for d_loss in loss_nonsaturating_d for GAN")
    
    @torch.no_grad()
    @graded()
    def test_2(self):
        """2b-2-basic: check shapes for loss_nonsaturating_g for GAN"""
        torch.manual_seed(SEED)
        x_real = torch.randn(self.batch_size, 1, 28, 28).to(device)

        g = Generator().to(device)
        d = Discriminator().to(device)

        torch.manual_seed(SEED)
        g_loss = submission.gan.loss_nonsaturating_g(g, d, x_real, device=device)
        self.assertTrue(g_loss.shape == torch.Size([]), "Incorrect shape for g_loss in loss_nonsaturating_g for GAN")
    
    @torch.no_grad()
    @graded()
    def test_3(self):
        """2b-3-basic: check values for loss_nonsaturating_g for GAN"""
        torch.manual_seed(SEED)
        x_real = torch.randn(self.batch_size, 1, 28, 28).to(device)

        g = Generator().to(device)
        d = Discriminator().to(device)

        torch.manual_seed(SEED)
        g_loss = submission.gan.loss_nonsaturating_g(g, d, x_real, device=device)
        self.assertTrue(np.allclose(g_loss, self.expected_g_loss, atol=self.atol), "Incorrect values for g_loss in loss_nonsaturating_g for GAN")

    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_4b(GradedTestCase):
    def setUp(self):
        self.batch_size = 4
        self.atol = 0.05
        self.sol_conditional_loss_nonsaturating_d = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.gan.conditional_loss_nonsaturating_d)
        self.sol_conditional_loss_nonsaturating_g = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.gan.conditional_loss_nonsaturating_g)
        self.expected_cond_d_loss = torch.tensor(1.3871)
        self.expected_cond_g_loss = torch.tensor(0.6864)
    
    @torch.no_grad()
    @graded()
    def test_0(self):
        """4b-0-basic: check shapes for conditional_loss_nonsaturating_d for GAN"""
        torch.manual_seed(SEED)
        x_real = torch.randn(self.batch_size, 1, 28, 28).to(device)
        y_real = torch.randint(10, (self.batch_size,), dtype=torch.long).to(device)
        g = ConditionalGenerator().to(device)
        d = ConditionalDiscriminator().to(device)

        torch.manual_seed(SEED)
        cond_d_loss = submission.gan.conditional_loss_nonsaturating_d(
            g, d, x_real, y_real, device=device
        )
        self.assertTrue(cond_d_loss.shape == torch.Size([]), "Incorrect shape for d_loss in conditional_loss_nonsaturating_d for GAN")
    
    @torch.no_grad()
    @graded()
    def test_1(self):
        """4b-1-basic: check outputs for conditional_loss_nonsaturating_d for GAN"""
        torch.manual_seed(SEED)
        x_real = torch.randn(self.batch_size, 1, 28, 28).to(device)
        y_real = torch.randint(10, (self.batch_size,), dtype=torch.long).to(device)
        g = ConditionalGenerator().to(device)
        d = ConditionalDiscriminator().to(device)

        torch.manual_seed(SEED)
        cond_d_loss = submission.gan.conditional_loss_nonsaturating_d(
            g, d, x_real, y_real, device=device
        )
        self.assertTrue(np.allclose(cond_d_loss, self.expected_cond_d_loss, atol=self.atol), "Incorrect values for d_loss in conditional_loss_nonsaturating_d for GAN")
    
    @torch.no_grad()
    @graded()
    def test_2(self):
        """4b-2-basic: check shapes for conditional_loss_nonsaturating_g for GAN"""
        torch.manual_seed(SEED)
        x_real = torch.randn(self.batch_size, 1, 28, 28).to(device)
        y_real = torch.randint(10, (self.batch_size,), dtype=torch.long).to(device)
        g = ConditionalGenerator().to(device)
        d = ConditionalDiscriminator().to(device)

        torch.manual_seed(SEED)
        cond_g_loss = submission.gan.conditional_loss_nonsaturating_g(
            g, d, x_real, y_real, device=device
        )
        self.assertTrue(cond_g_loss.shape == torch.Size([]), "Incorrect shape for g_loss in conditional_loss_nonsaturating_g for GAN")
    
    @torch.no_grad()
    @graded()
    def test_3(self):
        """4b-3-basic: check outputs for conditional_loss_nonsaturating_g for GAN"""
        torch.manual_seed(SEED)
        x_real = torch.randn(self.batch_size, 1, 28, 28).to(device)
        y_real = torch.randint(10, (self.batch_size,), dtype=torch.long).to(device)
        g = ConditionalGenerator().to(device)
        d = ConditionalDiscriminator().to(device)

        torch.manual_seed(SEED)
        cond_g_loss = submission.gan.conditional_loss_nonsaturating_g(
            g, d, x_real, y_real, device=device
        )
        self.assertTrue(np.allclose(cond_g_loss, self.expected_cond_g_loss, atol=self.atol), "Incorrect values for g_loss in conditional_loss_nonsaturating_g for GAN")

    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_5e(GradedTestCase):
    def setUp(self):
        self.batch_size = 4
        self.atol = 0.05
        self.sol_loss_wasserstein_gp_g = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.gan.loss_wasserstein_gp_g)
        self.sol_loss_wasserstein_gp_d = self.run_with_solution_if_possible(submission, lambda sub_or_sol:sub_or_sol.gan.loss_wasserstein_gp_d)
        self.expected_wass_d_loss = torch.tensor(9.2980)
        self.expected_wass_g_loss = torch.tensor(0.0050)

    @graded()
    def test_0(self):
        """5e-0-basic: check shapes for loss_wasserstein_gp_d for GAN"""
        torch.manual_seed(SEED)
        x_real = torch.randn(self.batch_size, 1, 28, 28).to(device)
        g = Generator().to(device)
        d = Discriminator().to(device)
        torch.manual_seed(SEED)
        wass_d_loss = submission.gan.loss_wasserstein_gp_d(g, d, x_real, device=device)
        self.assertTrue(wass_d_loss.shape == torch.Size([]), "Incorrect shape for d_loss in loss_wasserstein_gp_d for GAN")
    
    @graded()
    def test_1(self):
        """5e-1-basic: check outputs for loss_wasserstein_gp_d for GAN"""
        torch.manual_seed(SEED)
        x_real = torch.randn(self.batch_size, 1, 28, 28).to(device)
        g = Generator().to(device)
        d = Discriminator().to(device)
        torch.manual_seed(SEED)
        wass_d_loss = submission.gan.loss_wasserstein_gp_d(g, d, x_real, device=device)
        self.assertTrue(np.allclose(wass_d_loss.detach().numpy(), self.expected_wass_d_loss, atol=self.atol), "Incorrect values for d_loss in loss_wasserstein_gp_d for GAN")
    
    @torch.no_grad()
    @graded()
    def test_2(self):
        """5e-2-basic: check shapes for loss_wasserstein_gp_g for GAN"""
        torch.manual_seed(SEED)
        x_real = torch.randn(self.batch_size, 1, 28, 28).to(device)
        g = Generator().to(device)
        d = Discriminator().to(device)
        torch.manual_seed(SEED)
        wass_g_loss = submission.gan.loss_wasserstein_gp_g(g, d, x_real, device=device)
        self.assertTrue(wass_g_loss.shape == torch.Size([]), "Incorrect shape for g_loss in loss_wasserstein_gp_g for GAN")
    
    @torch.no_grad()
    @graded()
    def test_3(self):
        """5e-3-basic: check outputs for loss_wasserstein_gp_g for GAN"""
        torch.manual_seed(SEED)
        x_real = torch.randn(self.batch_size, 1, 28, 28).to(device)
        g = Generator().to(device)
        d = Discriminator().to(device)
        torch.manual_seed(SEED)
        wass_g_loss = submission.gan.loss_wasserstein_gp_g(g, d, x_real, device=device)
        self.assertTrue(np.allclose(wass_g_loss, self.expected_wass_g_loss, atol=self.atol), "Incorrect values for g_loss in loss_wasserstein_gp_g for GAN")

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