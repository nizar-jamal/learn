#!/usr/bin/env python3
import argparse
import inspect
import math
import sys
import unittest

import numpy as np
import submission
import torch
from diffusers import UNet2DModel
from graderUtil import CourseTestRunner, GradedTestCase, graded
from run_sampling import ddim_inference, ddpm_inference
from run_score_matching import run_experiment
from sampling_config import set_config
from utils import model_factory

#########
# TESTS #
#########

device = torch.device("cpu")
SEED = 42


def checkClose(x, y, rtol=1e-5, atol=0, reduce_func=all):
    if isinstance(x, torch.Tensor):
        return torch.allclose(x, y, rtol, atol)
    elif isinstance(y, torch.Tensor):
        return reduce_func(torch.allclose(xi, y, rtol, atol) for xi in x)
    else:
        return reduce_func(torch.allclose(xi, yi, rtol, atol) for xi, yi in zip(x, y))


class Test_3a(GradedTestCase):
    def setUp(self):
        torch.manual_seed(SEED)
 
    @torch.no_grad()
    @graded(is_extra_credit=True)
    def test_0(self):
        """3a-0-basic: check correct type and shape for log_p_theta"""
        log_p_theta = submission.log_p_theta(
            torch.randn(4, 2), torch.randn(4, 2), torch.randn(4, 2)
        )
        self.assertTrue(
            log_p_theta.shape == torch.Size([4]), "Incorrect shape for log_p_theta"
        )
        self.assertTrue(
            isinstance(log_p_theta, torch.Tensor), "Incorrect type for log_p_theta"
        )

    @graded(is_extra_credit=True)
    def test_1(self):
        """3a-1-basic: check correct type and shape from call to compute_score"""
        x = torch.randn(4, 2, requires_grad=True)
        mean = torch.randn(4, 2, requires_grad=True)
        log_var = torch.randn(4, 2, requires_grad=True)

        log_p_theta = submission.log_p_theta(x, mean, log_var)
        score = submission.compute_score(log_p_theta, x)
        self.assertTrue(
            score.shape == torch.Size([4, 2]),
            "Incorrect shape for score derived from compute_score",
        )
        self.assertTrue(
            isinstance(score, torch.Tensor),
            "Incorrect type for score derived from compute_score",
        )

    @graded(is_extra_credit=True)
    def test_2(self):
        """3a-2-basic: check correct type and shape for trace of Jacobian of the score"""
        x = torch.randn(4, 2, requires_grad=True)
        mean = torch.randn(4, 2, requires_grad=True)
        log_var = torch.randn(4, 2, requires_grad=True)

        log_p_theta = submission.log_p_theta(x, mean, log_var)
        score = submission.compute_score(log_p_theta, x)
        divergence = submission.compute_divergence(score, x)
        self.assertTrue(
            divergence.shape == torch.Size([4]),
            "Incorrect shape for trace of Jacobian of the score",
        )
        self.assertTrue(
            isinstance(divergence, torch.Tensor),
            "Incorrect type for trace of Jacobian of the score",
        )

    @graded(is_extra_credit=True)
    def test_3(self):
        """3a-3-basic: check correct type and shape for compute_l2norm_squared"""
        jacobian = torch.randn(4, 2)
        l2_norm_squared = submission.compute_l2norm_squared(jacobian)
        self.assertTrue(
            l2_norm_squared.shape == torch.Size([4]),
            "Incorrect shape for frobenius_norm_squared",
        )
        self.assertTrue(
            isinstance(l2_norm_squared, torch.Tensor),
            "Incorrect type for frobenius_norm_squared",
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_3c(GradedTestCase):
    def setUp(self):
        torch.manual_seed(SEED)

    @graded(is_extra_credit=True)
    def test_0(self):
        """3c-0-basic: check correct type and shape for add_noise"""
        x = torch.randn(4, 2)
        noise_std = 0.1
        noisy_x = submission.add_noise(x, noise_std)
        self.assertTrue(
            noisy_x.shape == torch.Size([4, 2]), "Incorrect shape for noisy_x"
        )
        self.assertTrue(isinstance(noisy_x, torch.Tensor), "Incorrect type for noisy_x")

    @graded(is_extra_credit=True)
    def test_1(self):
        """3c-1-basic: check correct type and shape for compute_gaussian_score"""
        x = torch.randn(4, 2)
        mean = torch.randn(4, 2)
        log_var = torch.randn(4, 2)
        score = submission.compute_gaussian_score(x, mean, log_var)
        self.assertTrue(score.shape == torch.Size([4, 2]), "Incorrect shape for score")
        self.assertTrue(isinstance(score, torch.Tensor), "Incorrect type for score")

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_4a(GradedTestCase):
    def setUp(self):
        self.config = set_config(
            experiment="ddpm",
            dataset="mnist",
            save_dir="",
            seed=SEED,
            image_path="",
        )

    @graded(timeout=10000)
    def test_0(self):
        """4a-0-basic: check correct type and shape for get_timesteps"""
        timesteps = submission.get_timesteps(4, 2)
        self.assertTrue(
            len(timesteps) == 2,
            "get_timesteps returns tensor of current and previous timesteps",
        )
        self.assertTrue(isinstance(timesteps, tuple), "Incorrect type for timesteps")
        self.assertTrue(
            timesteps[0].shape == torch.Size([2]),
            "Incorrect shape for current timesteps",
        )
        self.assertTrue(
            timesteps[1].shape == torch.Size([2]),
            "Incorrect shape for previous timesteps",
        )

    @graded()
    def test_1(self):
        """4a-1-basic: check correct type and shape for predict_x0"""
        predicted_noise = torch.randn(1, 1, 32, 32)
        t = 999
        sample_t = torch.randn(1, 1, 32, 32)
        scheduler_params = {
            "steps": 1000,
            "betas": torch.rand(1000),
            "alphas": torch.rand(1000),
            "alphas_bar": torch.rand(1000),
        }
        predicted = submission.predict_x0(
            predicted_noise, t, sample_t, scheduler_params
        )
        self.assertTrue(
            predicted.shape == torch.Size([1, 1, 32, 32]),
            "Incorrect shape for predicted image tensor",
        )
        self.assertTrue(
            isinstance(predicted, torch.Tensor), "Incorrect type for image tensor"
        )

    @graded()
    def test_2(self):
        """4a-2-basic: check correct type and shape for compute_forward_posterior_mean"""
        predicted_x0 = torch.randn(1, 1, 32, 32)
        noisy_image = torch.randn(1, 1, 32, 32)
        scheduler_params = {
            "steps": 1000,
            "betas": torch.rand(1000),
            "alphas": torch.rand(1000),
            "alphas_bar": torch.rand(1000),
        }
        t = 999
        t_prev = 998
        mean = submission.compute_forward_posterior_mean(
            predicted_x0, noisy_image, scheduler_params, t, t_prev
        )
        self.assertTrue(
            mean.shape == torch.Size([1, 1, 32, 32]), "Incorrect shape for mean tensor"
        )
        self.assertTrue(
            isinstance(mean, torch.Tensor), "Incorrect type for mean tensor"
        )

    @graded(timeout=10000)
    def test_3(self):
        """4a-3-basic: check correct type and shape for compute_forward_posterior_variance"""
        scheduler_params = {
            "steps": 1000,
            "betas": torch.rand(1000),
            "alphas": torch.rand(1000),
            "alphas_bar": torch.rand(1000),
        }
        t = 999
        t_prev = 998
        variance = submission.compute_forward_posterior_variance(
            scheduler_params, t, t_prev
        )
        self.assertTrue(
            variance.shape == torch.Size([]), "Incorrect shape for variance tensor"
        )
        self.assertTrue(
            isinstance(variance, torch.Tensor), "Incorrect type for variance tensor"
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_4b(GradedTestCase):
    def setUp(self):
        self.config = set_config(
            experiment="ddim",
            dataset="mnist",
            save_dir="",
            seed=SEED,
            image_path="",
        )
        self.unet = model_factory(
            model_type=self.config.model_type,
            model_id=self.config.model_id,
            weights_path=self.config.weights_path,
            device=self.config.device,
        )

    @graded(timeout=10000)
    def test_0(self):
        """4b-0-basic: check correct type and shape for get_stochasticity_std"""
        eta = 0.1
        t = 0
        t_prev = -1
        alphas_bar = torch.rand(1000)
        std = submission.get_stochasticity_std(eta, t, t_prev, alphas_bar)
        self.assertTrue(std.shape == torch.Size([]), "Incorrect shape for std")
        self.assertTrue(isinstance(std, torch.Tensor), "Incorrect type for std")

    @graded()
    def test_1(self):
        """4b-1-basic: check correct type and shape for predict_sample_direction"""
        alphas_bars_prev = 0.1
        predicted_noise = torch.randn(1, 1, 32, 32)
        std = torch.randn(1)

        pred_direction = submission.predict_sample_direction(
            alphas_bars_prev, predicted_noise, std
        )
        self.assertTrue(
            pred_direction.shape == torch.Size([1, 1, 32, 32]),
            "Incorrect shape for predicted direction tensor",
        )
        self.assertTrue(
            isinstance(pred_direction, torch.Tensor),
            "Incorrect type for predicted direction tensor",
        )

    @graded()
    def test_2(self):
        """4b-2-basic: check correct type and shape for stochasticity_term"""
        std = torch.randn(1)
        noise = torch.randn(1, 1, 32, 32)
        term = submission.stochasticity_term(std, noise)
        self.assertTrue(
            term.shape == torch.Size([1, 1, 32, 32]),
            "Incorrect shape for stochasticity term tensor",
        )
        self.assertTrue(
            isinstance(term, torch.Tensor),
            "Incorrect type for stochasticity term tensor",
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_5b(GradedTestCase):
    def setUp(self):
        self.config = set_config(
            experiment="ddpm",
            dataset="mnist",
            save_dir="",
            seed=SEED,
            image_path="",
        )
        self.unet = model_factory(
            model_type=self.config.model_type,
            model_id=self.config.model_id,
            weights_path=self.config.weights_path,
            device=self.config.device,
        )

    @graded()
    def test_0(self):
        """5b-0-basic: check correct type and shape for get_mask"""
        image = torch.randn(1, 3, 32, 32)
        mask = submission.get_mask(image)
        self.assertTrue(
            mask.shape == torch.Size([1, 3, 32, 32]), "Incorrect shape for mask"
        )
        self.assertTrue(isinstance(mask, torch.Tensor), "Incorrect type for mask")

    @graded()
    def test_1(self):
        """5b-1-basic: check correct type and shape for add_forward_tnoise"""
        image = torch.randn(1, 3, 32, 32)
        timestep = 999
        scheduler_data = {"alphas_bar": torch.rand(1000)}
        noisy_image = submission.add_forward_tnoise(image, timestep, scheduler_data)
        self.assertTrue(
            noisy_image.shape == torch.Size([1, 3, 32, 32]),
            "Incorrect shape for noisy image",
        )
        self.assertTrue(
            isinstance(noisy_image, torch.Tensor), "Incorrect type for noisy image"
        )

    @graded()
    def test_2(self):
        """5b-2-basic: check correct type and shape for apply_inpainting_mask"""
        original_image = torch.randn(1, 3, 32, 32)
        noisy_image = torch.randn(1, 3, 32, 32)
        mask = torch.randint(0, 2, (1, 3, 32, 32))
        timestep = 999
        scheduler_data = {"alphas_bar": torch.rand(1000)}
        inpainted_image = submission.apply_inpainting_mask(
            original_image, noisy_image, mask, timestep, scheduler_data
        )
        self.assertTrue(
            inpainted_image.shape == torch.Size([1, 3, 32, 32]),
            "Incorrect shape for inpainted image",
        )
        self.assertTrue(
            isinstance(inpainted_image, torch.Tensor),
            "Incorrect type for inpainted image",
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
