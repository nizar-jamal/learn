from typing import Callable, Dict

import torch
import torch.autograd as autograd


def log_p_theta(
    x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """
    Computes the log probability of x under a Gaussian distribution with diagonal covariance.

    This function calculates log p(x) where p is a Gaussian distribution with specified
    mean and variance parameters. The computation is done element-wise for efficiency
    and numerical stability by working in log space.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor for which the log probability is evaluated.
    mean : torch.Tensor
        Mean of the Gaussian distribution. Should have the same shape as x.
    log_var : torch.Tensor
        Logarithm of the variance for each component in the Gaussian.
        Should have the same shape as x.

    Returns
    -------
    torch.Tensor
        The element-wise log probability of x under the specified Gaussian distribution.
        We sum over the last dimension to obtain a tensor fo shape (B,)
    """
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###


def compute_l2norm_squared(vector: torch.Tensor) -> torch.Tensor:
    """
    Computes the l2 norm squared of the vector.

    Parameters
    ----------
    vector : torch.Tensor

    Returns
    -------
    torch.Tensor
        L2 norm squared of the vector.
    """
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###


def add_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """
    Adds Gaussian noise to the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to which noise is added.
    noise_std : float
        The standard deviation of the Gaussian noise.

    Returns
    -------
    torch.Tensor
        The input tensor with added Gaussian noise.
    """
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###


def compute_gaussian_score(
    x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """
    Computes the score function, which is the gradient of the log probability of a Gaussian distribution.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor for which the score is computed.
    mean : torch.Tensor
        Mean of the Gaussian distribution.
    log_var : torch.Tensor
        Logarithm of the variance for each component in the Gaussian.

    Returns
    -------
    torch.Tensor
        The score function evaluated at x.
    """
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###


def compute_target_score(
    x: torch.Tensor, noisy: torch.Tensor, std: float
) -> torch.Tensor:
    """
    Computes the target score for denoising score matching.

    This function calculates the ground truth score for noisy data points,
    which represents the gradient of the log probability of the clean data
    given the noisy observations.

    Parameters
    ----------
    x : torch.Tensor
        The original, clean input tensor.
    noisy : torch.Tensor
        The noisy version of the input tensor.
    std : float
        The standard deviation of the noise that was used to create
        the noisy tensor.

    Returns
    -------
    torch.Tensor
        The target score
    """
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###


def compute_score(log_p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes the score function as the gradient of the log probability with respect to x.

    The score function is defined as ∇_x log p(x)

    Parameters
    ----------
    log_p : torch.Tensor
    x : torch.Tensor

    Returns
    -------
    torch.Tensor
        The score function evaluated at x, with the same shape as x.
        Represents ∇_x log p(x) for each sample in the batch.


    """
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###


def compute_divergence(score: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes the divergence of the score function with respect to the input.

    The divergence is calculated as the sum of partial derivatives ∂score_i/∂x_i
    for each dimension i. This implementation computes the divergence by iteratively
    calculating each partial derivative using autograd.

    Parameters
    ----------
    score : torch.Tensor
        The score function evaluated at x, with shape (batch_size, d) where d is
        the dimension of the feature space.
    x : torch.Tensor
        The input tensor with shape (batch_size, d) with respect to which the
        divergence is computed. Must have requires_grad=True.

    Returns
    -------
    torch.Tensor
        The divergence for each sample in the batch, with shape (batch_size,).
        Represents ∑_i ∂score_i/∂x_i for each sample.

    Notes
    -----
    - The computation is done dimension by dimension to avoid memory issues that
      might arise from computing all derivatives at once.
    - retain_graph=True is used to keep the computational graph for subsequent
      iterations over dimensions.
    - create_graph=True enables computation of higher-order derivatives if needed.
    """
    divergence = torch.zeros(x.size(0), device=x.device)
    for i in range(x.size(1)):
        grad_i = torch.autograd.grad(
            score[:, i].sum(),
            x,
            retain_graph=True,  # Keep graph for subsequent iterations
            create_graph=True,  # Needed for higher-order derivatives
        )[0][
            :, i
        ]  # ∂score_i/∂x_i for all samples
        divergence += grad_i

    return divergence
