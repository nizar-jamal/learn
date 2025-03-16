import torch
from torch.nn import functional as F


def loss_nonsaturating_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    d_loss = None
    # You may find some or all of the below useful:
    #   - F.binary_cross_entropy_with_logits
    ### START CODE HERE ###
    x_fake = g(z)
    real_logits = d(x_real)
    fake_logits = d(x_fake.detach())
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    d_loss = real_loss + fake_loss
    return d_loss
    ### END CODE HERE ###
    raise NotImplementedError

def loss_nonsaturating_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    g_loss = None
    
    # You may find some or all of the below useful:
    #   - F.logsigmoid
    ### START CODE HERE ###
    x_fake = g(z)
    fake_logits = d(x_fake)
    g_loss = -F.logsigmoid(fake_logits).mean()
    return g_loss    
    ### END CODE HERE ###
    raise NotImplementedError


def conditional_loss_nonsaturating_d(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    d_loss = None

    ### START CODE HERE ###
    x_fake = g(z, y_real)
    logits_real = d(x_real, y_real)
    logits_fake = d(x_fake.detach(), y_real)
    loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
    d_loss = loss_real + loss_fake
    return d_loss 
    ### END CODE HERE ###
    raise NotImplementedError


def conditional_loss_nonsaturating_g(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    g_loss = None

    ### START CODE HERE ###
    x_fake = g(z, y_real)
    logits_fake = d(x_fake, y_real)
    g_loss = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
    return g_loss
    ### END CODE HERE ###
    raise NotImplementedError


def loss_wasserstein_gp_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): wasserstein discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    d_loss = None

    # You may find some or all of the below useful:
    #   - torch.rand
    #   - torch.autograd.grad(..., create_graph=True)
    ### START CODE HERE ###
    lambda_gp = 10
    x_fake = g(z).detach()
    d_real = d(x_real)
    d_fake = d(x_fake)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    x_interpolated = alpha * x_fake + (1 - alpha) * x_real
    x_interpolated.requires_grad_(True)
    d_interpolated = d(x_interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=x_interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradients_norm - 1) ** 2).mean()
    d_loss = d_fake.mean() - d_real.mean() + gradient_penalty
    return d_loss    
    ### END CODE HERE ###
    raise NotImplementedError


def loss_wasserstein_gp_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): wasserstein generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    g_loss = None
    
    ### START CODE HERE ###
    x_fake = g(z)
    d_fake = d(x_fake)
    g_loss = -d_fake.mean()
    return g_loss
    ### END CODE HERE ###
    raise NotImplementedError