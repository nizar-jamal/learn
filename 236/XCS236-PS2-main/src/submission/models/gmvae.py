import numpy as np
import torch
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

from torch import nn

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim) # from ./nns/v1.py

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl_z, rec
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        ### START CODE HERE ###
        m_mixture, v_mixture = prior  # Shapes: (1, k, z_dim)

        # Encode x to get q(z|x) parameters
        m_q, v_q  = self.enc(x)  # Encoder outputs mean and variance of q(z|x)

        # Sample z from q(z|x) using the reparameterization trick
        z_q = ut.sample_gaussian(m_q, v_q)  # Shape: (batch, z_dim)

        # Compute log probabilities
        log_p_z = ut.log_normal_mixture(z_q, m_mixture.squeeze(0), v_mixture.squeeze(0))  # Prior log-prob
        log_q_z_x = ut.log_normal(z_q, m_q, v_q)  # Posterior log-prob

        # Decode z to reconstruct x
        logits = self.dec(z_q)  # Decoder outputs logits for Bernoulli distribution
        log_p_x_given_z = ut.log_bernoulli_with_logits(x, logits) 
        # -nn.BCEWithLogitsLoss(reduction='none')(logits, x).sum(dim=-1)  # Reconstruction term

        # Compute KL divergence and reconstruction loss
        kl = torch.mean(log_q_z_x - log_p_z)  # KL divergence term
        rec = -torch.mean(log_p_x_given_z)  # Reconstruction term

        # Compute NELBO as KL + Reconstruction loss
        nelbo = kl + rec

        return nelbo, kl, rec
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   niwae, kl, rec
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        ### START CODE HERE ###
        m_mixture, v_mixture = prior  # Shapes: (1, k, z_dim)

        # Encode x to get q(z|x) parameters
        m_q, v_q = self.enc(x)  # Encoder outputs mean and variance of q(z|x)

        # Sample z from q(z|x) using reparameterization trick
        batch_size = x.size(0)
        z_q = ut.sample_gaussian(
            m_q.unsqueeze(1).expand(-1, iw, -1), 
            v_q.unsqueeze(1).expand(-1, iw, -1)
        )  # Shape: (batch_size, iw, z_dim)

        # Compute log probabilities for IWAE
        log_p_z = ut.log_normal_mixture(
            z_q.unsqueeze(2),  # Shape: (batch_size, iw, 1, z_dim)
            m_mixture.squeeze(0).unsqueeze(0),  # Shape: (1, 1, k, z_dim)
            v_mixture.squeeze(0).unsqueeze(0)   # Shape: (1, 1, k, z_dim)
        )  # Shape: (batch_size, iw)

        log_q_z_x = ut.log_normal(
            z_q,
            m_q.unsqueeze(1), 
            v_q.unsqueeze(1)
        )  # Shape: (batch_size, iw)

        logits = self.dec(z_q.view(-1, self.z_dim))  # Decode z
        log_p_x_given_z = ut.log_bernoulli_with_logits(
            x.unsqueeze(1).expand(-1, iw, -1).reshape(-1, x.size(1)), 
            logits
        ).view(batch_size, iw)  # Reconstruction term

        # Compute importance weights
        log_weights = log_p_x_given_z + log_p_z - log_q_z_x  # Shape: (batch_size, iw)
        
        # Normalize weights using log-sum-exp trick
        niwae = -torch.mean(torch.logsumexp(log_weights - torch.log(torch.tensor(iw)), dim=1))

        # ELBO decomposition (for comparison purposes)
        kl = torch.mean(log_q_z_x - log_p_z)  # KL divergence term (ELBO)
        rec = -torch.mean(log_p_x_given_z)  # Reconstruction term (ELBO)

        return niwae, kl, rec
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
