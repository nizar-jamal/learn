import torch
from torch import nn
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim) # from ./nns/v1.py

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

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
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl, rec
        ################################################################################
        ### START CODE HERE ###
        z_m, z_v = self.enc(x)
        z = ut.sample_gaussian(z_m, z_v)
        x_logits = self.dec(z)
        kl = ut.kl_normal(z_m, z_v, self.z_prior_m, self.z_prior_v)
        rec = ut.log_bernoulli_with_logits(x, x_logits)
        nelbo = kl - rec
        # nelbo = kl + rec # since log)_bernoulli_with_logits is negative
        return(nelbo.mean(), kl.mean(), -rec.mean())
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
        #
        # HINT: The summation over m may seem to prevent us from 
        # splitting the ELBO into the KL and reconstruction terms, but instead consider 
        # calculating log_normal w.r.t prior and q
        ################################################################################
        ### START CODE HERE ###

        # Encode the input to get mean and variance of q(z|x)
        z_m, z_v = self.enc(x)
        
        # Expand dimensions for importance sampling (batch, iw, dim)
        z_m = z_m.unsqueeze(1).expand(-1, iw, -1)
        z_v = z_v.unsqueeze(1).expand(-1, iw, -1)
        
        # Sample latent variables z from q(z|x)
        z = ut.sample_gaussian(z_m, z_v)  # shape: (batch, iw, z_dim)
        
        # Decode to get logits for p(x|z)
        x_logits = self.dec(z)  # shape: (batch, iw, dim)
        
        # Compute log probabilities
        log_p_x_given_z = ut.log_bernoulli_with_logits(
            x.unsqueeze(1).expand(-1, iw, -1), x_logits
        )  # shape: (batch, iw)
        
        log_p_z = ut.log_normal(
            z, self.z_prior_m.expand_as(z), self.z_prior_v.expand_as(z)
        )  # shape: (batch, iw)
        
        log_q_z_given_x = ut.log_normal(z, z_m, z_v)  # shape: (batch, iw)

        # Compute log importance weights
        log_weights = log_p_x_given_z + log_p_z - log_q_z_given_x  # shape: (batch, iw)

        # Compute IWAE bound using log-sum-exp trick for numerical stability
        max_log_weights = torch.max(log_weights, dim=1, keepdim=True)[0]  # shape: (batch, 1)
        iwae_log_sum_exp = max_log_weights + torch.log(
            torch.sum(torch.exp(log_weights - max_log_weights), dim=1)
        )  # shape: (batch,)
        
        niwae = -torch.mean(iwae_log_sum_exp)  # Negative IWAE bound

        # Compute ELBO KL and reconstruction terms for additional outputs
        kl = ut.kl_normal(
            z_m.squeeze(1), z_v.squeeze(1), self.z_prior_m.squeeze(), self.z_prior_v.squeeze()
        ).mean()  # Average over batch
        
        rec = ut.log_bernoulli_with_logits(x, self.dec(ut.sample_gaussian(z_m[:, 0], z_v[:, 0]))).mean()

        return niwae, kl, -rec

        ### END CODE HERE ###
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
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
