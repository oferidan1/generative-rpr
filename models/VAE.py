"""NICE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim, device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(VAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )


    def sample(self,sample_size,mu=None,logvar=None):
        '''
        :param sample_size: Number of samples
        :param mu: z mean, None for prior (init with zeros)
        :param logvar: z logstd, None for prior (init with zeros)
        :return:
        '''
        if mu==None:
            mu = torch.zeros((sample_size,self.latent_dim)).to(self.device)
        if logvar == None:
            logvar = torch.zeros((sample_size,self.latent_dim)).to(self.device)
        #TODO
        with torch.no_grad():
        	z = torch.randn((sample_size, self.latent_dim)).to(self.device)
	        recon = self.decoder(self.upsample(z).view(-1, 64, 7, 7))
        return recon

    def z_sample(self, mu, logvar):
        #TODO
        # let's do the reparametirization trick, i.e. we will sample from ~N(mu, var) using ~N(0,1)
        sigma = torch.exp(0.5 * logvar)
        # create vector from random normal distribution
        epsilon = torch.randn_like(sigma).to(self.device)
        # now the reparam trick:
        return mu + epsilon * sigma

    def loss1(self,x,recon, mu,logvar):
        #TODO
        # our loss criteria is
        # loss(theta, phi) = -E_q_phi[log(p_theta(xi|z)] + KL(q_phi(z|xi)||p(z))
        # Binary cross entropy loss
        BCE_loss = F.binary_cross_entropy(recon, x, reduction='sum')
        # KL Divergence loss between the output distribution and N(0,1)
        KL_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE_loss + 3 * KL_div_loss
        return loss, BCE_loss, KL_div_loss

    def loss(self, input, recons, mu, log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:       """

        kld_weight = 1# 0.00025  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = recons_loss + kld_weight * kld_loss
        return loss, recons_loss, kld_loss

    def forward(self, x):
        #TODO
        # encode image x to latent z
        z = self.encoder(x).view(-1, 64*7*7)
        # get mu and std of z
        mu = self.mu(z)
        logvar = self.logvar(z)
        # sample using the reparm trick
        z_s = self.z_sample(mu, logvar)
        # reconstruct the image using the sampled z
        recon = self.decoder(self.upsample(z_s).view(-1, 64, 7, 7))
        return recon, mu, logvar
