import torch
from torch import nn
from torch.nn import functional as F
from util import utils

class VanillaVAE(nn.Module):


    def __init__(self,
                 in_out_channels,
                 latent_dim,
                 bPoseCondition=False,
                 img_size=224):

        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.bPoseCondition = bPoseCondition
        self.img_size = img_size

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        in_channels = in_out_channels
        if bPoseCondition:
            in_channels += 1  # condition

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*49, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*49, latent_dim)


        # Build Decoder
        modules = []
        dec_in_channel = latent_dim
        if bPoseCondition:
            dec_in_channel += latent_dim

        self.decoder_input = nn.Linear(dec_in_channel, hidden_dims[-1] * 49)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= in_out_channels,
                                      kernel_size= 3, padding= 1),
                            #nn.Tanh())
                            nn.Sigmoid())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 7, 7)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, rel_pose):
        #encode the pose information and concat to input
        if self.bPoseCondition:
            rel_pose_emb1 = utils.positional_encoding(rel_pose.float(), num_encoding_functions=int(32), include_input=False, log_sampling=True)
            rel_pose_emb1 = rel_pose_emb1.repeat(1, 112)
            rel_pose_emb1 = rel_pose_emb1.view(-1, 1, self.img_size, self.img_size)
            input = torch.cat((input, rel_pose_emb1), dim=1)
        #encode the input
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        # encode the pose information and concat to latent
        if self.bPoseCondition:
            rel_pose_emb2 = utils.positional_encoding(rel_pose.float(), num_encoding_functions=int(32),
                                                      include_input=False, log_sampling=True)
            z = torch.cat((z, rel_pose_emb2), dim=1)
        #decode the image
        recon = self.decode(z)
        return recon, mu, log_var

    def loss(self, input, recons, mu, log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:       """


        kld_weight = 0.00025 # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)
        #recons_loss = F.binary_cross_entropy(recons, input, reduction='sum')

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        #kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = recons_loss + kld_weight * kld_loss
        return loss, recons_loss.detach(), kld_loss.detach()
        #return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples,
               current_device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)
        if self.bPoseCondition:
            pose = torch.zeros_like(z)
            z = torch.cat((z, pose), dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]