import torch
from torch import nn
from torch.nn import functional as F
from util import utils

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input):
        return input + self.resblock(input)


class VQVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 bPoseCondition = False,
                 hidden_dims = None,
                 beta: float = 0.25,
                 img_size: int = 224,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta
        self.bPoseCondition = bPoseCondition

        in_out_channels = in_channels
        if bPoseCondition:
            in_channels += 1 #condition

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]
            #hidden_dims = [192, 384]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)

        self.rel_pose_embedding1 = nn.Linear(7, img_size*img_size)
        self.rel_pose_embedding2 = nn.Linear(7, img_size//4*img_size//4)

        # Build Decoder
        modules = []
        if bPoseCondition:
            embedding_dim += 1
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=in_out_channels,
                                   kernel_size=4,
                                   stride=2, padding=1),
                #nn.Tanh()))
                nn.Sigmoid()))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input, rel_pose):
        # encode the pose information and concat to input
        if self.bPoseCondition:
            #rel_pose_emb1 = self.rel_pose_embedding1(rel_pose.float()).view(-1, 1, self.img_size, self.img_size)
            rel_pose_emb1 = utils.positional_encoding(rel_pose.float(), num_encoding_functions=int(32), include_input=False, log_sampling=True)
            rel_pose_emb1 = rel_pose_emb1.repeat(1, 112)
            rel_pose_emb1 = rel_pose_emb1.view(-1, 1, self.img_size, self.img_size)
            input = torch.cat((input, rel_pose_emb1), dim=1)
        #encode the input
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        # encode the pose information and concat to latent
        if self.bPoseCondition:
            #rel_pose_emb2 = self.rel_pose_embedding2(rel_pose.float()).view(-1, 1, self.img_size//4, self.img_size//4)
            rel_pose_emb2 = utils.positional_encoding(rel_pose.float(), num_encoding_functions=int(32), include_input=False, log_sampling=True)
            rel_pose_emb2 = rel_pose_emb2.repeat(1, 7)
            rel_pose_emb2 = rel_pose_emb2.view(-1, 1, self.img_size // 4, self.img_size // 4)
            quantized_inputs = torch.cat((quantized_inputs, rel_pose_emb2), dim=1)
        #decode the image
        recon = self.decode(quantized_inputs)
        #recon = recon/2+1
        return [recon, vq_loss, input]

    def loss(self, input, recons, vq_loss, log_var=0):
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return loss, recons_loss.detach(), vq_loss.detach()

    def sample(self,
               num_samples: int,
               current_device):
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]