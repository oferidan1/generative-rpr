"""Training procedure for Generative-RPR.
"""

import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
from models.VAE import VAE
from models.vanilla_vae import VanillaVAE
from models.vq_vae import VQVAE
from models.vq_vae2 import VQVAE2
import matplotlib.pyplot as plt
import os
from util import utils
from datasets.CameraPoseDataset import CameraPoseDataset
from datasets.RelPoseDataset import RelPoseDataset
from datasets.KNNCameraPoseDataset import KNNCameraPoseDataset
from torch.nn.functional import normalize
from PIL import Image

def train(vae, trainloader, optimizer, ep, device, bPoseCondition):
    vae.train()  # set to training mode
    cnt = 0
    running_loss = 0
    for batch_idx, minibatch in enumerate(trainloader):
        for k, v in minibatch.items():
            minibatch[k] = v.to(device)
        inputs = minibatch['query']
        refs = minibatch['ref']
        rel_pose = minibatch['rel_pose']
        #assert(inputs.any()>=0 and inputs.any()<=1)
        bs = inputs.shape[0]
        optimizer.zero_grad()
        recon, mu, logvar = vae(inputs, rel_pose)
        if bPoseCondition:
            # incase of pose-condition, compare output to reference image
            loss, recon_loss, kd_loss = vae.loss(refs, recon, mu, logvar)
        else:
            # incase of no pose condition, compare output to input image
            loss, recon_loss, kd_loss = vae.loss(inputs, recon, mu, logvar)
        running_loss += loss.item() / bs
        if batch_idx % 10 == 0:
            print(f"Epoch {ep}: batch_idx {batch_idx}, recon_loss: {recon_loss.item()}, kd_loss: {kd_loss.item()}, loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        cnt += 1
    return running_loss / cnt

def main(args):
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    np.random.seed(numpy_seed)
    device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")

    #vae = VAE(latent_dim=args.latent_dim, device=device).to(device)
    #vae = VanillaVAE(in_out_channels=1, latent_dim=args.latent_dim, bPoseCondition=args.bPoseCondition).to(device)
    bSample = 0
    vae = VQVAE(1, 64, 512, bPoseCondition=args.bPoseCondition, beta=args.vq_beta).to(device)
    #vae = VQVAE2(in_channel=1, bPoseCondition=args.bPoseCondition, beta=args.vq_beta).to(device)

    if args.checkpoint_path:
        vae.load_state_dict(torch.load(args.checkpoint_path, map_location=device), strict=False)

    #train mode for generative-RPR project
    if args.mode == 'train':

        transform = utils.train_transforms_vae2.get('baseline')
        if '7scenes' in args.labels_file:
            train_dataset = RelPoseDataset(args.dataset_path, args.labels_file, transform)
        else:
            train_dataset = KNNCameraPoseDataset(args.dataset_path, args.labels_file, args.refs_file, args.knn_file, transform, 1)

        loader_params = {'batch_size': args.batch_size, 'shuffle': True,
                         'num_workers': args.num_workers}
        trainloader = torch.utils.data.DataLoader(train_dataset, **loader_params)
        optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

        train_loss = []
        for ep in range(args.epochs):
            train_loss.append(train(vae, trainloader, optimizer, ep, device, args.bPoseCondition))
            if bSample and (ep % 1 == 0):
                 samples = vae.sample(args.sample_size, device)
                 samples_grid = torchvision.utils.make_grid(samples)
                 torchvision.utils.save_image(samples_grid, './samples/sample' + '_epoch_%d.png' % ep)

        torch.save(vae.state_dict(), os.path.join(args.out_path, 'vae_model.pth'))
        fig, ax = plt.subplots()
        ax.plot(train_loss)
        ax.set_title("Train Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.savefig(os.path.join(os.getcwd(), ".", f"vae_loss.png"))
    else: #test mode
        # Set to eval mode
        vae.eval()
        # Set the dataset and data loader
        transform = utils.train_transforms_vae2.get('baseline')
        if '7scenes' in args.labels_file:
            test_dataset = RelPoseDataset(args.dataset_path, args.labels_file, transform)
        else:
            test_dataset = KNNCameraPoseDataset(args.dataset_path, args.labels_file, args.refs_file, args.knn_file, transform, args.knn_len)

        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': args.num_workers}
        dataloader = torch.utils.data.DataLoader(test_dataset, **loader_params)
        sum = 0
        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                inputs = minibatch['query']
                refs = minibatch['ref']
                rel_pose = minibatch['rel_pose']
                recon, _, _ = vae(inputs, rel_pose)

                if i < 5:
                    torchvision.utils.save_image(recon[0], './samples/output' + str(i) + '.jpg')
                    torchvision.utils.save_image(inputs[0], './samples/input' + str(i) + '.jpg')
                    torchvision.utils.save_image(refs[0], './samples/ref' + str(i) + '.jpg')
                    psnr = utils.PSNR(inputs[0]*255, recon[0]*255)
                    sum += psnr
                else:
                    break

        avg_psnr = sum / i
        print(avg_psnr)

#example of test command:
#python train_vae.py --mode=test --checkpoint_path=out_vae/vae_model.pth --bPoseCondition=0
if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    #parser.add_argument("--dataset_path", help="path to the physical location of the dataset", default="/nfstemp/Datasets/CAMBRIDGE_dataset/")
    parser.add_argument("--dataset_path", help="path to the physical location of the dataset", default="/data/datasets/7Scenes/")
    parser.add_argument("--labels_file", help="pairs file", default="datasets/7Scenes/7scenes_training_pairs_fire.csv")
    #parser.add_argument("--labels_file", help="pairs file", default="datasets/CambridgeLandmarks/cambridge_four_scenes.csv")
    parser.add_argument("--refs_file", help="path to a file mapping reference images to their poses", default="datasets/CambridgeLandmarks/cambridge_four_scenes.csv")
    parser.add_argument("--knn_file", help="path to a file mapping query images to their knns", default="datasets/CambridgeLandmarks/cambridge_StMarysChurch.csv_with_netvlads.csv")
    parser.add_argument('--batch_size', help='number of images in a mini-batch.', type=int, default=64)
    parser.add_argument('--sample_size', help='number of images to sample.', type=int, default=64)
    parser.add_argument('--epochs', help='maximum number of iterations.', type=int, default=20)
    parser.add_argument('--latent_dim', help='.', type=int, default=128)
    parser.add_argument('--num_workers', help='.', type=int, default=4)
    parser.add_argument('--knn_len', help='.', type=int, default=1)
    parser.add_argument('--bPoseCondition', help='.', type=int, default=0)
    parser.add_argument('--vq_beta', help='.', type=float, default=0.25)
    parser.add_argument('--lr', help='initial learning rate.', type=float, default=0.001)
    parser.add_argument('--reduction', help='reduction', default='reduction_3')
    parser.add_argument('--out_path', help='out_path', default='out_vae')
    parser.add_argument('--checkpoint_path', help='checkpoint_path')
    parser.add_argument('--mode', help='train/test', default='train')
    parser.add_argument('--gpu', help='gpu index', default='0')

    args = parser.parse_args()
    main(args)
