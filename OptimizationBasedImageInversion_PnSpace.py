import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
# import lpips
from model import Generator
import numpy as np
from torchvision import utils
import lpips
import cv2
import time
from PCA_utils import IPCAEstimator


def ShowImageAtRunTime(inputTensor):
    x = torch.permute(inputTensor[0], (1, 2, 0)).cpu().detach().numpy()
    x = (x -  np.min(x))/(np.max(x) - np.min(x))
    plt.imshow(x)
    plt.show()

def ShowImageUsingOpenCV(inputTensor, waitTime = 0.01):
    '''
    Input: Tensor of shape (3, W, H)

    This Allows Images to be displayed as the loop runs

    Note: OpenCV allow BGR and not RGB.
    '''
    x = torch.permute(inputTensor[0], (1, 2, 0)).cpu().detach().numpy()

    x = x[...,::-1]
    x = (x -  np.min(x))/(np.max(x) - np.min(x))
    cv2.imshow("Predicted Image",x)
    time.sleep(waitTime)
def SaveImageFromTensor(sample, path):
    utils.save_image(
        sample,
        path,
        nrow=1,
        normalize=True,
        range=(-1, 1),
    )
    return





## Taken from II2S Repo! Modifeied!
def build_PCA_model(PCA_path):
    with torch.no_grad():
        latent = torch.randn((1000, 512))
        # latent = torch.randn((10000, 512), dtype=torch.float32)
        g_ema.style.cpu()
        pulse_space = torch.nn.LeakyReLU(5)(g_ema.style(latent)).numpy()
        g_ema.style.to(device)

    transformer = IPCAEstimator(512)
    X_mean = pulse_space.mean(0)
    transformer.fit(pulse_space - X_mean)
    X_comp, X_stdev, X_var_ratio = transformer.get_components()
    np.savez(PCA_path, X_mean=X_mean, X_comp=X_comp, X_stdev=X_stdev, X_var_ratio=X_var_ratio)


def load_PCA_model(PCA_path):
    device = self.opts.device

    # PCA_path = self.opts.ckpt[:-3] + '_PCA.npz'

    # if not os.path.isfile(PCA_path):
    #     self.build_PCA_model(PCA_path)

    # PCA_model = np.load(PCA_path)
    # self.X_mean = torch.from_numpy(PCA_model['X_mean']).float().to(device)
    # self.X_comp = torch.from_numpy(PCA_model['X_comp']).float().to(device)
    # self.X_stdev = torch.from_numpy(PCA_model['X_stdev']).float().to(device)



def plotDistributions(axis = [0, 1], space = "Z"):
    '''
    Saud: Enter Space name as a string, options are Z, W, and P.
    '''
    with torch.no_grad():
        latent = torch.randn((1000, 512))


        if space == "Z":
            plt.scatter(latent[:, axis[0]], latent[:, axis[1]], s = 0.1)
            plt.ylim(-5, 5)
            plt.xlim(-5,5)
            plt.grid()
            plt.show()
            return

        g_ema.style.cpu()
        w = g_ema.style(latent)
        if space == "W":
            plt.scatter(w[:, axis[0]], w[:, axis[1]], s = 0.1)
            plt.ylim(-5, 5)
            plt.xlim(-5,5)
            plt.grid()
            plt.show()
            return
        # latent = torch.randn((10000, 512), dtype=torch.float32)
        pulse_space = torch.nn.LeakyReLU(5)(w).numpy()
        g_ema.style.to(device)
        if space == "P":
            plt.scatter(pulse_space[:, axis[0]], pulse_space[:, axis[1]], s = 0.1)
            plt.ylim(-5, 5)
            plt.xlim(-5,5)
            plt.grid()
            plt.show()
            return




if __name__ == "__main__":

    print("Made w Love")
    device = "cuda"


    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8



    #Load Model
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"], strict=False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None




    pca_path = "PCA_Matrices.npz"

    build_PCA_model(pca_path)





    PCA_model = np.load(pca_path)
    X_mean = torch.from_numpy(PCA_model['X_mean']).float().to(device) # Size: (512)
    X_comp = torch.from_numpy(PCA_model['X_comp']).float().to(device) #Size: (512, 512)
    X_stdev = torch.from_numpy(PCA_model['X_stdev']).float().to(device) #Size: (512)


    z = torch.randn((1000, 512)).unsqueeze(0).to(device)
    w = g_ema.style(z)

    pn = (torch.nn.LeakyReLU(negative_slope=5)(w) - X_mean)

    print(pn.size())
    pn = pn.bmm(X_comp.T.unsqueeze(0)) / X_stdev


#
#     #Generate Target or Load from Image
#     with torch.no_grad():
#         g_ema.eval()
#         sample_z = torch.from_numpy(np.random.RandomState(44).randn(1, 512)).to(device)
#         print(sample_z[0][0])
#         sample_z = sample_z.float()
#         GroundTruth, _ = g_ema([sample_z], truncation_latent=None)
#
#
#
#     # ShowImageAtRunTime(GroundTruth)
#
#     #This is the Z that we are trying to learn
#     learnedZ = torch.randn(args.sample, args.latent, device=device)
#
#     #Require Gradients
#     learnedZ.requires_grad = True
#
#     #Define optimizer
#     optimizer = optim.Adam([learnedZ], lr=0.1)
#
#     #Define your loss
#     criterion = lpips.PerceptualLoss(
#         model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
#     )
#
#
#     for i in range(2000):
#         optimizer.zero_grad()
#
#         y, _ = g_ema([learnedZ], truncation_latent=None)
#         loss = criterion(y, GroundTruth).sum()
#
#
#         loss.backward()
#         optimizer.step()
#
#
#         print(f'i = {i} -- loss = {loss.item()}')
#
#
#
#         #This controls the learning rate
#         if i % 1000 == 0 and i != 0:
#             optimizer.param_groups[0]["lr"] *= 0.5
#
#
#
#         #If you dont want to view images as the loop runs, comment this block
#         ShowImageUsingOpenCV(y, 0.01)
#         if cv2.waitKey(1) == ord('q'):
#                 break
#
#
#         #Save the images!
#         if i % 50 == 0 and i != 0:
#             Save_path = f'learningZSpace/{i}.png'
#             SaveImageFromTensor(y, Save_path)
#             print("Saved Successfuly at ", Save_path)
#
#
#
#
# #Make sure to destroy all windows to avoid laggy feeling.
# cv2.destroyAllWindows()
