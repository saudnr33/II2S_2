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




    #Generate Target or Load from Image
    with torch.no_grad():
        g_ema.eval()
        sample_z = torch.from_numpy(np.random.RandomState(44).randn(1, 512)).to(device)
        print(sample_z[0][0])
        sample_z = sample_z.float()
        GroundTruth, _ = g_ema([sample_z], truncation_latent=None)



    # ShowImageAtRunTime(GroundTruth)

    #This is the Z that we are trying to learn, 14 is the number of hidden layers
    learnedWPlus = torch.randn(args.sample,14, args.latent, device=device)

    #Require Gradients
    learnedWPlus.requires_grad = True

    #Define optimizer
    optimizer = optim.Adam([learnedWPlus], lr=0.1)

    #Define your loss
    criterion = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )


    for i in range(4000):
        y, _ = g_ema([learnedWPlus], truncation_latent=None, inputInWPlusSpace = True)
        loss = criterion(y, GroundTruth).sum()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print(f'i = {i} -- loss = {loss.item()}')



        #This controls the learning rate
        if i % 1000 == -1:
            optimizer.param_groups[0]["lr"] *= 0.5



        #If you dont want to view images as the loop runs, comment this block
        ShowImageUsingOpenCV(y, 0.01)
        if cv2.waitKey(1) == ord('q'):
                break


        #Save the images!
        if i % 50 == -1:
            Save_path = f'learningZSpace/{i}.png'
            SaveImageFromTensor(y, Save_path)
            print("Saved Successfuly at ", Save_path)




#Make sure to destroy all windows to avoid laggy feeling.
cv2.destroyAllWindows()
