import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
import torchvision
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
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

def ShowImageAtRunTime(inputTensor):
    x = torch.permute(inputTensor, (1, 2, 0)).numpy()
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
    # time.sleep(waitTime)


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
    # print(g_ema([latent.cuda()], truncation_latent=None)[0].size())
    transformer = IPCAEstimator(64)
    X_mean = pulse_space.mean(0)
    transformer.fit(pulse_space - X_mean)
    X_comp, X_stdev, X_var_ratio = transformer.get_components()
    np.savez(PCA_path, X_mean=X_mean, X_comp=X_comp, X_stdev=X_stdev, X_var_ratio=X_var_ratio)


#Save as pkl so that it connects with StyleFlow pipeline easily.
def SavePickle(latent, path):
    dictt = {}
    dictt["Latent"] = latent
    with open(path, 'wb') as handle:
        pickle.dump(dictt, handle, protocol=3)



def cal_p_norm_loss(w):

    pn = (torch.nn.LeakyReLU(negative_slope=5)(w) - X_mean)
    pn = pn.bmm(X_comp.T.unsqueeze(0)) / X_stdev

    return pn.pow(2).mean()



if __name__ == "__main__":




    #########################################################################################
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
    #########################################################################################

    #Load CelebA
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    dataset = torchvision.datasets.ImageFolder('data/celeba_sample', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    #########################################################################################


    #Load Model
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None
    #########################################################################################

    #Set up your PCA Matrices, Make sure to Modify Number of Components.
    pca_path = "PCA_Matrices.npz"
    build_PCA_model(pca_path)
    PCA_model = np.load(pca_path)
    X_mean = torch.from_numpy(PCA_model['X_mean']).float().to(device) # Size: (512)
    X_comp = torch.from_numpy(PCA_model['X_comp']).float().to(device) #Size: (512, 512)
    X_stdev = torch.from_numpy(PCA_model['X_stdev']).float().to(device) #Size: (512)
    #########################################################################################


    #Define your LPIPS loss term
    loss_LPIPS = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )


    All_Ws = []

    for index in range(4):


        #Set up starting vector.
        with torch.no_grad():
            g_ema.eval()
            StartingZ = torch.randn(1, args.latent, device=device)
            LearnedW = g_ema.style(StartingZ)


        #Generate Target or Load from Image
        GroundTruth, _ = next(iter(dataloader))
        GroundTruth = (GroundTruth.cuda())


        print(GroundTruth.size())
        SaveImageFromTensor(GroundTruth, f'sample/{index}.png')



        #Require Gradients
        LearnedW.requires_grad = True

        optimizer = optim.Adam([LearnedW], lr=0.01)


        #LPIPS,      Pn,      MSE
        lambda1, lambda2, lambda3 = 1,0.1, 1

        Ws = None




        for i in range(1300):
            optimizer.zero_grad()


            y, _ = g_ema([LearnedW], truncation_latent=None, input_is_latent=True)


            #Set up the PN loss
            pn = (torch.nn.LeakyReLU(negative_slope=5)(LearnedW) - X_mean)
            pn = (pn @ X_comp.T )/ X_stdev


            pn_loss = (pn.squeeze(0))[:-1].pow(2).sum()
            lpips_loss = loss_LPIPS(y, GroundTruth).sum()
            mse_loss = F.mse_loss(y, GroundTruth)
            loss = lambda1 * lpips_loss  + lambda2 * pn_loss + lambda3 * mse_loss


            loss.backward()
            optimizer.step()


            print(f'i = {i} -- loss = {loss.item()} -- lpips = {lambda1 * lpips_loss.item()} -- pn = {lambda2 * pn_loss.item()} -- MSE = {lambda3 * mse_loss.item()}')



            #This controls the learning rate
            if i % 250 == -1:
                optimizer.param_groups[0]["lr"] *= 0.5



            #If you dont want to view images as the loop runs, comment this block
            ShowImageUsingOpenCV(y, 0.01)
            if cv2.waitKey(1) == ord('q'):
                    break


            #Save the images!
            if i % 100 == 0:
                lambda2 *= 1.15
                Save_path = f'learningPnSpace_001/{index}_{i}.png'
                SaveImageFromTensor(y, Save_path)

                Ws = LearnedW.unsqueeze(1).repeat(1, 18, 1)
                Ws = Ws.unsqueeze(0)
                # SavePickle(Ws.tolist(), f'pkls/pn_{index}.pickle')
                # print("Saved Successfuly at ", Save_path)

        if Ws != None:
            print("  --- One Array Has Been Appended! ----   ")
            All_Ws.append(Ws[0].tolist())
            SavePickle(All_Ws, f'pkls/pn_All_WS_External_64.pickle')






#Make sure to destroy all windows to avoid laggy feeling.
cv2.destroyAllWindows()
