import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from torchvision.utils import save_image
#from torchvision import datasets, transforms
import matplotlib.pyplot as plt
#import torchvision.utils as vutils

#from torch.nn.functional import batch_norm

#from torch.autograd import Variable

import sys


from torch.distributions.normal import Normal
import torch.distributions as dist

from torch.cuda.amp import autocast, GradScaler

from math import sqrt

import torchvision

import time

import math

import argparse


def GMMSampling(mus, covarmat, weights_norm, num_samples=1024):

    num_gaus = mus.shape[0]

    com_samples = []
    for idx in range(num_gaus):
        mean = mus[idx].detach().cpu()
        cov = covarmat[idx].detach().squeeze().cpu()

        num = round(num_samples*weights_norm.squeeze()[idx].item())
        
        try:
            multivar_normal = torch.distributions.MultivariateNormal(mean, cov)
            samples = multivar_normal.sample(sample_shape=(num + 100,))
            samples = samples[:num]           
            com_samples.append(samples)
        except:
            print("invaild covarmat, num = ", num)

    com_samples = torch.cat(com_samples, dim=0)
    com_samples = com_samples.permute(1, 0)

    return com_samples


class KDE_gau(nn.Module):
    def __init__(self, bandwidth=0.1):
        super().__init__()
        
        self.bandwidth = bandwidth

    def forward(self, data, dim=-1, c_X=None):

        LEN = data.shape[dim]

        data = data.transpose(dim, -1)
        res = data.unsqueeze(-1) - c_X.unsqueeze(0)
      
        standard_gaussian = Normal(torch.zeros_like(res[(0,)*res.dim()]), torch.ones_like(res[(0,)*res.dim()]))
        pdf = (1/self.bandwidth)*standard_gaussian.log_prob(res/self.bandwidth).exp()

        pdf = pdf.sum(dim=1)/(LEN*1.0)
        pdf = pdf.permute(1, 0)

        return pdf



class MargGMM(nn.Module):
    def __init__(self):
        super(MargGMM, self).__init__()

    def forward(self, mus, covdata, weights, projvec, c_X=None):

        projmus = torch.tensordot(mus, projvec, dims=([1], [0]))
    
        covarmat = torch.bmm(covdata.permute(0, 2, 1), covdata)/covdata.shape[1]
        covarmat = covarmat + torch.diag(torch.ones_like(mus[0]))*(1e-20)
        

        exprojvec = torch.bmm(projvec.permute(1, 0).unsqueeze(-1), projvec.permute(1, 0).unsqueeze(1))
        projcovar = torch.tensordot(covarmat, exprojvec, dims=([1, 2], [1, 2]))

        weights_norm = weights.abs()/(weights.abs().sum() + 1e-20)

        try:
            projsigmas = projcovar**(0.5)
            standard_gaussian = Normal(projmus, projsigmas)
        except:
            projsigmas = projcovar.abs()**(0.5)
            standard_gaussian = Normal(projmus, projsigmas)
            print("accuracy error")

        c_X = (c_X - torch.zeros_like(projmus).unsqueeze(-1)).permute(2, 0, 1)
        gaupdf = standard_gaussian.log_prob(c_X).exp()

        gaupdf = torch.tensordot(gaupdf, weights_norm, dims=([1], [0])).squeeze()


        return gaupdf


class MargData(nn.Module):
    def __init__(self):
        super(MargData, self).__init__()

        self.KDE = KDE_gau()
    
    def forward(self, data, projvec, c_X=None):

        project_data = torch.tensordot(data, projvec, dims=([1], [0]))
        outpdf = self.KDE(project_data, dim=0, c_X=c_X)
        
        return outpdf


class ProjVec(nn.Module):
    def __init__(self, num_vecs, num_dims):
        super(ProjVec, self).__init__()

        projvec_sph = torch.rand(num_dims-1, num_vecs)
        self.register_buffer("projvec_sph", projvec_sph)

        projvec_gau = torch.randn(num_dims, num_vecs)
        self.register_buffer("projvec_gau", projvec_gau)

    def forward(self, gauvec=True):
        
        projvec = torch.randn_like(self.projvec_gau)
        normlen = ((projvec**2).sum(dim=0)).pow(0.5)
        projvec = projvec/normlen

        return projvec



class GMM(nn.Module):
    def __init__(self, num_gaus, num_dims):
        super(GMM, self).__init__()

        self.mus     = nn.Parameter(torch.randn(num_gaus, num_dims ))
        self.covdata = nn.Parameter((torch.diag_embed(torch.empty(num_gaus, num_dims).fill_(num_dims**(0.5))) + torch.randn(num_gaus, num_dims, num_dims)*1e-8))
        self.weights = nn.Parameter(torch.randn(num_gaus))

    def sampling(self, N):

        weights_norm = self.weights.abs()/(self.weights.abs().sum() + 1e-20)

        covarmat = torch.bmm(self.covdata.permute(0, 2, 1), self.covdata)/self.covdata.shape[1]
        covarmat = covarmat + torch.diag(torch.ones_like(self.mus[0]))*(1e-20)

        com_samples = GMMSampling(self.mus, covarmat, weights_norm, N)
        com_samples = com_samples.permute(1, 0)

        return com_samples[torch.randperm(com_samples.shape[0])]



class MCMarg(nn.Module):
    def __init__(self, num_vecs, num_dims, num_gaus, maxval=6.0, bins=64):
        super(MCMarg, self).__init__()

        self.num_vecs = num_vecs
        self.num_dims = num_dims
        self.num_gaus = num_gaus

        self.maxval = maxval
        self.bins = bins
        
        c_X = torch.linspace(-maxval, maxval, bins)
        self.register_buffer("c_X", c_X)

        self.projVec    = ProjVec(num_vecs, num_dims)
        self.margGMM    = MargGMM()
        self.margData   = MargData()


    def forward(self, gmm, data, projvec=None):

        if projvec == None: projvec = self.projVec()

        gaupdf = self.margGMM(gmm.mus, gmm.covdata, gmm.weights, projvec, self.c_X)
        outpdf = self.margData(data, projvec, self.c_X)
       
        return gaupdf, outpdf, projvec



def biKLLoss(gauhist, outhist):

    loss_kl_out = outhist*(torch.log(outhist + 1e-20) - torch.log(gauhist + 1e-20))
    loss_kl_gau = gauhist*(torch.log(gauhist + 1e-20) - torch.log(outhist + 1e-20))
    
    loss_kl_out = loss_kl_out.sum(dim=0)
    loss_kl_gau = loss_kl_gau.sum(dim=0)

    loss_kl = F.relu(loss_kl_out) + F.relu(loss_kl_gau)
    loss_kl = loss_kl.sum()

    return loss_kl



def main(argc, argv):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath",    type=str, default='./data/samples_moons.pt')
    parser.add_argument("--batch_size",    type=int, default=1024*8)
    parser.add_argument("--num_epochs",    type=int, default=1001)

    
    args = parser.parse_args()
    
    trainz = torch.load(args.datapath) #'./samples_moons.pt')

    latent_dim = trainz.shape[1]

    num_dims = trainz.shape[1]
    num_gaus = 128
    num_projs = 16

    bins = 256


    gmm = GMM(num_gaus=num_gaus, num_dims=num_dims).to(device)

    optim_gau = torch.optim.Adam(gmm.parameters(), lr=1e-2)


    maxval = (trainz**2).sum(dim=1).max().sqrt()*1.1


    projectdis = MCMarg(num_projs, num_dims, num_gaus, maxval, bins).to(device)


    batch_size = args.batch_size
    num_epochs = args.num_epochs


    torch.manual_seed(999)

    trainz = trainz.to(device)


    plt.figure(figsize = (12, 5.6))

    for epoch in range(num_epochs):
        idx_imgs = torch.randperm(trainz.shape[0])

        for i in range(round(trainz.shape[0]/batch_size + 0.4999999999)):
            batch_idx = idx_imgs[i*batch_size:(i+1)*batch_size]
            z = trainz[batch_idx]

            gauhist, outhist, _ = projectdis(gmm, z)
            
            loss_kl = biKLLoss(gauhist, outhist)

            loss = loss_kl

            optim_gau.zero_grad()
            loss.backward(retain_graph=True)
            optim_gau.step()


        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, loss_kl: {loss_kl.item():.4f}")


        if epoch % 100 == 0:
            

            torch.save(gmm.state_dict(), './gmm.ckpt')

            
            N = trainz.shape[0]
            com_samples = gmm.sampling(N)[:N]

            plt.clf()
            
            showlval = -8.0
            showrval = 8.0


            delta = (maxval*2)/bins


            plt.subplot(1, 2, 1)
            plt.plot(trainz[:, 0].detach().cpu().numpy(), trainz[:, 1].detach().cpu().numpy(), '.', color=(0.1, 0.1, 0.1, 0.1), markersize=1)
            plt.xlim(showlval,showrval)
            plt.ylim(showlval,showrval)

            plt.subplot(1, 2, 2)
            plt.plot(com_samples[:, 0].detach().cpu().numpy(), com_samples[:, 1].detach().cpu().numpy(), '.', color=(0.1, 0.1, 0.1, 0.1), markersize=1)
            plt.xlim(showlval,showrval)
            plt.ylim(showlval,showrval)
            
            plt.tight_layout(pad=2.4)

            plt.pause(0.8)




if __name__ == '__main__':
    argc = len(sys.argv)
    argv = sys.argv
    

    main(argc, argv)
