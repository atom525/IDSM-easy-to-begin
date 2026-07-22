import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch import autograd
from colorMap import parula
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import time
import network as NNs
colourMap = parula()  # plt.cm.jet
import scipy.io as scio

print(device)
import torchvision
#import chainer
from scipy.io import savemat
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

#train, test = chainer.datasets.get_mnist(ndim=1)
#print(train[0][0])
#image64 = np.zeros([30000,64,64])
#for i in range(30000):
#    for j in range(64):
#        for k in range(64):
#            z1 = int(j * 28/64)
#            z2 = int(k * 28/64)
#            image64[i,j,k] = train[i][0].reshape(28, 28) [z1][z2]
#image64 = np.where(image64>0.3,1.5,1)
#image64 = {'contrast':image64}
#savemat('Mnist64.mat', image64)
#kkk3


Abs = True
dataFile = 'data/Phaseless_MnistRotaCir_12000PS2.mat'
data = scio.loadmat(dataFile)
N_in = 16   
N_re = 100     #number of receivers
index = [int(i * (16/N_in)) for i in range(N_in)]   # Number of used incidences
print(index)

Gs = torch.tensor(data['R_mat'])
norm_Gs = torch.abs(torch.mean(Gs * Gs) ** 0.5)
Gs = Gs/norm_Gs   # Fundamental solution, here I normalize it 


# incident wave, scattered wave, phaseless total wave
E_i = torch.tensor(data['E_i'])[:,index,:] 
E_s = torch.tensor(data['E_s'])[:,index,:]
E_t = torch.abs(E_i + E_s)

#The correction term
deltaE = (torch.abs(E_t) ** 2 - torch.abs(E_i) ** 2)/E_i


# Add noise to E_t
E_t_noise = torch.zeros_like(E_t)
noise = 0.01
for i in range(E_s.shape[2]):
    coeff = noise * torch.norm(E_t[:,:,i])*(1/(N_in *N_re)**0.5)
    E_t_noise[:,:,i] = coeff * torch.randn_like(E_t[:,:,i]) + E_t[:,:,i]
deltaE_Noise = (E_t_noise ** 2 - torch.abs(E_i) ** 2)/E_i

e = torch.randn_like(E_s)

# True contrast
Contrast = torch.tensor(data['Contrast'])
Contrast = Contrast.reshape(-1,1,64,64)


# Compute the indicator functions
IndFun = torch.matmul(torch.t(deltaE.reshape(N_re,-1)), Gs).reshape(N_in,-1,64,64)
IndFun = torch.transpose(IndFun,0,1)
IndFun = torch.transpose(IndFun,2,3)
IndFun = torch.abs(IndFun)


IndFun_Noise = torch.matmul(torch.t(deltaE_Noise.reshape(N_re,-1)), Gs).reshape(N_in,-1,64,64)
IndFun_Noise = torch.transpose(IndFun_Noise,0,1)
IndFun_Noise = torch.transpose(IndFun_Noise,2,3)
IndFun_Noise = torch.abs(IndFun_Noise)


a = torch.max(IndFun)

print(a)

mm = 500.0


# divide the indicator by 500 so that the indicator functions can take moderate values.
IndFun = 2.0 * IndFun/mm
IndFun_Noise = 2.0* IndFun_Noise/mm




# U-Net
unet = NNs.U_Net3Ab(img_ch=N_in,output_ch=1, N_ch=64).to(device)

mm = NNs.U_Net3Ab(img_ch=1,output_ch=1, N_ch=64).to(device)

# conut the number of parameters in the NN
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    
print("Total trainable parameters:", count_parameters(mm))


lr = 0.001
optimizers2 = torch.optim.Adam([{'params': unet.parameters(), 'lr': lr}
                              ])
scheduler=torch.optim.lr_scheduler.StepLR(optimizers2, 3, gamma=0.5, last_epoch=-1)
loss_func = torch.nn.MSELoss()

batch_size = 10
batch_number = 1000
mi = torch.tensor(1.0)
for i in range(30):
    unet.to(device)
    for j in range(batch_number):
        input = torch.abs(IndFun_Noise[batch_size*j:batch_size*(1+j)])
    
        output_nn = unet(input.to(device))
        g1 = output_nn[:,:,0:-2,:] - output_nn[:,:,1:-1,:]
        g2 = output_nn[:,:,:,0:-2] - output_nn[:,:,:,1:-1]
        Val = torch.mean(torch.abs(g1))+torch.mean(torch.abs(g2))
        output = Contrast[batch_size*j:batch_size*(1+j)].to(device)
        ss = ssim(output_nn,output.to(device),data_range=1.7,size_average=True)
        
        # The loss function consists of three terms
        Loss = loss_func(output_nn, output) + 0.05 * Val + 0.05 * (1-ss)
        optimizers2.zero_grad()
        Loss.backward()
        optimizers2.step()
    print('Loss',i, Loss,Val,ss)
    scheduler.step()

    #Compute the training error and testing error
    input_noise = (torch.abs(IndFun_Noise[10500:10699]))
    unetc= unet.cpu()
    output_nn_noise = unetc(input_noise).cpu()  
    output_test = Contrast[10500:10699]
    error_test = torch.mean(torch.norm(output_nn_noise-output_test, dim=[2,3])/torch.norm(output_test, dim=[2,3]))
    error_train = torch.mean(torch.norm(output_nn-output, dim=[2,3])/torch.norm(output, dim=[2,3]))
    print('error', error_train, error_test,mi)
    if error_test < mi:
        mi = error_test
        torch.save(unet, "model/UNETCircle-%s-Nin.pkl"%(N_in))

    for k in range(10):
        fig = plt.figure(figsize=(20, 5))
        colourMap = parula()  # plt.cm.jet
        plt.subplot(1, 3, 1)
        plt.xlabel('x')  # , fontsize=16, labelpad=15)
        plt.ylabel('y')  # , fontsize=16, labelpad=15)
        plt.title("Reconstruction")
        m = torch.nn.Threshold(1.0, 1.0)
        output_nn2 = m(output_nn_noise[k])
        plt.imshow(output_nn2.detach().reshape(64,64).cpu(), cmap=colourMap, extent=[xmin, xmax, ymin, ymax],
                   origin='lower',
                   aspect='auto')  # , vmin=0, vmax=1, )
        plt.colorbar()


        plt.subplot(1, 2, 2)
        plt.xlabel('x')  # , fontsize=16, labelpad=15)
        plt.ylabel('y')  # , fontsize=16, labelpad=15)
        plt.title("Exact")
        plt.imshow(output_test[k].reshape(64,64).cpu(), cmap=colourMap, extent=[xmin, xmax, ymin, ymax],
                   origin='lower',
                   aspect='auto')  # , vmin=0, vmax=1, )
        plt.colorbar()
      
      
        plt.savefig('WU/figure-%s'%(k),dpi=100)
        #plt.show()

