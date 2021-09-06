import torch
import torch.nn as nn
from piqa import ssim

###################################################################################################
######## This code was developed by Charlie Tan, Student @ University of Bristol, UK, 2021 ########
###################################################################################################

class PLF(nn.Module):
    "perceptual loss function as defined in the cvegan paper"
    def __init__(self):
        super().__init__()

        self.L1 = nn.L1Loss()
        self.MSE = nn.MSELoss()

        # comparing each channel independently
        self.SSIM = ssim.SSIM(n_channels=1)

        # last weight removed from official, scale too large for 96 x 96 
        self.MS_SSIM = ssim.MS_SSIM(n_channels=1, weights=torch.tensor((0.0448, 0.2856, 0.3001, 0.2363)))
        
    def forward(self, target, gen_output):
  
        # tuples of the different channel gen outputs and targets, slicing 0:1 instead of 0 to preserve dim
        tuple_Y = (gen_output[:,0:1,:,:], target[:,0:1,:,:])
        tuple_U = (gen_output[:,1:2,:,:], target[:,1:2,:,:])
        tuple_V = (gen_output[:,2:3,:,:], target[:,2:3,:,:])

        # weighted average of loss of different channels (comparing elements in each tuple)
        L1 = (self.L1(*tuple_Y)*4 + self.L1(*tuple_U) + self.L1(*tuple_V))/6
        MSE = (self.MSE(*tuple_Y)*4 + self.MSE(*tuple_U) + self.MSE(*tuple_V))/6
        SSIM = 1 - (self.SSIM(*tuple_Y)*4 + self.SSIM(*tuple_U) + self.SSIM(*tuple_V))/6
        MS_SSIM = 1 - (self.MS_SSIM(*tuple_Y)*4 + self.MS_SSIM(*tuple_U) + self.MS_SSIM(*tuple_V))/6

        PLF = 0.3 * torch.log(L1) + 0.1 * torch.log(MSE) + 0.2 * torch.log(SSIM) + 0.4 * torch.log(MS_SSIM) 

        return PLF, L1, MSE, SSIM, MS_SSIM

def inverse_stereographic_projection(x):
    x_u = torch.transpose(2 * x, 0, 1) / (torch.pow(torch.linalg.norm(x, axis=1), 2) + 1.0)
    x_v = (torch.pow(torch.linalg.norm(x_u, axis=0, keepdim=True), 2) - 1.0) / (torch.pow(torch.linalg.norm(x_u, axis=0, keepdim=True), 2) + 1.0)
    x_projection = torch.transpose(torch.cat([x_u, x_v], axis=0), 0, 1)

    return x_projection

def sphere_loss(x, y):
    batch_dot = (x * y).sum(-1)
    return torch.acos(batch_dot)

class GenReSphere(nn.Module):
    "generator GAN loss function as defined in the cvegan paper"
    def __init__(self, batch_size, rank, moment = 3):
        super().__init__()

        self.moment = moment
        dims = 1024
        self.north_pole = torch.nn.functional.one_hot(torch.tensor([dims]*batch_size)).to(rank)

    def forward(self, real, fake):
        
        fake_loss_1 = 0
        fake_loss_2 = 0

        real_projection = inverse_stereographic_projection(real)
        fake_projection = inverse_stereographic_projection(fake)

        for i in range(1, self.moment+1):
            fake_loss_1 -= torch.mean(torch.pow(sphere_loss(fake_projection, self.north_pole), i))
            fake_loss_2 += torch.mean(torch.pow(sphere_loss(fake_projection, real_projection), i))

        loss = fake_loss_1 + fake_loss_2

        return loss

class DiscReSphere(nn.Module):
    "discriminator loss function as defined in the cvegan paper"
    def __init__(self, batch_size, rank, moment = 3):
        super().__init__()

        self.moment = moment
        dims = 1024
        self.north_pole = torch.nn.functional.one_hot(torch.tensor([dims]*batch_size)).to(rank)

    def forward(self, real, fake):

        real_loss = 0
        fake_loss = 0
        fake_loss_plus = 0

        real_projection = inverse_stereographic_projection(real)
        fake_projection = inverse_stereographic_projection(fake)

        for i in range(1, self.moment+1):
            real_loss -= torch.mean(torch.pow(sphere_loss(real_projection, self.north_pole), i))
            fake_loss += torch.mean(torch.pow(sphere_loss(fake_projection, self.north_pole), i))
            fake_loss_plus -= torch.mean(torch.pow(sphere_loss(fake_projection, real_projection), i)) 

        loss = real_loss + fake_loss + fake_loss_plus

        return loss

