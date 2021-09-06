import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import *

###################################################################################################
######## This code was developed by Charlie Tan, Student @ University of Bristol, UK, 2021 ########
###################################################################################################

class Generator(nn.Module):
    def __init__(self, in_chan=3, base_chan=64, n_mul2res=7):
        super().__init__()

        self.name = "generator"
        self.n_mul2res = n_mul2res

        # 2 shallow feature extraction layers
        self.conv1 = nn.Sequential(nn.Conv2d(in_chan, base_chan, kernel_size=3, padding=1), nn.Mish())
        self.conv2 = nn.Sequential(nn.Conv2d(base_chan, base_chan, kernel_size=3, padding=1), nn.Mish())

        self.ERNB1 = ERNB(base_chan)
        
        # MulSquaredRes blocks

        self.mul2res_list = nn.ModuleList()
        self.mul2res_conv_list = nn.ModuleList()

        # iterate 2 to 9, this defines the conv layer input channels as 128 to 576 (assuming 7 Mul2Res)
        for i in range(2, self.n_mul2res+2):
            self.mul2res_list.append(MulSquaredResBlock(base_chan)) 
            self.mul2res_conv_list.append(nn.Sequential(nn.Conv2d(i*base_chan, base_chan, kernel_size=1, padding=0), nn.Mish()))

        # RL1, RL2, ECBAM
        self.RL1_RL2_ECBAM = nn.Sequential(
                nn.Conv2d(base_chan, base_chan, kernel_size=3, padding=1), 
                nn.Mish(),
                nn.Conv2d(base_chan, base_chan, kernel_size=3, padding=1),
                nn.Mish(),
                ECBAM(base_chan, reduction_ratio=4)
                )

        # ERNB, RL3, ECBAM
        self.ERNB_RL3_ECBAM = nn.Sequential(
                ERNB(base_chan),
                nn.Conv2d(base_chan, base_chan, kernel_size=3, padding=1), 
                nn.Mish(),
                ECBAM(base_chan, reduction_ratio=4),
                )

        # Final conv layer
        self.final_conv = nn.Sequential(nn.Conv2d(base_chan, in_chan, kernel_size=3, padding=1), nn.Tanh())

    def forward(self, x):

        # interpolate input
        x = F.interpolate(x, scale_factor=2)

        # 2 shallow feature extraction layers
        step = self.conv1(x)
        step = self.conv2(step)

        # MulSquaredResBlocks
        # list of skip connections
        mul2res_input = self.ERNB1(step)

        # add first input to list of skip connections
        concat_list = [mul2res_input]

        for i in range(self.n_mul2res):
            # ith MulSquaredResBlock
            step = self.mul2res_list[i](mul2res_input)

            # add the output to the list of skip connections
            concat_list.append(step)

            # concatenate list of skip connections (includes the output of this stage)
            step = torch.cat(concat_list, dim=1)
            step = self.mul2res_conv_list[i](step)
            
            # input for the next iteration
            mul2res_input = step
        
        # RL1, RL2, ECBAM
        step = self.RL1_RL2_ECBAM(step)
        
        # add first mul2res block input
        step = step + concat_list[0]
        
        # ERNB, RL3, ECBAM
        step = self.ERNB_RL3_ECBAM(step)
    
        # final convolution
        step = self.final_conv(step)

        step = step + x

        return torch.clamp(step, min = 0, max = 1)

class Discriminator(nn.Module):
    "model adapted from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/srgan/srgan.py"
    def __init__(self, input_shape):
        super().__init__()

        self.name = "discrim"

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape

        def discriminator_block(in_filters, out_filters):
            layers = []

            # only have bias on first conv2d (all others precede a BN layer)
            if out_filters == 64:
                var_bias = True
            else:
                var_bias = False

            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=var_bias))

            # original implementation has activation before BN
            layers.append(nn.LeakyReLU(0.2))
            
            # first block has ERNB in place of BN
            if out_filters == 64:
                layers.append(ERNB(out_filters))
            else:
                layers.append(nn.BatchNorm2d(out_filters))

            # last block has ERNB before 2nd conv
            if out_filters == 512:
                layers.append(ERNB(out_filters))

            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1, bias=False))

            # original implementation has activation before BN
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm2d(out_filters))

            return layers

        layers = []

        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters))
            in_filters = out_filters

        layers.append(nn.AvgPool2d(2, 2))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(4608,1024))
        layers.append(nn.Flatten())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
