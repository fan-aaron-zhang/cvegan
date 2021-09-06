import torch
import torch.nn as nn

import cbam

###################################################################################################
######## This code was developed by Charlie Tan, Student @ University of Bristol, UK, 2021 ########
###################################################################################################

class ECBAM(nn.Module):
    "adapted from https://github.com/Jongchan/attention-module"
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=['avg', 'max']):
        super().__init__()

        self.ChannelGate = cbam.ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = cbam.SpatialGate()

        self.conv = nn.Sequential(nn.Conv2d(gate_channels*2, gate_channels, kernel_size=1, padding=0), nn.Mish())

    def forward(self, x):

        step = self.ChannelGate(x)
        step = self.SpatialGate(step)
        step = torch.cat((step, x), dim = 1)
        step = self.conv(step)

        return step

class FeatureFusion(nn.Module):
    "very similar to the MulResBranch but kept seperate in case changes made to either"

    def __init__(self, in_chan, filters, kernel_size, padding):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_chan, filters, kernel_size, padding=padding), nn.Mish())
        self.conv2 = nn.Sequential(nn.Conv2d(filters, filters, kernel_size, padding=padding), nn.Mish())
        self.conv3 = nn.Conv2d(filters, filters, kernel_size, padding=padding) 

    def forward(self, x):
        
        hold = self.conv1(x)
        step = self.conv2(hold)
        step = self.conv3(step)
        step = step + hold

        return step

class ERNB(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        
        branch_chan = in_chan//2
        
        self.branch1 = nn.Sequential(nn.Conv2d(in_chan, branch_chan, kernel_size=1, padding=0), nn.Mish())
        self.branch2 = nn.Sequential(nn.Conv2d(in_chan, branch_chan, kernel_size=1, padding=0), nn.Mish())
        self.branch3 = nn.Sequential(nn.Conv2d(in_chan, branch_chan, kernel_size=1, padding=0), nn.Mish())
       
        self.feature_fus1 = FeatureFusion(in_chan, 1, 1, 0)
        self.feature_fus2 = FeatureFusion(branch_chan+1, in_chan, 1, 0)

    def forward(self, x):

        step = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        step = torch.cat((self.feature_fus1(step), self.branch3(x)), dim=1)
        step = torch.add(self.feature_fus2(step), x) # orig imp has 2 * x
        
        return step

class MulResBranch(nn.Module):
    def __init__(self, branch_chan, nested_branch_chan, var_kernel_size, var_padding):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(branch_chan, nested_branch_chan, 1, padding=0), nn.Mish())
        self.conv2 = nn.Sequential(nn.Conv2d(nested_branch_chan, nested_branch_chan, var_kernel_size, padding=var_padding), nn.Mish())
        self.conv3 = nn.Conv2d(nested_branch_chan, nested_branch_chan, var_kernel_size, padding=var_padding) 

    def forward(self, x):
        
        hold = self.conv1(x)
        step = self.conv2(hold)
        step = self.conv3(step)
        step = torch.add(step, hold)

        return step

class MulResBlock(nn.Module):
    def __init__(self, branch_chan):
        super().__init__()
        
        nested_branch_chan = branch_chan//4

        self.branch1 = MulResBranch(branch_chan, nested_branch_chan, 1, 0)
        self.branch2 = MulResBranch(branch_chan, nested_branch_chan, 3, 1)
        self.branch3 = MulResBranch(branch_chan, nested_branch_chan, 5, 2)
        self.branch4 = MulResBranch(branch_chan, nested_branch_chan, 7, 3)

        self.ECBAM = ECBAM(branch_chan, reduction_ratio=4)
	
    def forward(self, x):
       	
        step = torch.cat((self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)), dim=1)
        step = self.ECBAM(step)
        step = step + x

        return step

class MulSquaredResBranch(nn.Module):
    def __init__(self, in_chan, branch_chan, var_kernel_size, var_padding):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_chan, branch_chan, 1, padding=0), nn.Mish())
        self.conv2 = nn.Sequential(nn.Conv2d(branch_chan, branch_chan, var_kernel_size, padding=var_padding), nn.Mish())
        self.mul_res = MulResBlock(branch_chan)
        self.conv3 = nn.Sequential(nn.Conv2d(branch_chan, in_chan, 1, padding=0), nn.Mish())

    def forward(self, x):

        step = self.conv1(x)
        hold = self.conv2(step)
        
        step = self.mul_res(hold)
        step = step + hold
        step = self.conv3(step)

        return step

class MulSquaredResBlock(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        
        branch_chan = in_chan//4

        self.branch1 = MulSquaredResBranch(in_chan, branch_chan, 1, 0)
        self.branch2 = MulSquaredResBranch(in_chan, branch_chan, 3, 1)
        self.branch3 = MulSquaredResBranch(in_chan, branch_chan, 5, 2)
        self.branch4 = MulSquaredResBranch(in_chan, branch_chan, 7, 3)	
	
        self.ECBAM = ECBAM(in_chan, reduction_ratio=4)
	
    def forward(self, x):
       	
        step = self.branch1(x) + self.branch2(x) + self.branch3(x) + self.branch4(x)
        step = self.ECBAM(step)
        step = step + x

        return step

