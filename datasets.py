import glob

import torch
from torch.utils.data import Dataset

###################################################################################################
######## This code was developed by Charlie Tan, Student @ University of Bristol, UK, 2021 ########
###################################################################################################

"code developed from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/srgan/srgan.py"

class PatchDataset(Dataset):
    def __init__(self, root, subset, opt):

        self.files_hr = sorted(glob.glob(root + "/hr/*.*"))
        self.files_lr = sorted(glob.glob(root + "/lr/*.*"))

        if len(self.files_hr) != len (self.files_lr):
            print('patch mismatch!')
            return 1

        self.stack_size = torch.load(self.files_hr[0]).shape[0]

        print('{} dataset contains {} pairs of HR/LR patch blocks, {} patches per block'.format(subset, len(self.files_hr), self.stack_size))

    def __getitem__(self, index):

        "expects loaded tensors to already be floats in range [0,1] of shape [STACK, 3 x H x W]"
        patch_hr = torch.load(self.files_hr[index % len(self.files_hr)]).float()
        patch_lr = torch.load(self.files_lr[index % len(self.files_lr)]).float()

        return {"lr": patch_lr, "hr": patch_hr}

    def __len__(self):
        return len(self.files_hr)
