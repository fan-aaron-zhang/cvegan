import argparse
import os
import time
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.distributed.optim import ZeroRedundancyOptimizer
except ImportError: # torch.distributed.optim throwing ImportError on windows"
    pass

import numpy as np

import loss
from models import *
from datasets import *
from utils import *

###################################################################################################
######## This code was developed by Charlie Tan, Student @ University of Bristol, UK, 2021 ########
###################################################################################################

"code developed from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/srgan/srgan.py"

def train_epoch(epoch, opt, train_loader, loss_func_dict, loss_val_dict, generator, optimizer_G, discrim = None, optimizer_D = None):
  
    ### DEFAULT VALUES FOR AVG ###
    loss_val_dict["gen_tot_T"] = []
    loss_val_dict["gen_PLF_T"] = []
    loss_val_dict["gen_GAN_T"] = []
    loss_val_dict["disc_T"] = []

    # zero gradients before training
    optimizer_G.zero_grad(set_to_none=True)
    if opt.gan:
        optimizer_D.zero_grad(set_to_none=True)

    ### TRAINING ITERATIONS ###
    batch_start_time = time.time()
    for i, batch in enumerate(train_loader):

        ### INPUT ###
        batch_hr = batch["hr"].to(opt.local_rank, non_blocking=True).reshape(-1,3,96,96)
        batch_lr = batch["lr"].to(opt.local_rank, non_blocking=True).reshape(-1,3,48,48)

        ### SUBSAMPLED GENERATOR OUTPUT ###
        gen_hr420 = chroma_sub_420(generator(batch_lr))
        
        ### TRAIN DISCRIMINATOR ###
        if opt.gan: 
            ### DISCRIMINATOR OUTPUTS ###
            disc_real = discrim(batch_hr)
            disc_fake = discrim(gen_hr420.detach()) # not attached to gen graph

            ### DISCRIMINATOR LOSS ###
            loss_D = loss_func_dict["discGAN"](disc_real, disc_fake)

            loss_D = loss_D / opt.accumulation_steps # scale for gradient accumulation

            ### BACKPROP D ###
            if (i + 1) % opt.accumulation_steps != 0: 
                if opt.ddp:
                    with discrim.no_sync(): # not all_reduced across GPUs, no step()
                        loss_D.backward()
                else:
                    loss_D.backward()
            else:
                loss_D.backward() # all_reduced across GPUs
                optimizer_D.step()
                optimizer_D.zero_grad(set_to_none=True)

            ### NEW DISCRIMINATOR OUTPUTS ###
            disc_real = discrim(batch_hr).detach()
            disc_fake = discrim(gen_hr420) # attached to gen graph

            ### GENERATOR GAN LOSS ###
            loss_gen_GAN = loss_func_dict["genGAN"](disc_real, disc_fake)

        ### PLF ###
        loss_PLF, *_ = loss_func_dict["PLF"](batch_hr, gen_hr420)

        ### TOTAL GENERATOR LOSS ###
        if opt.gan:
            loss_G = loss_PLF + 5e-3 * loss_gen_GAN
        else:
            loss_G = loss_PLF

        loss_G = loss_G / opt.accumulation_steps # scale for gradient accumulation

        ### BACKPROP G ###
        if (i + 1) % opt.accumulation_steps != 0: 
            if opt.ddp:
                with generator.no_sync(): # not all_reduced across GPUs, no step()
                    loss_G.backward() 
            else:
                loss_G.backward()
        else:
            loss_G.backward() # all_reduced across GPUs
            optimizer_G.step()
            optimizer_G.zero_grad(set_to_none=True)

        ### SCALE BACK UP VALUES FOR PRINT / LOGGING ###
        loss_G_logging = loss_G.detach() * opt.accumulation_steps
        if opt.gan:
            loss_D_logging = loss_D.detach() * opt.accumulation_steps

        ### ROLLING AVGS ###
        loss_val_dict["gen_tot_T"].append(loss_G_logging.item())
        loss_val_dict["gen_PLF_T"].append(loss_PLF.item())

        if opt.gan:
            loss_val_dict["gen_GAN_T"].append(loss_gen_GAN.item())
            loss_val_dict["disc_T"].append(loss_D_logging.item())

        ### EMPTY PRINT FIELDS ###
        if not opt.gan:
            loss_gen_GAN = None
            loss_D_logging = None

        # scale up loss_G and loss_D for consistency
        print('[GPU {}] [Epoch {}] [Batch {}.{}] [Gen Total = {}] [Gen PLF = {}] [Gen GAN = {}] [Discrim = {}] [Time = {}]'
                .format(opt.local_rank, epoch, i//opt.accumulation_steps, i % opt.accumulation_steps, loss_G_logging, loss_PLF, 
                    loss_gen_GAN, loss_D_logging, time.time()-batch_start_time))

        batch_start_time = time.time()

        if opt.overfit:
            break

    loss_val_dict["gen_tot_T"] = np.mean(loss_val_dict["gen_tot_T"])
    loss_val_dict["gen_PLF_T"] = np.mean(loss_val_dict["gen_PLF_T"])

    if opt.gan:
        loss_val_dict["gen_GAN_T"] = np.mean(loss_val_dict["gen_GAN_T"])
        loss_val_dict["disc_T"] = np.mean(loss_val_dict["disc_T"])
    else:
        loss_val_dict["gen_GAN_T"] = 0
        loss_val_dict["disc_T"] = 0


def valid_epoch(epoch, opt, valid_loader, loss_func_dict, loss_val_dict, generator):
 
    ### DEFAULT VALUES FOR AVG ###
    loss_val_dict["gen_PLF_V"] = []
    loss_val_dict["gen_L1_V"] = []
    loss_val_dict["gen_MSE_V"] = []
    loss_val_dict["gen_SSIM_V"] = []
    loss_val_dict["gen_MS_SSIM_V"] = []

    ### NO GRAD FOR VALIDATION ###
    with torch.no_grad():

        ### VALIDATION ITERATIONS ###
        for i, batch in enumerate(valid_loader):
            batch_start_time = time.time()

            ### INPUT ###
            batch_hr = batch["hr"].to(opt.local_rank, non_blocking=True).reshape(-1,3,96,96)
            batch_lr = batch["lr"].to(opt.local_rank, non_blocking=True).reshape(-1,3,48,48)

            ### SUBSAMPLED GENERATOR OUTPUT ###
            gen_hr420 = chroma_sub_420(generator(batch_lr))

            ### VALIDATION LOSSES ###
            loss_PLF, loss_L1, loss_MSE, loss_SSIM, loss_MS_SSIM = loss_func_dict["PLF"](batch_hr, gen_hr420)

            loss_val_dict["gen_PLF_V"].append(loss_PLF.item())
            loss_val_dict["gen_L1_V"].append(loss_L1.item())
            loss_val_dict["gen_MSE_V"].append(loss_MSE.item())
            loss_val_dict["gen_SSIM_V"].append(loss_SSIM.item())
            loss_val_dict["gen_MS_SSIM_V"].append(loss_MS_SSIM.item())
                
            print('[GPU {}] [Epoch {}] [Validation Batch {}] [PLF = {}] [L1 = {}] [MSE = {}] [SSIM = {}] [MS_SSIM = {}] [Time = {}]'.format(opt.local_rank, epoch, i, loss_PLF, loss_L1, loss_MSE, loss_SSIM, loss_MS_SSIM, time.time()-batch_start_time))

            if opt.overfit:
                break

        loss_val_dict["gen_PLF_V"] = np.mean(loss_val_dict["gen_PLF_V"])
        loss_val_dict["gen_L1_V"] = np.mean(loss_val_dict["gen_L1_V"])
        loss_val_dict["gen_MSE_V"] = np.mean(loss_val_dict["gen_MSE_V"])
        loss_val_dict["gen_SSIM_V"] = np.mean(loss_val_dict["gen_SSIM_V"])
        loss_val_dict["gen_MS_SSIM_V"] = np.mean(loss_val_dict["gen_MS_SSIM_V"])
            
        ### IMAGE SAMPLES ###
        if opt.local_rank == 0: # only master process

            batch_lr_up = F.interpolate(batch_lr, scale_factor=2)

            batch_lr_up = make_grid(batch_lr_up[0:3,:,:,:], nrow=1, normalize=False)
            gen_hr420 = make_grid(gen_hr420[0:3,:,:,:], nrow=1, normalize=False)
            batch_hr = make_grid(batch_hr[0:3,:,:,:], nrow=1, normalize=False)

            grid = torch.cat((batch_lr_up, gen_hr420, batch_hr), 2)
            grid_rgb = yuv2rgb(grid, 10)

            save_image(grid_rgb, opt.save_path+"images/sample_{}.png".format(epoch), normalize=False)

def main():

    ### ARGUMENT PARSING ###
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_epoch", type=int, default=0, help="epoch to start/resume training from, if 0 start from scratch, if > 0 load files and resume from start_epoch+1")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--overfit", type=bool, default=False, help="to overfit a single batch (useful for debug without waiting for epochs)")
    parser.add_argument("--gan", type=bool, default = False, help="discriminator/GAN training, does not automatically switch, restart training with gan after init training")
    parser.add_argument("--load_discrim", type=bool, default = False, help="to load pretrained discriminator files at the start of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the micro batches [per GPU]")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="number of micro batches to accumulate before stepping optimizer")
    parser.add_argument("--lr_G", type=float, default=0.0001, help="learning rate for generator") 
    parser.add_argument("--lr_D", type=float, default=0.0001, help="learning rate for discriminator")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient") 
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--num_workers", type=int, default=4, help="number of cpu threads to use during batch generation [per GPU]")
    parser.add_argument("--train_folder", type=str, default=None, help="overall folder for models and images of sub training")
    parser.add_argument("--sub_name", type=str, default=None, help="subfolder to save models and images of sub training")
    parser.add_argument("--dataset_folder", type=str, default=None, help="folder of dataset")
    parser.add_argument("--seed", type=int, default=689, help = "provide a seed value to use to compare trainings")
    parser.add_argument("--zero", type=bool, default=True, help = "use ZeroRedundancyOptimizer")
    opt = parser.parse_args()
    print(opt)

    opt.n_gpus = torch.cuda.device_count()

    if opt.n_gpus > 1:
        opt.ddp = True
        opt.local_rank = os.environ["LOCAL_RANK"]
    else:
        opt.ddp = False
        opt.local_rank = 0
    
    opt.local_rank = int(opt.local_rank)

    ### REUSE SEED ###
    # this does not ensure reproducibility (dataloading is still non deterministic) but should help reduce difference due to initalisation 
    torch.manual_seed(opt.seed)
    
    ### FOLDER / PATH MANAGEMENT ###
    opt.save_path = opt.train_folder+"/"+opt.sub_name+"/"
    if opt.local_rank == 0: # master process only

        for sub_dir in ["images", "saved_models", "optimizers", "schedulers"]:
            os.makedirs(opt.save_path+sub_dir, exist_ok=True)

        print("Training with {} GPUs".format(opt.n_gpus))
        print(f"Each GPU process batch of {opt.batch_size} per iteration")
        print(f"Accumulate {opt.accumulation_steps} iterations per optimizer step")
        print(f"Effective batch size of {opt.n_gpus*opt.batch_size*opt.accumulation_steps}")

        ### LOG FILE SETUP ###
        opt.log_file = opt.save_path+"log.csv"
        write_log((vars(opt)).items(), opt.log_file)
        write_log(["Epoch", "Gan", "LR", "Gen Total Train", "Gen PLF Train", "Gen GAN Train", "Disc Train",
                "Gen PLF Valid", "Gen L1 Valid", "Gen MSE Valid", "Gen SSIM Valid", "Gen MS_SSIM Valid", "Time Taken"], opt.log_file)

    ### INITIALISE PROCESS GROUP ###
    if opt.ddp:
        dist.init_process_group(backend='nccl', world_size=opt.n_gpus, rank=opt.local_rank)
        print("GPU {} init process group".format(opt.local_rank))
    
    torch.cuda.set_device(opt.local_rank)

    ### PERFORMANCE ###
    torch.backends.cudnn.benchmark = True 

    ### INITIALISE MODELS + WRAP AS DDP ###
    generator = Generator().to(opt.local_rank)
    if opt.ddp:
        generator = nn.SyncBatchNorm.convert_sync_batchnorm(generator) # sync BN across GPUs + prevent buffer error
        generator = DDP(generator, device_ids=[opt.local_rank])
    if opt.gan:
        discrim = Discriminator(input_shape=(opt.channels, opt.hr_patch_dim, opt.hr_patch_dim)).to(opt.local_rank) # remove input shape?
        if opt.ddp:
            discrim = nn.SyncBatchNorm.convert_sync_batchnorm(discrim) # sync BN across GPUs + prevent buffer error
            discrim = DDP(discrim, device_ids=[opt.local_rank])
    else:
        discrim = None

    ### LOSSES ###
    loss_func_dict = {
            "PLF" : loss.PLF().to(opt.local_rank),
            "genGAN" : loss.GenReSphere(opt.batch_size, opt.local_rank).to(opt.local_rank),
            "discGAN" : loss.DiscReSphere(opt.batch_size, opt.local_rank).to(opt.local_rank),
            }

    ### OPTIMISERS ###
    if opt.ddp and opt.zero:
        optimizer_G = ZeroRedundancyOptimizer(generator.parameters(), optimizer_class=torch.optim.AdamW, lr=opt.lr_G, betas=(opt.b1, opt.b2))
        if opt.gan:
            optimizer_D = ZeroRedundancyOptimizer(discrim.parameters(), optimizer_class=torch.optim.Adam, lr=opt.lr_D, betas=(opt.b1, opt.b2))
    else:
        optimizer_G = torch.optim.AdamW(generator.parameters(), lr=opt.lr_G, betas=(opt.b1, opt.b2))
        if opt.gan:
            optimizer_D = torch.optim.Adam(discrim.parameters(), lr=opt.lr_D, betas=(opt.b1, opt.b2))

    if not opt.gan:
        optimizer_D = None 

    scheduler_G = None
    scheduler_D = None

    ### LOAD TRAINING CHECKPOINT ###
    if opt.start_epoch != 0:

        generator, optimizer_G, scheduler_G = load_checkpoint(generator, opt, optimizer_G, scheduler_G)

        if opt.gan and opt.load_discrim:
            discrim, optimizer_D, scheduler_D = load_checkpoint(discrim, opt, optimizer_D, scheduler_D)

        opt.start_epoch += 1 # start from epoch following loaded model

    ### DATASETS ###
    train_dataset = PatchDataset(opt.dataset_folder+"/train", "train", opt)
    valid_dataset = PatchDataset(opt.dataset_folder+"/valid", "valid", opt)

    if not opt.batch_size % train_dataset.stack_size:
        opt.batch_size //= train_dataset.stack_size
    else:
        print("invalid batch size for given dataset (patch stacks cannot be evenly placed into batch size}")
        return 1

    ### DATASAMPLERS ###
    if opt.ddp:
        mp.set_start_method('spawn') # for nccl compatability
        train_sampler = DistributedSampler(train_dataset, drop_last=True)
        valid_sampler = DistributedSampler(valid_dataset, drop_last=True)
        shuffle_on_load = False
    else:
        train_sampler = None
        valid_sampler = None

        if not opt.overfit:
            shuffle_on_load = True
        else:
            shuffle_on_load = False

    ### DATALOADERS ###    
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.batch_size,
        shuffle=shuffle_on_load,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True, # forces consistent batch dim
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True, # forces consistent batch dim
     )

    print("GPU {} train loader len {}".format(opt.local_rank, len(train_loader)))
    print("GPU {} valid loader len {}".format(opt.local_rank, len(valid_loader)))

    ### EPOCH ITERATIONS ###
    for epoch in range(opt.start_epoch, opt.n_epochs):
        epoch_start_time = time.time()

        ### TRAINING MODE ###
        generator.train()
        if opt.gan:
            discrim.train()

        if opt.ddp: 
            train_sampler.set_epoch(epoch) # ensures batches are shuffled when using ddp
            valid_sampler.set_epoch(epoch) # ensures batches are shuffled when using ddp
            torch.distributed.barrier() # syncs processes before epoch starts

        loss_val_dict = {}

        ### TRAINING EPOCH ###
        train_epoch(epoch, opt, train_loader, loss_func_dict, loss_val_dict, generator, optimizer_G, discrim, optimizer_D)

        ### EVALUATION MODE ###
        generator.eval()
        if opt.gan:
            discrim.eval()

        ### VALIDATION EPOCH ###
        valid_epoch(epoch, opt, valid_loader, loss_func_dict, loss_val_dict, generator)

        loss_list = []
        for loss_val in ["gen_tot_T", "gen_PLF_T", "gen_GAN_T", "disc_T", "gen_PLF_V", "gen_L1_V", "gen_MSE_V", "gen_SSIM_V", "gen_MS_SSIM_V"]:
            loss_list.append(loss_val_dict[loss_val])

        loss_tensor = torch.tensor(loss_list)
        
        ### REDUCE DISTIRBUTED VALUES ###
        if opt.ddp:
            torch.distributed.barrier() # syncs processes before loss is reduced
            torch.distributed.reduce(loss_tensor, 0)
        
            ### CONSOLIDATE OPTIMIZER STATE DICTS ### 
            optimizer_G.consolidate_state_dict()
            if opt.gan:
                optimizer_D.consolidate_state_dict()

        ### LOGGING ###
        if opt.local_rank == 0: # only master process 

            loss_list = (loss_tensor/opt.n_gpus).tolist()

            gen_tot_T = loss_list[0]
            gen_PLF_T = loss_list[1]
            gen_PLF_V = loss_list[4]
            gen_L1_V = loss_list[5]
            gen_MSE_V = loss_list[6]
            gen_SSIM_V = loss_list[7]
            gen_MS_SSIM_V = loss_list[8]

            if opt.gan:
                gen_GAN_T = loss_list[2]
                disc_T = loss_list[3]
            else:
                gen_GAN_T = None
                disc_T = None
            
            epoch_time = time.time()-epoch_start_time

            ### PRINT TO OUTPUT FILE ###
            print('[Epoch {}] [Gen Total Train = {}] [Gen PLF Train = {}] [Gen GAN Train = {}] [Discrim Train = {}]\n\
                    [Gen PLF Valid = {}] [Gen L1 Valid = {}] [Gen Loss Valid = {}] [Gen SSIM Valid {}] [Gen MS_SSIM Valid {}] [Time = {}]'.format(epoch, gen_tot_T,
                    gen_PLF_T, gen_GAN_T, disc_T, gen_PLF_V, gen_L1_V, gen_MSE_V, gen_SSIM_V, gen_MS_SSIM_V, epoch_time))

            ### WRITE TO LOG FILE ###
            write_log([epoch, str(opt.gan), opt.lr_G, gen_tot_T, gen_PLF_T, gen_GAN_T, disc_T, gen_PLF_V, 
                gen_L1_V, gen_MSE_V, gen_SSIM_V, gen_MS_SSIM_V, epoch_time], opt.log_file)

            ### SAVE CHECKPOINT ### 
            torch.save(generator.state_dict(), opt.save_path+"saved_models/generator_{}.pth".format(epoch))
            torch.save(optimizer_G.state_dict(), opt.save_path+"optimizers/optimizer_G_{}.pth".format(epoch))
            if opt.gan:
                torch.save(discrim.state_dict(), opt.save_path+"saved_models/discrim_{}.pth".format(epoch))
                torch.save(optimizer_D.state_dict(), opt.save_path+"optimizers/optimizer_D_{}.pth".format(epoch))

    if opt.ddp:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
