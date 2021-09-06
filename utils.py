import math
import os
import csv
import pdb

import torch
import torch.nn.functional as F
import numpy as np

###################################################################################################
######## This code was developed by Charlie Tan, Student @ University of Bristol, UK, 2021 ########
###################################################################################################

"some functions adapted from original CVEGAN source code"

def read_frame_yuv(filename, height, width, frame_index, bit_depth):
    "reads frame from filename, returns un-normalised numpy array"

    with open(filename, "br") as stream:
        if bit_depth == 8:

            datatype = np.uint8
            stream.seek(int(frame_index*1.5*width*height))
            
            # U,V smaller dimensions than Y due to 4:2:0 subsampling
            Y = np.fromfile(stream, dtype=datatype, count=width*height).reshape((height, width))
            U = np.fromfile(stream, dtype=datatype, count=(width//2)*(height//2)).reshape((height//2, width//2))
            V = np.fromfile(stream, dtype=datatype, count=(width//2)*(height//2)).reshape((height//2, width//2))

        else:

            datatype = np.int16
            stream.seek(frame_index*3*width*height)

            # U,V smaller dimensions than Y due to 4:2:0 subsampling
            Y = np.fromfile(stream, dtype=datatype, count=width*height).reshape((height, width))
            U = np.fromfile(stream, dtype=datatype, count=(width//2)*(height//2)).reshape((height//2, width//2))
            V = np.fromfile(stream, dtype=datatype, count=(width//2)*(height//2)).reshape((height//2, width//2))

        # upsample U and V to same dimensions as Y
        U_up = np.kron(U, np.ones((2, 2), dtype=datatype))
        V_up = np.kron(V, np.ones((2, 2), dtype=datatype))
 
        yuv = np.stack((Y, U_up, V_up), axis=-1)

        # move to [C, H, W] 
        yuv = np.moveaxis(yuv, 2, 0)
  
        return yuv

def write_frame_yuv(filename, frame, bit_depth, mode):
    "writes YUV numpy frame (frame) to filename"
    "does not interpolate, indexes with step to subsample UV"
    "interpolate before input if desired"

    with open(filename, mode) as stream:
        # 4:2:0 subsampling
        subsample_x = 2
        subsample_y = 2

        height = frame.shape[1]
        width = frame.shape[2]

        print(height, width)

        frame_size = height*width

        # U,V smaller dimensions than Y due to subsampling
        frame_size_colour = int(frame_size/(subsample_x*subsample_y))

        # expects floating point values in range [0, 1]
        if bit_depth == 10:
            frame = np.around(frame*(1023))
            datatype = 'int16'
        else:
            frame = np.around(frame*(255))
            datatype = 'int8'

        # preps Y values
        frame_Y = frame[0,:,:].reshape(frame_size).astype(datatype)

        # takes subsampled UV values
        frame_U = frame[1,0:height:subsample_y,0:width:subsample_x].reshape(frame_size_colour).astype(datatype)
        frame_V = frame[2,0:height:subsample_y,0:width:subsample_x].reshape(frame_size_colour).astype(datatype)

        # writes/appends bytes to file
        frame_Y.tofile(stream) 
        frame_U.tofile(stream)
        frame_V.tofile(stream)
 
class YUVFile:
    "class to handle YUV files"
    def __init__(self, filename, write=False, bit_depth_read=None, bit_depth_write=None):

        self.write = write
        self.filename = filename

        splitstr = self.filename.split('_')

        self.seq_name = splitstr[0]
        resolution = splitstr[1]
        resolution = resolution.split('x')
        self.width = int(resolution[0])
        self.height = int(resolution[1])
        self.frame_rate = splitstr[2]
        self.frame_rate = int(self.frame_rate[0:-3])

        self.bit_depth_orig = splitstr[3]
        self.bit_depth_orig = int(self.bit_depth_orig[0:-3])

        if bit_depth_read != None: # loading bit depth overriden
            self.bit_depth_read = bit_depth_read
        else:
            self.bit_depth_read = self.bit_depth_orig

        if bit_depth_write != None: # saving bit depth overriden
            self.bit_depth_write = bit_depth_write
        else:
            self.bit_depth_write = self.bit_depth_orig

        self.max_value_load = pow(2, self.bit_depth_read) - 1

        colorsamplingstr = splitstr[4]
        colorsamplingstr = colorsamplingstr.split('.')
        self.color_sampling = int(colorsamplingstr[0])

        try:
            statinfo = os.stat(self.filename)
            self.num_frames = int(statinfo.st_size / math.ceil(self.bit_depth_read / 8) / self.height / self.width / 1.5)  # for 420 only
        except FileNotFoundError:
            self.num_frames = 0

    def print_parameters(self):

        print('file name: ' + self.filename)
        print('sequence name: ' + self.seq_name)
        print('write approved: ' + str(self.write))
        print('width: ' + str(self.width) + ' height: ' + str(self.height))
        print('orignal bitdepth: ' + str(self.bit_depth_orig))
        print('color sampling: ' + str(self.color_sampling))
        print('frame rate: ' + str(self.frame_rate))

    def load_frame(self, index, normalise=True):
        "load a frame from the file, returns it as tensor"

        frame_np = read_frame_yuv(self.filename, self.height, self.width, index, self.bit_depth_read)
        frame_torch = torch.from_numpy(frame_np)

        if normalise:
            frame_torch = frame_torch / self.max_value_load

        frame_torch = frame_torch.unsqueeze(0)

        return frame_torch

    def write_frame(self, tensor_frame, mode):
        "write a frame (tensor) to the file"

        if self.write:
            write_frame_yuv(self.filename, tensor_frame.numpy(), self.bit_depth_write, mode)
        else:
            print("file not approved for writing, init VideoFile with write = True to write")

def write_log(row_list, filename, mode='a'):
    with open(filename, mode) as file:
        writer = csv.writer(file)
        writer.writerow(row_list)

def chroma_sub_420(pixels):
    "pixels = tensor of pixels in [B, C, H, W]"
    "assumes channels = 3 and first channel is Y"

    pixels_Y = pixels[:,0:1,:,:]
    pixels_UV = pixels[:,1:,:,:]

    pixels_UV_down = F.interpolate(pixels_UV, scale_factor=0.5, mode="bilinear", align_corners=True)
    pixels_UV_sub = F.interpolate(pixels_UV_down, scale_factor=2)

    pixels_sub = torch.cat((pixels_Y, pixels_UV_sub), dim = 1)

    return pixels_sub

def iterative_mean(old_mean, new_value, iteration):
    "avoids storing lots of values of metrics"
    new_mean = (iteration-1)/iteration * old_mean + new_value/iteration

    return new_mean

class FoldUnfold():
    "class to handle folding tensor frame into batch of patches and back to frame again"

    "https://stackoverflow.com/questions/45828265/is-there-a-function-to-extract-image-patches-in-pytorch"
    "https://stackoverflow.com/questions/62995726/pytorch-sliding-window-with-unfold-fold"
    "https://pytorch.org/docs/stable/generated/torch.nn.Fold.html#torch.nn.Fold"

    def __init__(self, height, width, kernel, overlap, ratio=2):

        if height % 2 or width % 2 or kernel % 2 or overlap % 2:
            print("only defined for even values of height, width, kernel size and overlap, odd values will reconstruct incorrectly")
            return 1

        self.height = height
        self.width = width

        self.kernel = kernel
        self.overlap = overlap
        self.stride = kernel - overlap

        self.ratio = ratio

    def fold_to_patches(self, frame):
        "idea is to extract overlapping patches of size = kernel"
        "pads the frame with zeros so fold does not crop"
        "this also means the corners of the final frame were not at the fringes of the patches"

        # number of blocks in each direction
        n_blocks_h = (self.height // (self.stride)) + 1 
        n_blocks_w = (self.width // (self.stride)) + 1 
    
        # how much to pad each edge by 
        self.pad_h = (self.stride * n_blocks_h + self.overlap - self.height) // 2
        self.pad_w = (self.stride * n_blocks_w + self.overlap - self.width) // 2
    
        # pad the frame so kernels (with overlap) fit perfectly (unfold would crop after the last full kernel), pad by relfection so boundary pixels of frame do not experience fringe effects
        frame_pad = F.pad(frame, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode='reflect')
    
        self.height_pad, self.width_pad = frame_pad.shape[2:]
    
        # unfold into tensor of [frames, chans * kernel ** 2, num_patches]
        frame_unfold = F.unfold(frame_pad, self.kernel, stride=self.stride)
    
        # permute and reshape into [num_patches, chans, kernel, kernel]
        patches = frame_unfold.permute(2, 1, 0).reshape(-1, 3, self.kernel, self.kernel)
    
        return patches
    
    def unfold_to_frame(self, patches):

        # patches being unfolded have been upsampled
        if self.ratio == 2:
            self.kernel *= 2
            self.height_pad *= 2
            self.width_pad *= 2
            self.stride *= 2
            self.pad_h *= 2
            self.pad_w *= 2

        # reshape and permute back into [frames, chans * kernel ** 2, num_patches] as expected by fold
        frame_unfold = patches.reshape(-1, 3 * self.kernel ** 2, 1).permute(2, 1, 0)
    
        # fold into tensor of shape pad_shape
        frame_fold = F.fold(frame_unfold, (self.height_pad, self.width_pad), self.kernel, stride=self.stride)
    
        # unfold sums overlaps instead of averaging so tensor of ones unfolded and
        # folded to track overlaps and take mean of overlapping pixels
        ones = torch.ones_like(frame_fold)
        ones_unfold = F.unfold(ones, self.kernel, stride=self.stride)
    
        # divisor is tensor of shape pad_shape where each element is the number of values that have overlapped
        # 1 = no overlaps
        divisor = F.fold(ones_unfold, (self.height_pad, self.width_pad), self.kernel, stride=self.stride)
    
        # divide reconstructed frame by divisor
        frame_div = frame_fold / divisor
    
        # crop frame to remove the padded areas
        frame_crop = frame_div[:,:,self.pad_h:-self.pad_h,self.pad_w:-self.pad_w].clone()
    
        return frame_crop

def yuv2rgb(pixels_yuv, bit_depth):
    "expects input in range [0, 1]"

    if len(pixels_yuv.shape) != 4:
        pixels_yuv = pixels_yuv.unsqueeze(0)
        batch = False
    else:
        batch = True

    Y = pixels_yuv[:,0,:,:] 
    U = pixels_yuv[:,1,:,:] 
    V = pixels_yuv[:,2,:,:] 

    fY = Y
    fU = U - 0.5
    fV = V - 0.5

    KR, KG, KB = 0.2627, 0.6780, 0.0593

    R = fY + 1.4746 * fV
    B = fY + 1.8814 * fU
    G = -(B*KB+KR*R-Y)/KG
  
    pixels_rgb = torch.stack((R, G, B), dim=1)

    if not batch:
        pixels_rgb = pixels_rgb[0,:,:,:]

    return pixels_rgb

# adapted from https://github.com/V-Sense/360SR/blob/master/ws_psnr.py
def ws_psnr(frame1, frame2, bit_depth):
    "weighted-spherical PSNR"

    max_val = 2 ** bit_depth - 1

    H, W = frame1.shape[-2:]

    height_grid = torch.from_numpy(np.mgrid[0:H, 0:W][0]).unsqueeze(0)

    weight_map = torch.cos((height_grid - (H/2) + 0.5 ) * math.pi/H) 

    ws_mse = []
    ws_psnr = []

    for channel in range(3):
        frame1_channel = frame1[0][channel]
        frame2_channel = frame2[0][channel]

        ws_mse.append(torch.sum(torch.multiply((frame1_channel-frame2_channel) ** 2, weight_map)) / torch.sum(weight_map))

        try:
            ws_psnr.append(20 * math.log10(max_val / math.sqrt(ws_mse[channel])))
        except ZeroDivisionError:
            ws_psnr.append(np.inf)

    return ws_mse, ws_psnr


def output_name(input_sequence, opt):
    "replaces the folder path and dimensions in a filename, creating a new filename"

    old_height = input_sequence.height 
    old_width = input_sequence.width
    old_location = opt.lr_folder
    
    new_height = old_height * opt.ratio
    new_width = old_width * opt.ratio
    new_location = opt.save_folder

    old_name = input_sequence.filename

    replacements = [[str(old_width), str(new_width)], [str(old_height), str(new_height)], [old_location, new_location]]

    new_name = old_name
    for replacement in replacements:
        new_name = new_name.replace(*replacement)

    return new_name

def load_checkpoint(model, opt, optimizer=None, scheduler=None):
    "loads training checkpoint for further training / evaluation"

    name = model.name
    initial = name[0].upper()

    model_file = opt.save_path+f"saved_models/{name}_{opt.start_epoch}.pth"

    state_dict = (torch.load(model_file, map_location=torch.device(f'cuda:{opt.local_rank}')))

    if not opt.ddp:
        ### ALLOWS LOADING OF DDP MODEL TO SINGLE GPU ###
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")

    model.load_state_dict(state_dict)

    print(f"{model_file} loaded!")

    if optimizer != None:

        optimizer_file = opt.save_path+f"optimizers/optimizer_{initial}_{opt.start_epoch}.pth"

        optimizer.load_state_dict(torch.load(optimizer_file, map_location=torch.device(f'cuda:{opt.local_rank}')))
        print(f"{optimizer_file} loaded!")

        if scheduler != None:

            scheduler_file = opt.save_path+f"schedulers/scheduler_{initial}_{opt.start_epoch}.pth"
            scheduler.load_state_dict(torch.load(scheduler_file, map_location=torch.device(f'cuda:{opt.local_rank}')))
            print(f"{scheduler_file} loaded!")

    return model, optimizer, scheduler


