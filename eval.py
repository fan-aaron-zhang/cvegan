import argparse
import time
import glob
import torch

from models import *
from utils import *

###################################################################################################
######## This code was developed by Charlie Tan, Student @ University of Bristol, UK, 2021 ########
###################################################################################################

"warning: code setup to override bitdepth lr and codec seuqences are read at to 10"

def process_frame(sequence, frame_index, generator, opt):

    ### LOAD FRAME ###
    frame_in = sequence.load_frame(frame_index).cuda()
    
    frame_in_shape = frame_in.shape
    
    ### PROCESS AS BATCHES ###
    if opt.split:

        ### PREPARE PATCHES ###
        fold_unfold = FoldUnfold(*frame_in.shape[-2:], opt.lr_patch_dim, opt.overlap)
        patches_in = fold_unfold.fold_to_patches(frame_in)

        num_patches = patches_in.shape[0]
        num_batches = num_patches // opt.batch_size + 1

        del frame_in # free GPU memory

        ### EMPTY FRAME TO FILL WITH PROCESSED BATCHES ###
        patches_gen = torch.empty_like(patches_in).tile(1,1,2,2)

        ### BATCH ITERATIONS ###
        for batch_index in range(num_batches): 

            patch_index = batch_index * opt.batch_size

            if batch_index != num_batches - 1:

                batch_in = patches_in[patch_index:patch_index+opt.batch_size, :, :, :]
                patches_gen[patch_index:patch_index+opt.batch_size,:, :, :] = generator(batch_in)

            else: # if last batch, take what's left

                batch_in = patches_in[patch_index:, :, :, :]
                patches_gen[patch_index:,:, :, :] = generator(batch_in)

        ### RECONSTRUCT FRAME ###
        frame_gen = fold_unfold.unfold_to_frame(patches_gen)

    ### PROCESS AS WHOLE FRAME ###
    else:
        frame_gen = generator(frame_in)

    return chroma_sub_420(frame_gen)

def process_sequence(filename, generator, opt):

    sequence_in = YUVFile(filename, bit_depth_read=10) # overridng bit depth for reading
    sequence_in.print_parameters()
    
    filename_gen = output_name(sequence_in, opt)
    sequence_gen = YUVFile(filename_gen, write=True, bit_depth_write=10) # overridng bit depth for write

    if opt.num_frames != -1:
        num_frames = opt.num_frames
    else:
        num_frames = sequence_in.num_frames
     
    mode = 'wb' # write / overwrite on first iteration of for loop

    ### FRAME ITERATIONS ###    
    for frame_index in range(num_frames):
        start_time = time.time()

        print(f"upsampling {filename} frame {frame_index + 1}")

        frame_gen = process_frame(sequence_in, frame_index, generator, opt).cpu()

        ### SAVE FRAME ###
        sequence_gen.write_frame(frame_gen[0], mode)

        print("[*] upsampling took: %4.4fs, LR size: %s /  generated HR size: %s" % (
            time.time() - start_time, (sequence_in.height, sequence_in.width), frame_gen.shape))
            
        mode = 'ab' # append on subquent iterations

    return filename_gen


def evaluate_sequence(video_file_gen, video_file_hr, video_file_codec, opt):
    "could of been incorporated into processing but was worried about memory"

    print("hr sequence", video_file_hr)
    print("generator sequence", video_file_gen)
    print("codec sequence", video_file_codec)

    write_log(["lr file", video_file_gen], opt.stats_file_gan)
    write_log(["hr file", video_file_hr], opt.stats_file_gan)
    write_log(["frame", "PSNR frame Y", "PSNR frame U", "PSNR frame V", "PSNR frame Total"], opt.stats_file_gan)

    if opt.benchmark_codec:
        write_log(["lr file", video_file_codec], opt.stats_file_codec)
        write_log(["hr file", video_file_hr], opt.stats_file_codec)
        write_log(["frame", "PSNR frame Y", "PSNR frame U", "PSNR frame V", "PSNR frame Total"], opt.stats_file_codec)

    start_time = time.time()

    sequence_gen = YUVFile(video_file_gen, bit_depth_read = 10) # overridng bit depth for reading
    sequence_hr = YUVFile(video_file_hr)

    if opt.benchmark_codec:
        sequence_codec = YUVFile(video_file_codec, bit_depth_read = 10) # overridng bit depth for reading
    
    ### INITIAL VALUES FOR ITERATIVE MEAN ###
    sequence_gen_ws_psnr = [] 
    sequence_codec_ws_psnr = [] 

    if opt.num_frames != -1:
        num_frames = opt.num_frames
    else:
        num_frames = sequence_gen.num_frames

    ### FRAME ITERATIONS ###    
    for frame_index in range(num_frames):

        ### LOAD FRAMES ###
        frame_gen = sequence_gen.load_frame(frame_index, normalise=False)
        frame_hr = sequence_hr.load_frame(frame_index, normalise=False)

        if sequence_hr.bit_depth_orig == 8: # normalise to overriden bit depth
            frame_hr = frame_hr * (1023 / 255)
            frame_hr = torch.round(frame_hr).int()

        ### CALCULATE WS-PSNR ###
        _, frame_gen_ws_psnr = ws_psnr(frame_gen, frame_hr, 10)
        frame_gen_ws_psnr.append(np.mean(frame_gen_ws_psnr))
        sequence_gen_ws_psnr.append(frame_gen_ws_psnr)

        print("frame_gen WS-PSNR [Y,U,V,Total] :", frame_gen_ws_psnr)
        write_log([frame_index, *frame_gen_ws_psnr], opt.stats_file_gan)

        if opt.benchmark_codec:

            ### LOAD CODEC FRAME ###
            frame_codec = sequence_codec.load_frame(frame_index, normalise=False)

            ### CALCULATE WS-PSNR ###
            _, frame_codec_ws_psnr = ws_psnr(frame_codec, frame_hr, 10)
            frame_codec_ws_psnr.append(np.mean(frame_codec_ws_psnr))
            sequence_codec_ws_psnr.append(frame_codec_ws_psnr)

            print("frame_codec WS-PSNR [Y,U,V,Total] :", frame_codec_ws_psnr)
            write_log([frame_index, *frame_codec_ws_psnr], opt.stats_file_codec)

        print("\n\n")

    sequence_gen_means = np.mean(sequence_gen_ws_psnr, axis=0)
    write_log(["full sequence", *sequence_gen_means], opt.stats_file_gan)

    print("sequence_gen WS-PSNR [Y,U,V,Total]:", sequence_gen_means)

    if opt.benchmark_codec == True:
        sequence_codec_means = np.mean(sequence_codec_ws_psnr, axis=0)
        write_log(["full sequence", *sequence_codec_means], opt.stats_file_codec)

        print("sequence_codec WS-PSNR [Y,U,V,Total]:", sequence_codec_means)
    else:
        sequence_codec_means = None

    return sequence_gen_means, sequence_codec_means

def main():
    ### ARGUMENT PARSING ###
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=bool, default=True, help="to split frames into patches or not")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--num_frames", type=int, default=-1, help="number of frames per sequence to process")
    parser.add_argument("--lr_patch_dim", type=int, default=48, help="height/width of hr square patch")
    parser.add_argument("--ratio", type=int, default=2, help="upsampling ratio (only tested for ratio = 2)")
    parser.add_argument("--overlap", type=int, default=4, help="overlap of patches at lr")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--lr_folder", type=str, default=None, help="folder with lr sequeneces to evaluate")
    parser.add_argument("--save_folder", type=str, default=None, help="folder to save processed frames to")
    parser.add_argument("--hr_folder", type=str, default=None, help="folder with hr sequeneces to compare to")
    parser.add_argument("--codec_folder", type=str, default=None, help="folder with codec sequences to benchmark")
    parser.add_argument("--metrics", type=bool, default=True, help="to compare to hr video")
    parser.add_argument("--benchmark_codec", type=bool, default=True, help="to compare the files in codec_folder with hr_folder")
    parser.add_argument("--model_file", type=str, default=None, help="model file to evaluate")
    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.save_folder, exist_ok=True)

    ### STAT FILE SETUP ###
    opt.stats_file_gan = opt.save_folder+"/stats_gan.csv"
    write_log((vars(opt)).items(), opt.stats_file_gan)
    
    if opt.benchmark_codec:
        opt.stats_file_codec = opt.codec_folder+"/stats_codec.csv"
        write_log((vars(opt)).items(), opt.stats_file_codec)

    torch.cuda.set_device(0)

    ### PERFORMANCE ###
    torch.backends.cudnn.benchmark = True 

    ### INITIALISE MODEL ###
    generator = Generator().cuda()

    ### LOAD MODEL FILE ###
    state_dict = torch.load(opt.model_file, map_location=torch.device('cuda'))
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    generator.load_state_dict(state_dict)
    print(opt.model_file+" loaded!")
        
    ### EVALUATION MODE ###
    generator.eval()

    video_list_in = sorted(glob.glob(opt.lr_folder+"/*.yuv"))
    video_list_hr = sorted(glob.glob(opt.hr_folder+"/*.yuv"))

    if len(video_list_in) != len(video_list_hr):
        print("lr and hr sequnece mismatch")
        return 1

    if opt.benchmark_codec:
        video_list_codec = sorted(glob.glob(opt.codec_folder+"/*.yuv"))
        if len(video_list_codec) != len(video_list_hr):
            print("codec and hr sequnece mismatch")
            return 1
    else:
        video_list_codec = [None] * len(video_list_lr)

    set_gen = []
    set_codec = []
        
    with torch.no_grad():
        ### VIDEO SEQUENCE ITERATIONS ###
        for i, video_file_in in enumerate(video_list_in):

            ### PROCESS SEQUENCE ###
            video_file_gen = process_sequence(video_file_in, generator, opt)

            ### COMPUTE METRICS ###
            if opt.metrics:
                sequence_gen_means, sequence_codec_means = evaluate_sequence(video_file_gen, video_list_hr[i], video_list_codec[i], opt)
                set_gen.append(sequence_gen_means)
                set_codec.append(sequence_codec_means)

        set_gen_means = np.mean(set_gen, axis=0)
        print(set_gen_means)
        write_log(["full test set", *set_gen_means], opt.stats_file_gan)

        if opt.benchmark_codec: 
            set_codec_means = np.mean(set_codec, axis=0)
            print(set_codec_means)
            write_log(["full test set", *set_codec_means], opt.stats_file_codec)
            
if __name__ == '__main__':
    main()
