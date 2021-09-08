### Source code for CVEGAN

These scripts were developed and run with Pytorch 1.9.0, torchvision 0.10.0 and CUDA 11.1.

### Evaluation

To evaluate a trained model, run eval.py with the following arguments:

lr_folder = "folder with lr sequences to evaluate" <br />
save_folder = "folder to save processed frames to" <br />
hr_folder = "folder with hr sequences to compare to" <br />
codec_folder = "folder with codec sequences to benchmark" <br />
model_file = "model file to evaluate"

python eval.py --lr_folder="FOLDER1" --save_folder="FOLDER2" --hr_folder="FOLDER3" --codec_folder="FOLDER4" --model_file="MODELFILE"

All other arguments are optional.

Note: the evaluation code assumes 10 bit YUV from the decoder for both 10 bit and 8 bit input sequences. There are arguments to the frame reading / writing methods for overriding the automatic (filename based) bit rate and code normalising the 8 bit hr files to 10 bit values.

### Training

Code supports single gpu training (linux and windows) and multi-gpu training (linux only). 

To train CVEGAN, call train.py with the following arguments:

train_folder = "overall folder to save models and image of training"<br />
sub_name = "subfolder to save models and images of training"<br />
dataset_folder = "folder of dataset" (contains two folders (train / valid) each with two folders (hr / lr)<br />

Note that the folder used is train_folder/sub_name so please do not pass sub_name a full path.

All other arguments are optional.

Note that batch_size is defined per GPU and per accumulation step, an effective batch size is printed to stdout at the start of trining. However, if patches are saved "stacked" (with batch dimension > 1) batch size is scaled accordingly.

Initial (without discriminator) training does not pass directly into GAN (with discriminator) training. Please restart training with --start_epoch = "EPOCH OF MODEL FILE TO LOAD" and --gan = True, the relevant files will be loaded and training will resume from the following epoch.

### Reference

This code was originally developed by Dr Di Ma using Tensorflow, and further implemented by Mr Charlie Tan using Pytorch.
 
[1] Ma, D., Zhang, F. and Bull, D.R., 2020. CVEGAN: A Perceptually-inspired GAN for Compressed Video Enhancement. arXiv preprint arXiv:2011.09190.

### Author

- Dr Di Ma
- [char-tan](https://github.com/char-tan)
- [fan-aaron-zhang](https://github.com/an-aaron-zhang)
