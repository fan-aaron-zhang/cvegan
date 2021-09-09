### Source code for CVEGAN

CVEGAN was originally developed by Dr Di Ma using Tensorflow, this code is a Pytorch migration conducted by Mr Charlie Tan as part of his University of Bristol Faculty of Engineering Summer Research Internship [1]. Multi-gpu training and the batch based evaluation procedure are additional contributions of Mr Tan.

These scripts were implemented and run with Pytorch 1.9.0, torchvision 0.10.0 and CUDA 11.1. 

### Evaluation

To evaluate a trained model, run eval.py with the following arguments:

- lr_folder = folder with lr sequences to evaluate 
- save_folder = folder to save processed frames to
- hr_folder = folder with hr sequences to compare to
- codec_folder = folder with codec sequences to benchmark
- model_file = model file to evaluate

python eval.py --lr_folder="" --save_folder="" --hr_folder="" --codec_folder="" --model_file=""

All other arguments are optional.

Notes:
- the evaluation code assumes 10 bit YUV420 from the decoder for both 10 bit and 8 bit input sequences. There are arguments to the frame reading / writing methods for overriding the automatic (filename based) bit rate and code normalising the 8 bit hr files to 10 bit values.

### Training

Code supports single gpu training (linux and windows) and single node multi-gpu training (linux only). 

To train CVEGAN, call train.py with the following arguments:
- train_folder = overall folder to save models and image of training
- sub_name = subfolder to save models and images of training
- dataset_folder = folder of dataset (contains two folders (train / valid) each with two folders (hr / lr)

python train.py --train_folder="" --sub_name="" --dataset_folder=""

For multi-gpu replace "python" with "python -m torch.distributed.run --nproc_per_node=N --use_env", where N is the number of gpus to use.

All other arguments are optional.

Notes:
- the folder used is train_folder/sub_name (do not pass sub_name a full path).
- patches are assumed to be stored as tensors normalised to range [0, 1].
- batch size is defined per GPU and per accumulation step, an effective batch size is printed to stdout at the start of training. However, if patches are saved "stacked" (with batch dimension > 1) batch size is scaled accordingly.
- initial (without discriminator) training does not pass directly into GAN (with discriminator) training. Please restart training with --start_epoch=E and --gan=True, where E is the epoch from which to load the generator files. The relevant files will be loaded and training will resume from the following epoch.
- example batches provided are from the BVI-DVC database [2].

### References

Training code used the following repository as a starting point: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/srgan/srgan.py

cbam.py taken directly from official implmentation: https://github.com/Jongchan/attention-module [3]

[1] D. Ma, F. Zhang, and D. R. Bull, ‘CVEGAN: A Perceptually-inspired GAN for Compressed Video Enhancement’, arXiv:2011.09190 [cs, eess], Nov. 2020, Accessed: Jun. 24, 2021. [Online]. Available: http://arxiv.org/abs/2011.09190

[2] D. Ma, F. Zhang, and D. R. Bull, ‘BVI-DVC: A Training Database for Deep Video Compression’, arXiv:2003.13552 [cs, eess], Oct. 2020, Accessed: Jun. 24, 2021. [Online]. Available: http://arxiv.org/abs/2003

[3] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, ‘CBAM: Convolutional Block Attention Module’, 2018, pp. 3–19. Accessed: Sep. 08, 2021. [Online]. Available: https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html 

### Authors

- Dr Di Ma
- [char-tan](https://github.com/char-tan)
- [fan-aaron-zhang](https://github.com/an-aaron-zhang)
