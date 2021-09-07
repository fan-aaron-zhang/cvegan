## Source code for CVEGAN

These scripts were developped and run with Pytorch 1.9.0, torchvision 0.10.0 and CUDA 11.1 

### Evaluation

To evaluate a trained model run eval.py with the following arguments:

lr_folder = folder with lr sequences to evaluate <br />
save_folder = folder to save processed frames to <br />
hr_folder = folder with hr sequences to compare to (can be omitted if "metrics" = False) <br />
codec_folder = folder with codec sequences to benchmark (can be omitted if either "metrics" or "benchmark_codec" = False <br />
model_file = model file to evaluate

All other arguments are optional.

Note: the evaluation code assumes 10 bit YUV from the decoder for both 10 bit and 8 bit input sequences. There are arguments to the frame reading / writing methods for overriding the automatic (filename based) bit rate and code normalising the 8 bit hr files to 10 bit values.

### Training

### Reference

This code was originally developed by Dr Di Ma using Tensorflow, and further implemented by Mr Charlie Tan using Pytorch.
 
[1] Ma, D., Zhang, F. and Bull, D.R., 2020. CVEGAN: A Perceptually-inspired GAN for Compressed Video Enhancement. arXiv preprint arXiv:2011.09190.

### Author

- Dr Di Ma
- [char-tan](https://github.com/char-tan)
- [fan-aaron-zhang](https://github.com/an-aaron-zhang)
