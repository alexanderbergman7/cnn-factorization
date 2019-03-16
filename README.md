# cnn-factorization

This repository contains the code that was used to generate the results described in the EE367 final report, Factorized Convolution Kernels in Image Processing. This README describes the usage of the files, an a sample pipeline that can be run in order to use a model for denoising and profile a model.

## TO RUN THE EXAMPLE CODE: <br/>
-


## DESCRIPTION OF FILES: <br/>
### In the scripts/ directory: <br/>
#### addNoise.m: <br/>
generates a noisy dataset from the Berkeley Segmentation Dataset, where the BSD is in a directory on the same level as this repository. <br/>
#### BM3D_denoising.m: <br/>
denoises the noisy images generated into a directory in the dataset using the BM3D algorithm described at http://www.cs.tut.fi/~foi/GCF-BM3D/. This directory must be downloaded and in the MATLAB path so that the BM3D function can be used. <br/>
#### processOutput.m: <br/>
after having run the denoising process using one of the models, this will take the output images and calculate PSNR, SSIM, and MSE from the original grayscale images from the BSD.

### Python files: <br/>
#### CP_factorization.py: <br/>
inside of this file, a model is read (name has to be manually changed) and then a new model is outputted based on the CP decomposition of this model. This CP decomposition model is also evaluated on the testing dataset. The value for the rank of the CP decomposition is contained in constants.py <br/>
#### UNet2D.py: <br/>
Trains a U-Net model based on the hyperparameters described in constants.py, and then evaluates it on the testing data. <br/>
#### UNet2D_separable.py: <br/>
Trains a U-Net model implemented with depthwise separable convolutions based on the hyperparameters in constants.py and then evaluates it on the testing data. <br/>
#### constants.py: <br/>
Contains hyperparameters for training, image sizing, details about running on a GPU or CPU, and rank for CP decomposition. Stores constant parameters for all files in the directory. <br/>
#### profile.py: <br/>
Takes in a model as a command line argument, and then computes the number of operations and parameters that would be used for evaluating a batch input with dimensions described in constants.py <br/>
#### pytorch_prototyping.py: <br/>
Contains implementation for the U-Net model with standard convolutions. <br/>
#### separable_model_prototyping.py: <br/>
Contains implementation for the U-Net model implemented with depthwise separable convolutions. <br/>

### In the models/ directory: <br/>
Pretrained models for the U-Net, U-Net implemented with depthwise separable convolutions, CP decomposed U-Net, and U-Net trained on cropped images.
