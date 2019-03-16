import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils
import torch.optim as optim
import torchvision

import numpy as np

import math
import numbers
import random

# UNet models
import separable_model_prototyping

from skimage import io
import os

# hacky method since running in virtual environment on my machine or X11 forwarding to view images through SSH
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#import psutil
import scipy.io as sio

import tensorly.decomposition as d

import constants

USE_GPU = constants.USE_GPU
USE_CROP = constants.USE_CROP
CROP_SIZE = constants.CROP_SIZE
IMAGE_SIZE = constants.IMAGE_SIZE

if USE_GPU:
	print(torch.cuda.is_available())
	torch.cuda.set_device(0)

# class used for interfacing the dataset
class Dataset():

	# initialize the dataset based on the directory that images will be contained in
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.train = os.listdir(os.path.join(root_dir, 'train_gray'))
		self.val = os.listdir(os.path.join(root_dir, 'val_gray'))

	# get a training pair in tensor form
	def getTrainItem(self, idx):
		image_original = torch.from_numpy(io.imread(os.path.join(self.root_dir, 'train_gray', self.train[idx])))
		image_noisy = torch.from_numpy(io.imread(os.path.join(self.root_dir, 'noisy_train_sigma01', self.train[idx])))
		image_original = image_original.type('torch.FloatTensor')
		image_noisy = image_noisy.type('torch.FloatTensor')

		return image_original, image_noisy

	# get a validation/testing pair in tensor form
	def getValItem(self, idx):
		image_original = torch.from_numpy(io.imread(os.path.join(self.root_dir, 'val_gray', self.val[idx])))
		image_noisy = torch.from_numpy(io.imread(os.path.join(self.root_dir, 'noisy_val_sigma01', self.val[idx])))
		image_original = image_original.type('torch.FloatTensor')
		image_noisy = image_noisy.type('torch.FloatTensor')

		return image_original, image_noisy

	# get a batch of training data, either cropped or not
	def getTrainingDataBatch(self, start, size):
		images_orig = []
		images_noisy = []
		for i in range(start, start+size):
			image_original, image_noisy = self.getTrainItem(i)

			if USE_CROP:
				image_original, image_noisy = self.transformCrop(image_original, image_noisy, CROP_SIZE[0], CROP_SIZE[1])

			images_orig.append(image_original)
			images_noisy.append(image_noisy)

		images_orig = torch.stack(images_orig, 0)
		images_noisy = torch.stack(images_noisy, 0)
		images_orig = torch.unsqueeze(images_orig, 1)
		images_noisy = torch.unsqueeze(images_noisy, 1)

		return images_orig, images_noisy

	# crop two tensor images
	def transformCrop(self, image_original, image_noisy, th, tw):
		cols = IMAGE_SIZE[1]
		rows = IMAGE_SIZE[0]
		if cols == tw and rows == th:
			i = 0
			j = 0
		else:
			i = random.randint(0, rows - th)
			j = random.randint(0, cols - tw)

		image_original = torch.from_numpy(image_original.numpy()[i:i+th, j:j+tw]).type('torch.FloatTensor')
		image_noisy = torch.from_numpy(image_noisy.numpy()[i:i+th, j:j+tw]).type('torch.FloatTensor')

		return image_original, image_noisy

	# get a batch of validation data
	def getValidationDataBatch(self, start, size):
                images_orig = []
                images_noisy = []
                for i in range(start, start+size):
                        image_original, image_noisy = self.getValItem(i)

                        images_orig.append(image_original)
                        images_noisy.append(image_noisy)

                images_orig = torch.stack(images_orig, 0)
                images_noisy = torch.stack(images_noisy, 0)
                images_orig = torch.unsqueeze(images_orig, 1)
                images_noisy = torch.unsqueeze(images_noisy, 1)

                return images_orig, images_noisy

    # get the length of the training data
	def trainingLength(self):
		return len(self.train)

	# get the length of the validation data
	def validationLength(self):
		return len(self.val)

	# get validation filenames
	def validationFiles(self):
		return self.val

# Load 2D UNet model which has already been trained
if USE_GPU:
	UNet2D = torch.load('models/UNet_model.pt')
	UNet2D = UNet2D.cuda()
else:
	UNet2D = torch.load('models/UNet_model.pt', map_location='cpu')

# keep dictionary of modules to replace
modules = {}

# iterate through layers
for name, layer in UNet2D.named_modules():
	if isinstance(layer, nn.Conv2d):
		# code based off of https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning

		# perform CP decomposition on layer if it is a convolutional layer
		last, first, vertical, horizontal = d.parafac(layer.weight.data.cpu().numpy(), rank=constants.rank, init='svd')

		# take input pointwise layer and set up as pointwise conv2d with 1x1 kernel
		pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
	            out_channels=first.shape[1], kernel_size=1, stride=1, padding=0, 
	            dilation=layer.dilation, bias=False)

		# implement depthwise vertical layer as a grouped convolution, preserve the stride of the original convolution
		depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1], 
	            out_channels=vertical.shape[1], kernel_size=(vertical.shape[0], 1),
	            stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation,
	            groups=vertical.shape[1], bias=False)

		# implement depthwise horizontal layer as a grouped convolution, preserve the stride of the original convolution
		depthwise_horizontal_layer = \
	        torch.nn.Conv2d(in_channels=horizontal.shape[1], \
	            out_channels=horizontal.shape[1], 
	            kernel_size=(1, horizontal.shape[0]), stride=layer.stride,
	            padding=(0, layer.padding[0]), 
	            dilation=layer.dilation, groups=horizontal.shape[1], bias=False)

	    # iplement pointwise layer as conv2d with 1x1 kernel also
		pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
	            out_channels=last.shape[0], kernel_size=1, stride=1,
	            padding=0, dilation=layer.dilation, bias=False)

		# convert if using GPU
		if USE_GPU:
			horizontal = torch.from_numpy(horizontal).type('torch.FloatTensor').cuda()
			vertical = torch.from_numpy(vertical).type('torch.FloatTensor').cuda()
			first = torch.from_numpy(first).type('torch.FloatTensor').cuda()
			last = torch.from_numpy(last).type('torch.FloatTensor').cuda()
		else:
			horizontal = torch.from_numpy(horizontal).type('torch.FloatTensor')
			vertical = torch.from_numpy(vertical).type('torch.FloatTensor')
			first = torch.from_numpy(first).type('torch.FloatTensor')
			last = torch.from_numpy(last).type('torch.FloatTensor')

		# initialize weights of the layers constructed with the tensors obtained from the CP decomposition
		depthwise_horizontal_layer.weight.data = \
	        torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
		depthwise_vertical_layer.weight.data = \
	        torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
		pointwise_s_to_r_layer.weight.data = \
	        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
		pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

		# new layers to replace the existing layer
		new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, \
	                    depthwise_horizontal_layer, pointwise_r_to_t_layer]
	    
	    # update the module reference in this named layer
		modules[name] = nn.Sequential(*new_layers)

# replace the modules with "name" with what is in the dictionary, based on code found at
# https://discuss.pytorch.org/t/change-all-conv2d-and-batchnorm2d-to-their-3d-counterpart/24780
for name in modules:
    parent_module = UNet2D
    objs = name.split(".")
    if len(objs) == 1:
        UNet2D.__setattr__(name, modules[name])
        continue

    for obj in objs[:-1]:
        parent_module = parent_module.__getattr__(obj)

    parent_module.__setattr__(objs[-1], modules[name])

if USE_GPU:
	UNet2D = UNet2D.cuda()
else:
	UNet2D = UNet2D.cpu()

# Create dataset representation
d = Dataset('../dataset/images')

# Get validation data to test on
valDataY, valDataX = d.getValidationDataBatch(0,50)
if USE_GPU:
	valDataY = valDataY.cuda()
	valDataX = valDataX.cuda()

# save model, get output for validation data
torch.save(UNet2D, 'CP_decomposed_model.pt')

output = UNet2D(valDataX)

# write this data to be post-processed for MSE/PSNR values
output = torch.squeeze(output)
if USE_GPU:
	output = output.cpu()

# save output images and filenames
outputIm = output.detach().numpy()
sio.savemat('outputIm_CP.mat', {'outputIm': outputIm})
val = d.validationFiles()
sio.savemat('filenames_CP.mat', {'filenames': val})
