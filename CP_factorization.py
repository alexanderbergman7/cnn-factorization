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

USE_GPU = True
USE_CROP = False

class Dataset():

	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.train = os.listdir(os.path.join(root_dir, 'train_gray'))
		self.val = os.listdir(os.path.join(root_dir, 'val_gray'))

	def getTrainItem(self, idx):
		image_original = torch.from_numpy(io.imread(os.path.join(self.root_dir, 'train_gray', self.train[idx])))
		image_noisy = torch.from_numpy(io.imread(os.path.join(self.root_dir, 'noisy_train_sigma01', self.train[idx])))
		image_original = image_original.type('torch.FloatTensor')
		image_noisy = image_noisy.type('torch.FloatTensor')

		return image_original, image_noisy

	def getValItem(self, idx):
		image_original = torch.from_numpy(io.imread(os.path.join(self.root_dir, 'val_gray', self.val[idx])))
		image_noisy = torch.from_numpy(io.imread(os.path.join(self.root_dir, 'noisy_val_sigma01', self.val[idx])))
		image_original = image_original.type('torch.FloatTensor')
		image_noisy = image_noisy.type('torch.FloatTensor')

		return image_original, image_noisy

	def getTrainingDataBatch(self, start, size):
		images_orig = []
		images_noisy = []
		for i in range(start, start+size):
			image_original, image_noisy = self.getTrainItem(i)

			if USE_CROP:
				image_original, image_noisy = self.transformCrop(image_original, image_noisy, 32, 32)

			images_orig.append(image_original)
			images_noisy.append(image_noisy)

		images_orig = torch.stack(images_orig, 0)
		images_noisy = torch.stack(images_noisy, 0)
		images_orig = torch.unsqueeze(images_orig, 1)
		images_noisy = torch.unsqueeze(images_noisy, 1)

		return images_orig, images_noisy

	def transformCrop(self, image_original, image_noisy, th, tw):
		cols = 480
		rows = 320
		if cols == tw and rows == th:
			i = 0
			j = 0
		else:
			i = random.randint(0, rows - th)
			j = random.randint(0, cols - tw)

		image_original = torch.from_numpy(image_original.numpy()[i:i+th, j:j+tw]).type('torch.FloatTensor')
		image_noisy = torch.from_numpy(image_noisy.numpy()[i:i+th, j:j+tw]).type('torch.FloatTensor')

		return image_original, image_noisy

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

	def trainingLength(self):
		return len(self.train)

	def validationLength(self):
		return len(self.val)

	def validationFiles(self):
		return self.val

# Get 2D UNet model, single channel input, output single channel, features at top level of denoised image
# params: input channels, output channels, number of top level features, down/up-samples, maximum features, use dropout
UNet2D = torch.load('final_model_UNet.pt')
if USE_GPU:
	UNet2D = UNet2D.cuda()

modules = {}
for name, layer in UNet2D.named_modules():
	if isinstance(layer, nn.Conv2d):
		last, first, vertical, horizontal = d.parafac(layer.weight.data.cpu().numpy(), rank=6, init='svd')

		pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
	            out_channels=first.shape[1], kernel_size=1, stride=1, padding=0, 
	            dilation=layer.dilation, bias=False)

		depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1], 
	            out_channels=vertical.shape[1], kernel_size=(vertical.shape[0], 1),
	            stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation,
	            groups=vertical.shape[1], bias=False)

		depthwise_horizontal_layer = \
	        torch.nn.Conv2d(in_channels=horizontal.shape[1], \
	            out_channels=horizontal.shape[1], 
	            kernel_size=(1, horizontal.shape[0]), stride=layer.stride,
	            padding=(0, layer.padding[0]), 
	            dilation=layer.dilation, groups=horizontal.shape[1], bias=False)

		pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
	            out_channels=last.shape[0], kernel_size=1, stride=1,
	            padding=0, dilation=layer.dilation, bias=True)

		horizontal = torch.from_numpy(horizontal).type('torch.FloatTensor').cuda()
		vertical = torch.from_numpy(vertical).type('torch.FloatTensor').cuda()
		first = torch.from_numpy(first).type('torch.FloatTensor').cuda()
		last = torch.from_numpy(last).type('torch.FloatTensor').cuda()

		depthwise_horizontal_layer.weight.data = \
	        torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
		depthwise_vertical_layer.weight.data = \
	        torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
		pointwise_s_to_r_layer.weight.data = \
	        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
		pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

		new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, \
	                    depthwise_horizontal_layer, pointwise_r_to_t_layer]
	    
		modules[name] = nn.Sequential(*new_layers)

for name in modules:
    parent_module = UNet2D
    objs = name.split(".")
    if len(objs) == 1:
        UNet2D.__setattr__(name, modules[name])
        continue

    for obj in objs[:-1]:
        parent_module = parent_module.__getattr__(obj)

    parent_module.__setattr__(objs[-1], modules[name])

UNet2D = UNet2D.cuda()

# Create dataset representation
d = Dataset('../dataset/images')

# Get validation data to test on
valDataY, valDataX = d.getValidationDataBatch(0,50)
if USE_GPU:
	valDataY = valDataY.cuda()
	valDataX = valDataX.cuda()

# save model, get output for validation data
torch.save(UNet2D, 'test_model_CP_rT.pt')

output = UNet2D(valDataX)
print(output.size())

# write this data to be post-processed for MSE/PSNR values
output = torch.squeeze(output)
if USE_GPU:
	output = output.cpu()

outputIm = output.detach().numpy()
sio.savemat('outputIm_CPr6.mat', {'outputIm': outputIm})
val = d.validationFiles()
sio.savemat('filenames_CPr1.mat', {'filenames': val})

#process = psutil.Process(os.getpid())
#print(process.memory_info().rss)