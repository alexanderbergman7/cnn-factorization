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
import pytorch_prototyping

from skimage import io
import os

# hacky method since running in virtual environment on my machine or X11 forwarding to view images through SSH
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#import psutil
import scipy.io as sio

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

# hyperparameters
EPOCHS = constants.EPOCHS
BatchSize = constants.BatchSize
learning_rate = constants.learning_rate
input_channels = constants.input_channels
output_channels = constants.output_channels
highest_level_features = constants.highest_level_features
depth = constants.depth

# Get 2D UNet model, single channel input, output single channel, features at top level of denoised image
# params: input channels, output channels, number of top level features, down/up-samples, maximum features, use dropout
UNet2D = pytorch_prototyping.Unet(input_channels, output_channels, highest_level_features, depth, 512, False)
if USE_GPU:
	UNet2D = UNet2D.cuda()

# Define loss function
criterion = nn.MSELoss()

# Create dataset representation
d = Dataset('../dataset/images')

# iterate for epochs and batches and train using MSE loss function
optimizer = optim.SGD(UNet2D.parameters(), lr=learning_rate)
for i in range(0,EPOCHS):
	for j in range(0, int(d.trainingLength()/BatchSize)):
		batchY, batchX = d.getTrainingDataBatch(j*BatchSize, BatchSize)

		if USE_GPU:
			batchY = batchY.cuda()
			batchX = batchX.cuda()

		optimizer.zero_grad()
		output = UNet2D(batchX)

		loss = criterion(output, batchY)
		loss.backward()
		optimizer.step()

	print("Epoch " + str(i+1))

# Get validation data to test on
valDataY, valDataX = d.getValidationDataBatch(0,100)
if USE_GPU:
	valDataY = valDataY.cuda()
	valDataX = valDataX.cuda()

# save model, get output for validation data
torch.save(UNet2D, 'model.pt')
output = UNet2D(valDataX)

# write this data to be post-processed for MSE/PSNR values
output = torch.squeeze(output)
if USE_GPU:
	output = output.cpu()

# save output images and filenames
outputIm = output.detach().numpy()
sio.savemat('outputIm.mat', {'outputIm': outputIm})
val = d.validationFiles()
sio.savemat('filenames.mat', {'filenames': val})
