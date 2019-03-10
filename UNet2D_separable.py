import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils
import torch.optim as optim

import numpy as np

import math
import numbers

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

			images_orig.append(image_original)
			images_noisy.append(image_noisy)

		images_orig = torch.stack(images_orig, 0)
		images_noisy = torch.stack(images_noisy, 0)
		images_orig = torch.unsqueeze(images_orig, 1)
		images_noisy = torch.unsqueeze(images_noisy, 1)

		return images_orig, images_noisy

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

# hyperparameters
EPOCHS = 25
BatchSize = 40
learning_rate = 0.01
input_channels = 1
output_channels = 1
highest_level_features = 1
depth = 2

# Get 2D UNet model, single channel input, output single channel, features at top level of denoised image
# params: input channels, output channels, number of top level features, down/up-samples, maximum features, use dropout
UNet2D = separable_model_prototyping.Unet(input_channels, output_channels, highest_level_features, depth, 512, False)

# Define loss function
criterion = nn.MSELoss()

# Create dataset representation
d = Dataset('../dataset/images')

# iterate for epochs and batches and train using MSE loss function
optimizer = optim.SGD(UNet2D.parameters(), lr=learning_rate)
for i in range(0,EPOCHS):
	for j in range(0, int(d.trainingLength()/BatchSize)):
		batchY, batchX = d.getTrainingDataBatch(j*BatchSize, BatchSize)

		optimizer.zero_grad()
		output = UNet2D(batchX)

		loss = criterion(output, batchY)
		loss.backward()
		optimizer.step()

	print("Epoch " + str(i+1))

# Get validation data to test on
valDataY, valDataX = d.getValidationDataBatch(0,100)

# save model, get output for validation data
torch.save(UNet2D, 'E25_BS40_ic1_oc1_f1_d2_separable_noBNReLU/UNet2D_E25_BS40_ic1_oc1_f1_d2_separable_noBNReLU.pt')
output = UNet2D(valDataX)

# write this data to be post-processed for MSE/PSNR values
output = torch.squeeze(output)
outputIm = output.detach().numpy()
sio.savemat('E25_BS40_ic1_oc1_f1_d2_separable_noBNReLU/outputIm.mat', {'outputIm': outputIm})
val = d.validationFiles()
sio.savemat('E25_BS40_ic1_oc1_f1_d2_separable_noBNReLU/filenames.mat', {'filenames': val})

#process = psutil.Process(os.getpid())
#print(process.memory_info().rss)
