import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils
import torch.optim as optim

import numpy as np

import math
import numbers

# UNet models
import pytorch_prototyping

from skimage import io
import os

# hacky method since running in virtual environment on my machine or X11 forwarding to view images through SSH
import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('GTK')
import matplotlib.pyplot as plt

import psutil


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

	def getTrainingData(self):
		images_orig = []
		images_noisy = []
		for i in range(0, len(self.train)):
			image_original, image_noisy = self.getTrainItem(i)

			images_orig.append(image_original)
			images_noisy.append(image_noisy)

		return images_orig, images_noisy

	def getValidationData(self):
                images_orig = []
                images_noisy = []
                for i in range(0, len(self.val)):
                        image_original, image_noisy = self.getValItem(i)

                        images_orig.append(image_original)
                        images_noisy.append(image_noisy)

                return images_orig, images_noisy

# Get 2D UNet model, single channel input, 16 features at top level, output single channel of denoised image
# params: input channels, output channels, number of top level features, down/up-samples, maximum features, use dropout
UNet2D = pytorch_prototyping.Unet(1, 1, 1, 2, 512, False)

# Define loss function
criterion = nn.MSELoss()

d = Dataset('../dataset/images')

trainingDataY, trainingDataX = d.getTrainingData()

trainingDataX = torch.stack(trainingDataX, 0)
trainingDataX = torch.unsqueeze(trainingDataX, 1)
trainingDataY = torch.stack(trainingDataY, 0)
trainingDataY = torch.unsqueeze(trainingDataY, 1)

optimizer = optim.SGD(UNet2D.parameters(), lr=0.01)
EPOCHS = 20
for i in range(0,EPOCHS):
	optimizer.zero_grad()
	output = UNet2D(trainingDataX)

	loss = criterion(output, trainingDataY)
	print(loss)
	loss.backward()
	optimizer.step()

valDataY, valDataX = d.getValidationData()
valDataX = torch.stack(valDataX, 0)
valDataX = torch.unsqueeze(valDataX, 1)
valDataY = torch.stack(valDataY, 0)
valDataY = torch.unsqueeze(valDataY, 1)

output = UNet2D(valDataX)
#plt.imshow(trainingDataX[1,0,:,:].numpy())
#plt.show()
#plt.imshow(trainingDataY[1,0,:,:].numpy())
#plt.show()
#plt.imshow(output[1,0,:,:].detach().numpy())
#plt.show()

#process = psutil.Process(os.getpid())
#print(process.memory_info().rss)

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(valDataX[1,0,:,:].numpy())
axarr[0,1].imshow(valDataY[1,0,:,:].numpy())
axarr[1,0].imshow(output[1,0,:,:].detach().numpy())
plt.show()
