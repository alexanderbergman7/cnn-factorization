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

# hacky method since running in virtual environment on my machine
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt

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

# Get 2D UNet model, single channel input, 16 features at top level, output single channel of denoised image
# params: input channels, output channels, number of top level features, down/up-samples, maximum features, use dropout
UNet2D = pytorch_prototyping.Unet(1, 1, 8, 2, 512, False)

# Define loss function
criterion = nn.MSELoss()

d = Dataset('../dataset/images')

trainingDataY, trainingDataX = d.getTrainingData()

trainingDataX = torch.stack(trainingDataX, 0)
trainingDataX = torch.unsqueeze(trainingDataX, 1)
trainingDataY = torch.stack(trainingDataY, 0)
trainingDataY = torch.unsqueeze(trainingDataY, 1)

#output = UNet2D(trainingDataX)
#print(output.size())
#print(trainingDataY.size())

#image, image_noisy = d.getTrainItem()
#plt.imshow(image)
#plt.show()

#output = UNet2D(image2)

loss = criterion(trainingDataX, trainingDataY)
print(loss)
#UNet2D.zero_grad()
#loss.backward()
