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

# Get 2D UNet model, single channel input, 16 features at top level, output single channel of denoised image
# params: input channels, output channels, number of top level features, down/up-samples, maximum features, use dropout
UNet2D = pytorch_prototyping.Unet(1, 1, 1, 2, 512, False)

# Define loss function
criterion = nn.MSELoss()

d = Dataset('../dataset/images')


print(d.getTrainItem(0))
exit()

optimizer = optim.SGD(UNet2D.parameters(), lr=0.01)
EPOCHS = 20
BatchSize = 40
for i in range(0,EPOCHS):
	for j in range(0, int(d.trainingLength()/BatchSize)):
		batchY, batchX = d.getTrainingDataBatch(j*BatchSize, BatchSize)

		optimizer.zero_grad()
		output = UNet2D(batchX)

		loss = criterion(output, batchY)
		print(loss)
		loss.backward()
		optimizer.step()

valDataY, valDataX = d.getValidationDataBatch(0,1)

output = UNet2D(valDataX)
#plt.imshow(trainingDataX[1,0,:,:].numpy())
#plt.show()
#plt.imshow(trainingDataY[1,0,:,:].numpy())
#plt.show()
#plt.imshow(output[1,0,:,:].detach().numpy())
#plt.show()

#process = psutil.Process(os.getpid())
#print(process.memory_info().rss)

#f, axarr = plt.subplots(2,2)
#axarr[0,0].imshow(valDataX[0,0,:,:].numpy())
#axarr[0,1].imshow(valDataY[0,0,:,:].numpy())
#axarr[1,0].imshow(output[0,0,:,:].detach().numpy())
#plt.show()

imageREC = torch.squeeze(output)
imageRECNP = imageREC.detach().numpy()
sio.savemat('irec.mat', {'x': imageRECNP})

#print(criterion(output, valDataX))
